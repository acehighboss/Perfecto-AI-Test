from __future__ import annotations

import io
import os
import re
import asyncio
import json
import time
import tempfile
import pathlib
import typing as t
import traceback
from urllib.parse import urlparse, urljoin, urlsplit
import urllib.robotparser as urp
import tldextract
from collections import defaultdict

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# -----------------------------
# Optional parsers
# -----------------------------
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

try:
    from readability import Document as ReadabilityDoc
    READABILITY_AVAILABLE = True
except ImportError:
    ReadabilityDoc = None
    READABILITY_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:
    pdf_extract_text = None

try:
    import docx
except ImportError:
    docx = None

from llama_parse import LlamaParse
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


# -----------------------------------
# NEW: Async Web Crawler Components
# -----------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 25
ROBOTS_CACHE = {}

def is_http_url(url: str) -> bool:
    return url.strip().startswith(("http://", "https://"))

def normalize_url(base: str, href: str) -> t.Optional[str]:
    if not href:
        return None
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    try:
        url = urljoin(base, href)
        parts = list(urlsplit(url))
        parts[4] = ""  # fragment 제거
        return urlunsplit(parts)
    except Exception:
        return None

async def get_robots(session: aiohttp.ClientSession, base_url: str) -> urp.RobotFileParser:
    origin = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    robots_url = urljoin(origin, "/robots.txt")
    if robots_url in ROBOTS_CACHE:
        return ROBOTS_CACHE[robots_url]
    
    rp = urp.RobotFileParser()
    rp.set_url(robots_url)
    try:
        async with session.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT) as r:
            if r.status == 200:
                text = await r.text()
                rp.parse(text.splitlines())
            else:
                rp.allow_all = True # 실패 시 허용
    except Exception:
        rp.allow_all = True # 실패 시 허용
    
    ROBOTS_CACHE[robots_url] = rp
    return rp

def extract_main_content(html: str, url: str | None = None) -> tuple[str, str]:
    """ (title, main_text) 추출 """
    title, text = "", ""
    
    if TRAFILATURA_AVAILABLE:
        try:
            res = trafilatura.extract(
                html, url=url, include_comments=False, include_tables=False,
                favor_recall=True, output_format="json"
            )
            if res:
                data = json.loads(res)
                title = (data.get("title") or "").strip()
                text = (data.get("text") or "").strip()
                if text: return title, text
        except Exception: pass

    if READABILITY_AVAILABLE:
        try:
            doc = ReadabilityDoc(html)
            title = (doc.short_title() or "").strip()
            article_html = doc.summary()
            soup = BeautifulSoup(article_html, "html.parser")
            text = re.sub(r"\s+", " ", soup.get_text(strip=True)).strip()
            if text: return title, text
        except Exception: pass

    # Fallback
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = re.sub(r"\s+", " ", soup.get_text(strip=True)).strip()
    return title, text

async def render_page_with_playwright(url: str) -> str:
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright is not installed.")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await page.wait_for_timeout(1500)
            html = await page.content()
        finally:
            await context.close()
            await browser.close()
    return html

def should_render_js_auto(raw_html: str | None) -> bool:
    if not raw_html: return True
    text_like = re.sub(r"<[^>]+>", "", raw_html)
    return len(text_like.strip()) < 200 or raw_html.count("<script") > 5

async def process_url(session: aiohttp.ClientSession, url: str, use_js_render: bool) -> Document | None:
    try:
        rp = await get_robots(session, url)
        if not rp.can_fetch(USER_AGENT, url):
            print(f"[CRAWL] Blocked by robots.txt: {url}")
            return None

        # 1. 일반 GET 요청
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        async with session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True) as resp:
            if resp.status != 200 or "text/html" not in (resp.headers.get("Content-Type") or ""):
                return None
            html = await resp.text(errors="ignore")

        # 2. 필요 시 Playwright로 JS 렌더링
        if use_js_render and should_render_js_auto(html):
            print(f"[CRAWL] Rendering JS for: {url}")
            try:
                html = await render_page_with_playwright(url)
            except Exception as e:
                print(f"[CRAWL] Playwright rendering failed for {url}: {e}")
                # 렌더링 실패 시 원본 HTML로 계속 진행

        # 3. 본문 및 제목 추출
        title, main_text = extract_main_content(html, url)
        if main_text and len(main_text) >= 100:
            return Document(page_content=main_text, metadata={"source": url, "title": title or url})

    except Exception as e:
        print(f"[CRAWL] Failed to process URL {url}: {e}")
        # traceback.print_exc() # 디버깅 시 사용
    return None

async def _get_documents_from_urls_async(urls: t.List[str], use_js_render: bool) -> t.List[Document]:
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(session, url, use_js_render) for url in urls if is_http_url(url)]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc]

def get_documents_from_urls_robust(urls: t.List[str], use_js_render: bool = True) -> t.List[Document]:
    """ Robust URL ingestion using async crawling and optional JS rendering. """
    if not urls:
        return []
    # Streamlit 같은 동기 환경에서 비동기 함수 실행
    return asyncio.run(_get_documents_from_urls_async(urls, use_js_render))


# -----------------------------------
# File Processing (기존 코드 유지 및 개선)
# -----------------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _read_bytes_to_text(data: bytes) -> str:
    try:
        import charset_normalizer
        result = charset_normalizer.from_bytes(data).best()
        enc = (result.encoding if result else "utf-8") or "utf-8"
        return data.decode(enc, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def _extract_pdf_with_llamaparse(data: bytes) -> str:
    if not LlamaParse or not LLAMA_CLOUD_API_KEY: return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="text")
        docs = parser.load_data(tmp_path)
        return "\n\n".join([d.text for d in docs]).strip()
    except Exception:
        return ""
    finally:
        os.remove(tmp_path)

def _extract_from_pdf_bytes_local(data: bytes) -> str:
    if not pdf_extract_text: return ""
    try:
        return _clean_text(pdf_extract_text(io.BytesIO(data)))
    except Exception: return ""

def _extract_from_docx_bytes(data: bytes) -> str:
    if not docx: return ""
    try:
        doc = docx.Document(io.BytesIO(data))
        return _clean_text("\n".join([p.text for p in doc.paragraphs if p.text.strip()]))
    except Exception: return ""

def get_documents_from_uploaded_files(files: t.List, min_chars: int = 100) -> t.List[Document]:
    docs: t.List[Document] = []
    if not files:
        return docs

    for uf in files:
        try:
            name = getattr(uf, "name", "uploaded")
            suffix = (name.split(".")[-1] if "." in name else "").lower()
            raw = uf.read()
            text = ""

            if suffix == "pdf":
                text = _extract_pdf_with_llamaparse(raw) or _extract_from_pdf_bytes_local(raw)
            elif suffix == "docx":
                text = _extract_from_docx_bytes(raw)
            else: # txt, md, csv, log, json etc.
                text = _clean_text(_read_bytes_to_text(raw))

            if text and len(text) >= min_chars:
                docs.append(Document(page_content=text, metadata={"source": f"file:{name}", "title": name, "filename": name}))
        except Exception:
            continue
    return docs

from __future__ import annotations

import io
import os
import re
import asyncio
import tempfile
import pathlib
import typing as t
from urllib.parse import urlparse
from urllib import robotparser

import httpx
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from langchain_core.documents import Document

# --- (이하 라이브러리 임포트는 이전과 동일) ---
try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    from readability import Document as ReadabilityDoc
except Exception:
    ReadabilityDoc = None
try:
    from playwright.async_api import async_playwright # 비동기 버전으로 변경
except Exception:
    async_playwright = None # 비동기 버전으로 변경
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None
try:
    import docx
except Exception:
    docx = None
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko,ko-KR;q=0.9,en;q=0.8",
}

def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u.strip())
        return p.scheme in {"http", "https"} and bool(p.netloc)
    except Exception:
        return False

def _clean_text(text: str) -> str:
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()

# --- (HTML 및 파일 처리 함수들은 이전과 거의 동일) ---
def extract_readable_text(html: str, url: str) -> t.Tuple[str, str]:
    # ... (기존 extract_readable_text 함수 내용)
    text, title = "", ""
    try:
        text, title = trafilatura.extract(html, url=url), trafilatura.extract_metadata(html).title
    except: pass
    if len(text) < 200:
        try:
            doc = ReadabilityDoc(html)
            text = doc.summary()
            title = title or doc.short_title()
        except: pass
    return _clean_text(text), title

# ★★★ Playwright 로직을 비동기(async)로 수정 ★★★
async def render_with_playwright_async(url: str) -> str:
    if not async_playwright:
        return ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])
            await page.goto(url, wait_until="networkidle", timeout=20000)
            await page.wait_for_timeout(1500) # 추가 대기
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        print(f"Playwright rendering failed for {url}: {e}")
        return ""

# ★★★ URL 처리를 위한 핵심 비동기 함수 ★★★
async def process_url(
    client: httpx.AsyncClient,
    url: str,
    respect_robots: bool,
    use_js_render: bool,
    js_only_when_needed: bool,
    min_chars: int
) -> Document | None:
    if respect_robots:
        # (간략화된 버전, 실제로는 robots.txt 파싱 필요)
        pass

    try:
        # 1. 일반 GET 요청
        resp = await client.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()
        
        # PDF 처리
        if "pdf" in content_type or url.endswith(".pdf"):
            # (PDF 처리 로직은 생략, 기존과 유사)
            return None

        # HTML 처리
        html = await resp.aread()
        encoding = from_bytes(html).best().encoding
        html_str = html.decode(encoding, errors="replace")

        extracted_text, title = extract_readable_text(html_str, url)

        # 2. 필요 시 JS 렌더링
        if use_js_render and (not js_only_when_needed or len(extracted_text) < min_chars):
            rendered_html = await render_with_playwright_async(url)
            if rendered_html:
                t2, tt2 = extract_readable_text(rendered_html, url)
                if len(t2) > len(extracted_text):
                    extracted_text, title = t2, tt2 or title

        if len(extracted_text) >= min_chars:
            return Document(page_content=extracted_text, metadata={"source": url, "title": title or url})

    except Exception as e:
        print(f"Failed to process URL {url}: {e}")
    return None


# ★★★ Public API를 비동기 방식으로 수정 ★★★
async def get_documents_from_urls_async(
    urls: t.List[str], **kwargs
) -> t.List[Document]:
    
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, verify=False) as client:
        tasks = [process_url(client, url, **kwargs) for url in urls if _is_valid_url(url)]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc]

# Streamlit과 같은 동기 환경에서 비동기 함수를 실행하기 위한 래퍼
def get_documents_from_urls_robust(
    urls: t.List[str], **kwargs
) -> t.List[Document]:
    return asyncio.run(get_documents_from_urls_async(urls, **kwargs))

# --- (get_documents_from_uploaded_files 함수는 기존과 동일) ---
def get_documents_from_uploaded_files(files: t.List, **kwargs) -> t.List[Document]:
    # ... (기존 파일 처리 로직)
    docs = []
    # ...
    return docs

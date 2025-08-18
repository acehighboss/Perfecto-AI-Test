from __future__ import annotations

import io
import os
import re
import time
import tempfile
import pathlib
import typing as t
from urllib.parse import urlparse
from urllib import robotparser

import httpx
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from langchain_core.documents import Document

# -----------------------------
# Optional parsers (graceful)
# -----------------------------
# HTML content extractors
try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from readability import Document as ReadabilityDoc
except Exception:
    ReadabilityDoc = None

try:
    from boilerpy3 import extractors as bp_extractors
except Exception:
    bp_extractors = None

try:
    from newspaper import fulltext as newspaper_fulltext
except Exception:
    newspaper_fulltext = None

# PDF extractors
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# DOCX
try:
    import docx  # python-docx
except Exception:
    docx = None

# Playwright (JS rendering)
try:
    from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError
except Exception:
    sync_playwright = None

# LlamaParse (for PDF high-fidelity parsing)
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


# -----------------------------
# Networking & heuristics
# -----------------------------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko,ko-KR;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}
VALID_SCHEMES = {"http", "https"}


def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u.strip())
        return p.scheme in VALID_SCHEMES and bool(p.netloc)
    except Exception:
        return False


def _robots_allowed(url: str, user_agent: str = DEFAULT_HEADERS["User-Agent"]) -> bool:
    try:
        p = urlparse(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # robots 접근 실패 시 관용적으로 허용
        return True


def _http_fetch(url: str, timeout: float = 20.0, max_retries: int = 2) -> httpx.Response:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
                # 일부 서버는 HEAD 미지원이므로 예외 무시
                try:
                    client.head(url)
                except Exception:
                    pass
                resp = client.get(url)
                resp.raise_for_status()
                return resp
        except Exception as e:
            last_exc = e
            time.sleep(0.3 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("HTTP fetch failed")


def _detect_encoding(content: bytes) -> str:
    try:
        result = from_bytes(content).best()
        return (result.encoding if result else "utf-8") or "utf-8"
    except Exception:
        return "utf-8"


def _looks_like_pdf(resp: httpx.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    return "application/pdf" in ct or url.lower().endswith(".pdf")


def _clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _title_from_html(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        return og["content"].strip()
    return ""


# -----------------------------
# HTML extraction pipeline
# -----------------------------
def _extract_with_trafilatura(html: str, url: str) -> t.Tuple[str, str]:
    if not trafilatura:
        return "", ""
    try:
        text = trafilatura.extract(html, include_links=False, include_comments=False, url=url) or ""
        meta = trafilatura.extract_metadata(html)
        title = meta.title if meta else ""
        return _clean_text(text), (title or "")
    except Exception:
        return "", ""


def _extract_with_readability(html: str) -> t.Tuple[str, str]:
    if not ReadabilityDoc:
        return "", ""
    try:
        doc = ReadabilityDoc(html)
        title = doc.short_title() or ""
        article_html = doc.summary()
        soup = BeautifulSoup(article_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return _clean_text(text), title
    except Exception:
        return "", ""


def _extract_with_boilerpy(html: str) -> t.Tuple[str, str]:
    if not bp_extractors:
        return "", ""
    try:
        extractor = bp_extractors.ArticleExtractor()
        text = extractor.get_content(html) or ""
        return _clean_text(text), ""
    except Exception:
        return "", ""


def _extract_with_newspaper(html: str, url: str) -> t.Tuple[str, str]:
    if not newspaper_fulltext:
        return "", ""
    try:
        text = newspaper_fulltext(html, url=url) or ""
        return _clean_text(text), ""
    except Exception:
        return "", ""


def _extract_with_bs_heuristic(html: str) -> t.Tuple[str, str]:
    try:
        soup = BeautifulSoup(html, "html.parser")
        title = _title_from_html(soup)
        selectors = [
            "article", "main", "[role=main]",
            ".entry-content", ".post-content", ".article-content",
            "#content", ".content", ".wiki", "#mw-content-text",
        ]
        candidates = []
        for sel in selectors:
            node = soup.select_one(sel)
            if node and node.get_text(strip=True):
                candidates.append(node.get_text(separator="\n", strip=True))
        if not candidates:
            ps = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join([p for p in ps if len(p) > 0])
        else:
            text = max(candidates, key=len)
        return _clean_text(text), title
    except Exception:
        return "", ""


def extract_readable_text(html: str, url: str) -> t.Tuple[str, str]:
    """
    Multi-extractor pipeline for static HTML.
    """
    # 1
    text, title = _extract_with_trafilatura(html, url)
    if len(text) >= 400:
        return text, _extract_title_fallback(html, title)

    # 2
    t2, tt2 = _extract_with_readability(html)
    if len(t2) > len(text):
        text, title = t2, (title or tt2)
    if len(text) >= 400:
        return text, _extract_title_fallback(html, title)

    # 3
    t3, tt3 = _extract_with_boilerpy(html)
    if len(t3) > len(text):
        text, title = t3, (title or tt3)
    if len(text) >= 400:
        return text, _extract_title_fallback(html, title)

    # 4
    t4, tt4 = _extract_with_newspaper(html, url)
    if len(t4) > len(text):
        text, title = t4, (title or tt4)
    if len(text) >= 400:
        return text, _extract_title_fallback(html, title)

    # 5
    t5, tt5 = _extract_with_bs_heuristic(html)
    if len(t5) > len(text):
        text, title = t5, (title or tt5)

    return text, _extract_title_fallback(html, title)


def _extract_title_fallback(html: str, title_now: str) -> str:
    if title_now:
        return title_now
    try:
        soup = BeautifulSoup(html, "html.parser")
        return _title_from_html(soup)
    except Exception:
        return title_now

# ---------------------------------------------
# ADVANCED PLAYWRIGHT RENDERING & EXTRACTION
# ---------------------------------------------

JS_PRE_CLEAN_PAGE = """
() => {
    const kill = (sel) => document.querySelectorAll(sel).forEach(el => el.remove());

    // Kill common noise elements
    kill('header, nav, aside, footer');
    kill('.sidebar, .side, .widget, .trending, .related, .recommend, .recommended');
    kill('.ad, .ads, [class*="ad-"], [id*="ad"], .banner, .popup, .modal, .newsletter, .subscribe');
    kill('.share, .sns, .social, .breadcrumb, .cookie, .toast, .sticky, .floating');
    kill('#comments, .comments, [id*="comment"]');

    // Kill accessibility skip links
    kill('a.skip-link, a[href="#content"], a[href="#main"]');

    // Kill content-blocking overlays
    kill('[style*="position: fixed"][style*="z-index:"][style*="background:"]');

    // Heuristically remove recommendation blocks
    Array.from(document.querySelectorAll('section, div')).forEach(el => {
      const t = (el.textContent || '').trim();
      if (!t) return;
      const first = (t.split('\\n')[0] || '').trim();
      if (t.startsWith('추천 콘텐츠')) el.remove();
      if (/^관련(기사|글)|^많이 본|^인기/.test(first)) el.remove();
    });
}
"""

JS_CLICK_READ_MORE = """
async () => {
    const selectors = ['button', 'a'];
    const keywords = ['더보기', '더 보기', 'read more', 'load more', 'continue reading'];
    for (const el of document.querySelectorAll(selectors.join(','))) {
        const t = (el.innerText || '').trim().toLowerCase();
        if (t && keywords.some(k => t.includes(k))) {
            el.click();
            await new Promise(r => setTimeout(r, 300)); // wait a bit after click
        }
    }
}
"""

JS_AUTO_SCROLL = """
async (duration) => {
    const start = Date.now();
    const step = () => {
        window.scrollBy(0, 250);
        if (Date.now() - start < duration) requestAnimationFrame(step);
    };
    step();
}
"""

def _interact_and_clean_page(page: "Page"):
    """Run a sequence of interactions and cleanups on the page."""
    try:
        page.evaluate(JS_PRE_CLEAN_PAGE)
        time.sleep(0.2)
        page.evaluate(JS_CLICK_READ_MORE)
        time.sleep(0.5)
        page.evaluate(JS_AUTO_SCROLL, 1500)
        time.sleep(1.5)
        page.evaluate(JS_PRE_CLEAN_PAGE) # Clean again after interactions
    except Exception:
        # Errors during interaction are not fatal
        pass

def render_with_playwright_robust(
    url: str,
    wait_until: str = "domcontentloaded",
    wait_ms: int = 2000,
    timeout_ms: int = 25000,
    extra_ua: str | None = None,
) -> str:
    """
    Advanced JS rendering with pre-cleaning and interaction.
    """
    if not sync_playwright:
        return ""
    
    content = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=extra_ua or DEFAULT_HEADERS["User-Agent"],
                # Block assets for speed
                java_script_enabled=True,
                route_intercept_handler=lambda route: route.abort() if route.request.resource_type in {
                    'image', 'stylesheet', 'font', 'media', 'manifest'
                } else route.continue_(),
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            
            try:
                page.goto(url, wait_until=wait_until)
            except PlaywrightTimeoutError:
                # Fallback to 'load' on timeout
                page.goto(url, wait_until="load")

            _interact_and_clean_page(page)
            
            # Final wait to ensure scripts have run
            page.wait_for_timeout(wait_ms)
            
            content = page.content()
            browser.close()
    except Exception as e:
        print(f"Playwright rendering failed for {url}: {e}")
        return ""
        
    return content or ""


# -----------------------------
# PDF extraction via LlamaParse
# -----------------------------
def _extract_pdf_with_llamaparse_from_path(path: str) -> str:
    if not LlamaParse or not LLAMA_CLOUD_API_KEY:
        return ""
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        max_timeout=600,
    )
    try:
        docs = parser.load_data(path)
        texts = [getattr(d, "text", "") or "" for d in docs if getattr(d, "text", "")]
        return "\n\n".join(texts).strip()
    except Exception:
        return ""


def _extract_pdf_with_llamaparse_from_bytes(data: bytes) -> str:
    if not LlamaParse or not LLAMA_CLOUD_API_KEY:
        return ""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "uploaded.pdf"
        p.write_bytes(data)
        return _extract_pdf_with_llamaparse_from_path(str(p))


# -----------------------------
# Local fallbacks (files)
# -----------------------------
def _normalize_text(text: str) -> str:
    return _clean_text(text)


def _read_bytes_to_text(data: bytes) -> str:
    try:
        result = from_bytes(data).best()
        enc = (result.encoding if result else "utf-8") or "utf-8"
        return data.decode(enc, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _extract_from_pdf_bytes_local(data: bytes) -> str:
    if not pdf_extract_text:
        return ""
    try:
        bio = io.BytesIO(data)
        txt = pdf_extract_text(bio) or ""
        return _normalize_text(txt)
    except Exception:
        return ""


def _extract_from_docx_bytes(data: bytes) -> str:
    if not docx:
        return ""
    try:
        bio = io.BytesIO(data)
        d = docx.Document(bio)
        paras = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        txt = "\n".join(paras)
        return _normalize_text(txt)
    except Exception:
        return ""


# -----------------------------
# Public API: URL → Documents
# -----------------------------
def get_documents_from_urls_robust(
    urls: t.List[str],
    *,
    respect_robots: bool = True,
    min_chars: int = 200,
    per_domain_delay: float = 0.2,
    use_js_render: bool = False,
    js_only_when_needed: bool = True,
) -> t.List[Document]:
    docs: t.List[Document] = []
    last_domain_ts: dict[str, float] = {}

    for raw in urls:
        url = raw.strip()
        if not url or not _is_valid_url(url):
            continue

        if respect_robots and not _robots_allowed(url):
            continue

        netloc = urlparse(url).netloc
        now = time.time()
        prev = last_domain_ts.get(netloc, 0.0)
        if now - prev < per_domain_delay:
            time.sleep(per_domain_delay - (now - prev))
        last_domain_ts[netloc] = time.time()

        try:
            resp = _http_fetch(url)
        except Exception:
            resp = None

        if resp and _looks_like_pdf(resp, url):
            text = ""
            if resp.content:
                text = _extract_pdf_with_llamaparse_from_bytes(resp.content) or _extract_from_pdf_bytes_local(resp.content)
            if text and len(text) >= min_chars:
                docs.append(Document(page_content=text, metadata={"source": url, "title": url}))
            continue

        html, title, extracted_text = "", "", ""
        if resp is not None:
            enc = _detect_encoding(resp.content)
            try:
                html = resp.content.decode(enc, errors="replace")
            except Exception:
                html = resp.text
            extracted_text, title = extract_readable_text(html, url)
        
        should_render_js = use_js_render and (not js_only_when_needed or len(extracted_text) < min_chars)
        
        if should_render_js:
            rendered_html = render_with_playwright_robust(url)
            if rendered_html:
                t2, tt2 = extract_readable_text(rendered_html, url)
                if len(t2) > len(extracted_text):
                    extracted_text, title = t2, (title or tt2)

        if extracted_text and len(extracted_text) >= min_chars:
            docs.append(Document(page_content=extracted_text, metadata={"source": url, "title": title or url}))

    return docs


# -----------------------------
# Public API: Files → Documents
# -----------------------------
def get_documents_from_uploaded_files(
    files: t.List,
    *,
    min_chars: int = 100,
) -> t.List[Document]:
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
                text = _extract_pdf_with_llamaparse_from_bytes(raw) or _extract_from_pdf_bytes_local(raw)
            elif suffix in {"docx"}:
                text = _extract_from_docx_bytes(raw)
            else:
                text = _normalize_text(_read_bytes_to_text(raw))

            if text and len(text) >= min_chars:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": f"uploaded:{name}",
                            "title": name,
                            "filename": name,
                            "kind": "file",
                        },
                    )
                )
        except Exception:
            continue

    return docs

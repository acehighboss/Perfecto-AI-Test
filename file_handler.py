# file_handler.py

from __future__ import annotations
import typing as t
import re
import time
import io
from urllib.parse import urlparse
import httpx
from charset_normalizer import from_bytes
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from urllib import robotparser

# 옵션 파서들
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

# PDF
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# Playwright 존재 여부 확인
try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

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
        # 타임아웃/에러 무시하고 관용적으로 허용하는 쪽으로
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        # robots.txt에 접근 실패 시 과도한 차단 피하려면 True/False 선택 가능
        return True

def _http_fetch(url: str, timeout: float = 20.0, max_retries: int = 2) -> httpx.Response:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
                # HEAD로 컨텐츠 타입/길이 빠르게 체크(일부 서버는 HEAD 미지원)
                try:
                    head = client.head(url)
                    # 일부 서버가 405 주면 무시하고 GET 진행
                except Exception:
                    pass
                resp = client.get(url)
                resp.raise_for_status()
                return resp
        except Exception as e:
            last_exc = e
            # 소폭 백오프
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
    if "application/pdf" in ct:
        return True
    return url.lower().endswith(".pdf")

def _extract_pdf(resp: httpx.Response) -> str:
    if not pdf_extract_text:
        return ""
    try:
        bio = io.BytesIO(resp.content)
        txt = pdf_extract_text(bio) or ""
        return _clean_text(txt)
    except Exception:
        return ""

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

def _extract_with_trafilatura(html: str, url: str) -> t.Tuple[str, str]:
    if not trafilatura:
        return "", ""
    try:
        # trafilatura는 html(str) 보다 URL을 직접 주면 내부 fetch를 다시 하기도 함
        # 여기서는 이미 받은 html로 처리
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
        # CMS별 흔한 본문 컨테이너 우선 순위
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
            # 폴백: 가장 긴 <p>들의 합
            ps = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            text = "\n\n".join([p for p in ps if len(p) > 0])
        else:
            text = max(candidates, key=len)
        return _clean_text(text), title
    except Exception:
        return "", ""

def _extract_title_fallback(html: str, title_now: str) -> str:
    if title_now:
        return title_now
    try:
        soup = BeautifulSoup(html, "html.parser")
        return _title_from_html(soup)
    except Exception:
        return title_now

def extract_readable_text(html: str, url: str) -> t.Tuple[str, str]:
    """
    다중 파서 파이프라인:
    1) trafilatura
    2) readability-lxml
    3) boilerpy3
    4) newspaper3k
    5) BeautifulSoup 휴리스틱
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

# Playwright 렌더링
def render_with_playwright(
    url: str,
    wait_until: str = "networkidle",   # "load" | "domcontentloaded" | "networkidle"
    wait_ms: int = 2500,
    timeout_ms: int = 20000,
    extra_ua: str | None = None,
) -> str:
    """
    간단한 JS 렌더링. 페이지 로드 후 약간 대기한 뒤 최종 HTML을 반환.
    """
    if not sync_playwright:
        return ""

    content = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=extra_ua or DEFAULT_HEADERS["User-Agent"])
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until=wait_until)
            # 약간 더 대기(클라이언트 렌더/추가 요청)
            page.wait_for_timeout(wait_ms)
            content = page.content()
            browser.close()
    except Exception:
        return ""
    return content or ""

def get_documents_from_urls_robust(
    urls: t.List[str],
    *,
    respect_robots: bool = True,
    min_chars: int = 200,
    per_domain_delay: float = 0.2,
    use_js_render: bool = False,          # JS 렌더링 토글
    js_only_when_needed: bool = True,     # 정적 추출이 짧을 때만 JS 사용
) -> t.List[Document]:
    """
    - robots.txt 허용 시에만 수집(respect_robots=True)
    - PDF 자동 처리
    - 다중 파서로 본문 추출
    - 필요 시 Playwright로 JS 렌더링 후 재시도
    """
    docs: t.List[Document] = []
    last_domain_ts: dict[str, float] = {}

    for raw in urls:
        url = raw.strip()
        if not url or not _is_valid_url(url):
            continue

        # robots.txt 체크
        if respect_robots and not _robots_allowed(url):
            # 허용되지 않으면 스킵
            continue

        # 간단한 도메인 레이트리밋
        netloc = urlparse(url).netloc
        now = time.time()
        prev = last_domain_ts.get(netloc, 0)
        if now - prev < per_domain_delay:
            time.sleep(per_domain_delay - (now - prev))
        last_domain_ts[netloc] = time.time()

        # 1) HTTP로 먼저 시도
        try:
            resp = _http_fetch(url)
        except Exception:
            resp = None

        html = ""
        title = ""
        extracted_text = ""

        # PDF?
        if resp and _looks_like_pdf(resp, url):
            text = _extract_pdf(resp)
            if text and len(text) >= min_chars:
                docs.append(Document(page_content=text, metadata={"source": url, "title": url}))
            continue

        # 정적 HTML 추출 시도
        if resp is not None:
            enc = _detect_encoding(resp.content)
            try:
                html = resp.content.decode(enc, errors="replace")
            except Exception:
                html = resp.text

            extracted_text, title = extract_readable_text(html, url)

        # 2) (옵션) 정적 추출이 짧으면 JS 렌더링 시도
        need_js = use_js_render and (not extracted_text or len(extracted_text) < min_chars)
        if need_js and (not js_only_when_needed or (js_only_when_needed and (not extracted_text or len(extracted_text) < min_chars))):
            rendered_html = render_with_playwright(url)
            if rendered_html:
                t2, tt2 = extract_readable_text(rendered_html, url)
                if len(t2) > len(extracted_text):
                    extracted_text, title = t2, (title or tt2)

        if extracted_text and len(extracted_text) >= min_chars:
            docs.append(
                Document(
                    page_content=extracted_text,
                    metadata={"source": url, "title": title or url}
                )
            )

    return docs

# text_scraper.py
# -----------------------------------------------------------------------------
# 목적
# - URL 목록을 병렬로 가져와서 "깨끗한 본문 텍스트"로 정제하여 LangChain Document 리스트로 반환
# - main.py가 import 하는 clean_html_parallel 제공
# - 부가 유틸: get_links, filter_noise
#
# 의존성 (있으면 사용, 없어도 동작):
# - requests, beautifulsoup4, chardet, readability-lxml, trafilatura
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import concurrent.futures as cf
import datetime as dt
import os
import re
import sys
from urllib.parse import urljoin, urlparse

# ---- LangChain Document 호환 --------------------------------------------------
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# ---- 선택적 의존성 ------------------------------------------------------------
try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    import chardet
except Exception:
    chardet = None  # type: ignore

try:
    from readability import Document as ReadabilityDoc  # readability-lxml
except Exception:
    ReadabilityDoc = None  # type: ignore

try:
    import trafilatura
except Exception:
    trafilatura = None  # type: ignore


# -----------------------------------------------------------------------------
# 내부 유틸
# -----------------------------------------------------------------------------
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

def _detect_encoding(b: bytes) -> str:
    if not b:
        return "utf-8"
    if chardet is not None:
        try:
            g = chardet.detect(b) or {}
            enc = g.get("encoding")
            if enc:
                return enc
        except Exception:
            pass
    return "utf-8"

def _strip_html_keep_text(html: str) -> str:
    """BeautifulSoup로 가볍게 태그 제거."""
    if not html:
        return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "iframe"]):
                tag.decompose()
            # 네비게이션/푸터 제거 시도
            for tag in soup.find_all(["header", "footer", "nav", "aside"]):
                tag.decompose()
            text = soup.get_text("\n")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()
        except Exception:
            pass
    # 최소 폴백
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", "", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _readability_extract(html: str) -> Tuple[str, Optional[str]]:
    """readability-lxml이 있으면 본문/제목 추출."""
    if not html or ReadabilityDoc is None:
        return "", None
    try:
        rd = ReadabilityDoc(html)
        title = rd.short_title()
        content_html = rd.summary(html_partial=True)
        text = _strip_html_keep_text(content_html)
        return text.strip(), (title.strip() if title else None)
    except Exception:
        return "", None

def _trafilatura_extract(html: str, base_url: Optional[str]) -> Tuple[str, Optional[str]]:
    """trafilatura가 있으면 더 정교하게 추출."""
    if not html or trafilatura is None:
        return "", None
    try:
        # favor_recall=True로 본문 회수율 우선
        text = trafilatura.extract(
            html,
            url=base_url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            with_metadata=True,
            output="txt",
        )
        if not text:
            return "", None
        # trafilatura는 메타데이터 포함 포맷을 반환할 수도 있음 -> 본문만 추출
        # 보통 ---METADATA--- 구분이 없으나, 안전하게 처리
        if isinstance(text, str) and "\n\n" in text:
            # 이미 깔끔 텍스트
            pass
        title = None
        # 메타데이터 접근이 어렵다면 soup로 타이틀 재시도
        return text.strip(), title
    except Exception:
        return "", None

def _extract_title_and_canonical(html: str, base_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    title = None
    canonical = None
    if BeautifulSoup is None or not html:
        return title, canonical
    try:
        soup = BeautifulSoup(html, "html.parser")
        # title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        ogt = soup.find("meta", property="og:title")
        if not title and ogt and ogt.get("content"):
            title = ogt["content"].strip()

        # canonical
        ln = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
        if ln and ln.get("href"):
            canonical = urljoin(base_url or "", ln["href"])
        ogu = soup.find("meta", property="og:url")
        if not canonical and ogu and ogu.get("content"):
            canonical = urljoin(base_url or "", ogu["content"])
    except Exception:
        pass
    return title, canonical

def filter_noise(text: str) -> str:
    """공통 노이즈 제거: 'Skip to content' 등 사이트 공통 문구 제거."""
    if not text:
        return ""
    # 대표적 잡문 제거
    patterns = [
        r"^\s*skip to (the )?content\s*$",
        r"^\s*skip to content\s*$",
        r"^\s*공유하기\s*$",
        r"^\s*관련\s*기사\s*$",
        r"^\s*이전\s*글\s*$",
        r"^\s*다음\s*글\s*$",
    ]
    lines = [ln for ln in text.splitlines()]
    cleaned: List[str] = []
    for ln in lines:
        l = ln.strip()
        if not l:
            cleaned.append("")
            continue
        if any(re.match(pat, l, flags=re.I) for pat in patterns):
            continue
        # 너무 짧은 네비 라인 제거
        if len(l) <= 2 and not re.search(r"[가-힣A-Za-z0-9]", l):
            continue
        cleaned.append(l)
    out = "\n".join(cleaned)
    # 공백 정리
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def get_links(html_or_url: str, base_url: Optional[str] = None) -> List[str]:
    """HTML에서 a[href] 절대URL 리스트를 뽑아준다. html_or_url이 URL이면 GET."""
    html = html_or_url
    if html_or_url.lower().startswith("http"):
        if requests is None:
            return []
        try:
            resp = requests.get(html_or_url, headers={"User-Agent": _UA}, timeout=15, allow_redirects=True)
            html = resp.text
            base_url = html_or_url
        except Exception:
            return []
    if BeautifulSoup is None:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if not href:
                continue
            absu = urljoin(base_url or "", href)
            links.append(absu)
        # 중복 제거
        seen = set()
        uniq = []
        for u in links:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq
    except Exception:
        return []


# -----------------------------------------------------------------------------
# 핵심: URL → Document (본문 정제)
# -----------------------------------------------------------------------------
def _fetch(url: str, timeout: int = 20) -> Tuple[str, int, bytes]:
    if requests is None:
        raise ImportError("requests 라이브러리가 필요합니다. `pip install requests`")
    resp = requests.get(
        url,
        headers={"User-Agent": _UA, "Accept-Language": "ko, en;q=0.8"},
        timeout=timeout,
        allow_redirects=True,
    )
    content = resp.content or b""
    return resp.url or url, getattr(resp, "status_code", 0), content

def _extract_main_text(html: str, base_url: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """가용한 라이브러리 순서대로 본문/제목/캐노니컬을 추출."""
    # 1) trafilatura 시도
    text, title_t = _trafilatura_extract(html, base_url)
    if not text:
        # 2) readability 시도
        text, title_r = _readability_extract(html)
        title_t = title_t or title_r
    if not text:
        # 3) 수동 Soup 제거
        text = _strip_html_keep_text(html)
    text = filter_noise(text)

    # 제목/캐노니컬
    title_meta, canonical = _extract_title_and_canonical(html, base_url)
    title = title_t or title_meta
    return text, title, canonical

def _to_document(url: str, timeout: int = 20) -> Document:
    fetched_url, status, raw = _fetch(url, timeout=timeout)
    enc = _detect_encoding(raw)
    try:
        html = raw.decode(enc, errors="replace")
    except Exception:
        html = raw.decode("utf-8", errors="replace")

    text, title, canonical = _extract_main_text(html, base_url=fetched_url)
    # 메타데이터 구성
    meta: Dict[str, Any] = {
        "source": fetched_url,
        "url": fetched_url,
        "canonical": canonical or "",
        "title": title or "",
        "type": "html",
        "length": len(text),
        "status_code": status,
        "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    return Document(page_content=text, metadata=meta)

def clean_html_parallel(
    urls: Iterable[str],
    *,
    render: bool = False,          # 자리표시자: JS 렌더링은 기본 미지원(서버 경량화 목적)
    timeout: int = 20,
    max_workers: int = 8,
) -> List[Document]:
    """
    URL 리스트를 병렬로 크롤링 → 본문 텍스트 정제 → Document 리스트 반환.
    - 각 Document.metadata:
        source/url/canonical/title/type(length/status_code/fetched_at)
    - 실패 시에도 빈 본문 Document를 반환하여 출처 보존
    """
    urls = list(urls or [])
    if not urls:
        return []

    docs: List[Document] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_to_document, u, timeout): u for u in urls}
        for fut in cf.as_completed(future_map):
            u = future_map[fut]
            try:
                doc = fut.result()
                docs.append(doc)
            except Exception as e:
                # 실패 시에도 출처가 남도록 빈 문서 반환
                md = {
                    "source": u,
                    "url": u,
                    "title": "",
                    "type": "html",
                    "error": str(e),
                    "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
                }
                docs.append(Document(page_content="", metadata=md))
    return docs

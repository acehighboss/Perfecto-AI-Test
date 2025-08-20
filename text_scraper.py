# text_scraper.py
# ------------------------------------------------------------------------------
# - 일부 사이트(예: KIEP, SKT 뉴스룸)의 400/403을 완화하기 위해
#   현실적인 브라우저 헤더/리퍼러/재시도 로직 추가
# - BeautifulSoup가 없으면 정규식 기반 간이 파서로 폴백
# - 기존 공개 API 시그니처 유지: get_links, clean_html_parallel, filter_noise
# ------------------------------------------------------------------------------

from __future__ import annotations

import re
import time
import random
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False


# ----------------------------- HTTP 유틸 -----------------------------

_BASE_HEADERS = {
    # 현실적인 데스크탑 UA
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "close",
}

_DEF_TIMEOUT = (8, 20)  # (connect, read)


def _build_headers_for(url: str) -> dict:
    h = dict(_BASE_HEADERS)
    # 일부 사이트는 리퍼러가 있으면 통과하는 경우가 있음
    parsed = urlparse(url)
    h["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
    return h


def _fetch_html(url: str, max_tries: int = 3) -> str:
    """
    400/403 완화를 위해 헤더·재시도·지연을 기본 적용.
    강한 차단(자바스크립트 검사, 쿠키 벽)은 코드만으로는 한계가 있음.
    """
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            headers = _build_headers_for(url)
            # 일부 사이트는 HTTP/1.1 keep-alive에서 문제 -> 연결 끊기 옵션 사용
            with requests.Session() as s:
                s.headers.update(headers)
                resp = s.get(url, timeout=_DEF_TIMEOUT, allow_redirects=True)
            if resp.status_code in (200, 201) and resp.text:
                return resp.text

            # 3xx/4xx/5xx 대응: 재시도 전 대체 헤더 시도
            if resp.status_code in (400, 401, 403, 406, 429, 500, 502, 503):
                # 간단한 헤더 변주
                time.sleep(0.8 * attempt)
                continue

            # 기타 상태코드는 예외
            resp.raise_for_status()

        except Exception as e:
            last_err = e
            time.sleep(0.6 * attempt + random.uniform(0, 0.2))

    # 모두 실패
    if last_err:
        raise last_err
    raise RuntimeError("Unknown fetch error")


# ----------------------------- 파싱 유틸 -----------------------------

def _strip_controls(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_main_text_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # NAV/ASIDE/FOOTER 제거
    for tag in soup.select("nav, header, footer, aside, script, style, noscript"):
        tag.extract()

    # 흔한 컨테이너 우선순위
    candidates = [
        "article",
        "main",
        "div#content",
        "div.content",
        "section.content",
        "div.article",
        "div.post",
    ]
    for sel in candidates:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            txt = "\n".join(p.get_text(" ", strip=True) for p in el.find_all(["p", "li", "h2", "h3", "h4"]))
            if txt:
                return _strip_controls(txt)

    # fallback: 전체에서 p/li/h2.. 추출
    txt = "\n".join(t.get_text(" ", strip=True) for t in soup.find_all(["p", "li", "h2", "h3", "h4"]))
    return _strip_controls(txt)


def _extract_main_text_regex(html: str) -> str:
    # 매우 단순한 백업 파서
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<(header|footer|nav|aside)[^>]*>.*?</\1>", " ", text)
    # 문단 비슷한 태그만 남김
    chunks = re.findall(r"(?is)<(p|li|h2|h3|h4)[^>]*>(.*?)</\1>", text)
    joined = "\n".join(_strip_controls(re.sub(r"(?is)<[^>]+>", " ", c[1])) for c in chunks)
    return _strip_controls(joined) or _strip_controls(re.sub(r"(?is)<[^>]+>", " ", text))


def _html_to_text(html: str) -> str:
    if HAS_BS4:
        return _extract_main_text_bs4(html)
    return _extract_main_text_regex(html)


# --------------------------- 공개 API (main.py가 사용) ---------------------------

def clean_html_parallel(urls: List[str]) -> List[str]:
    """여러 URL에서 본문 텍스트만 가져옴. 실패한 건 빈 문자열로 채움(스트림릿 UI 유지)."""
    out: List[str] = []
    for u in urls:
        try:
            html = _fetch_html(u, max_tries=3)
            text = _html_to_text(html)
            out.append(text)
        except Exception as e:
            print(f"[WARN] URL 요청 실패: {u} ({e})")
            out.append("")  # 실패한 경우에도 리스트 길이 유지
    return out


def filter_noise(text: str) -> str:
    """간단한 노이즈 제거(사이트 공통 상투구, 중복 공백 등)."""
    # 흔한 접근성 안내/스킵 링크 제거
    text = re.sub(r"(?im)^\s*skip to the content\s*$", "", text)
    text = re.sub(r"(?im)^\s*관련기사\s*$", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return _strip_controls(text)


def get_links(url: str) -> List[str]:
    """(옵션) 본문 내 추가 링크 추출. UI에서 쓰지 않더라도 시그니처 유지."""
    try:
        html = _fetch_html(url, max_tries=2)
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    links.append(href)
            return list(dict.fromkeys(links))
        # regex fallback
        links = re.findall(r'href="(https?://[^"]+)"', html, flags=re.I)
        return list(dict.fromkeys(links))
    except Exception:
        return []

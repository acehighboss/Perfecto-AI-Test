import asyncio
import bs4
from playwright.async_api import async_playwright
from langchain_core.documents import Document as LangChainDocument
from urllib.parse import urlparse
import httpx  # requests 대신 비동기 라이브러리 사용
from file_handler import get_documents_from_files

# --- [ 신규 추가 영역 시작 ] ---

def analyze_robots_paths(robots_content: str, path: str) -> str | None:
    if not path:
        path = "/"
    if not robots_content:
        return None

    specific_rules = {'allow': [], 'disallow': []}
    user_agent_active = False

    for line in robots_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            if key == 'user-agent':
                user_agent_active = (value == '*')
            elif user_agent_active and key in ['allow', 'disallow']:
                if value:
                    specific_rules[key].append(value)
        except ValueError:
            continue

    # 현재 경로에 매칭되는 가장 구체적인 규칙 찾기
    best_allow_match = ""
    best_disallow_match = ""

    for p in specific_rules['allow']:
        if path.startswith(p) and len(p) > len(best_allow_match):
            best_allow_match = p
    for p in specific_rules['disallow']:
        if path.startswith(p) and len(p) > len(best_disallow_match):
            best_disallow_match = p

    # 규칙 적용 (더 긴, 즉 더 구체적인 경로 규칙이 우선)
    if best_allow_match or best_disallow_match:
        if len(best_allow_match) > len(best_disallow_match):
            return f"경로별 허용 우선 ({best_allow_match})"
        elif len(best_disallow_match) > len(best_allow_match):
            return f"경로별 금지 우선 ({best_disallow_match})"
        elif best_disallow_match: # 길이가 같으면 금지 규칙을 우선
            return f"경로별 금지 우선 ({best_disallow_match})"

    return "전체 허용"


async def check_robots_txt(url: str) -> tuple[bool, str]:
    """
    (재민씨 작업2.txt의 로직을 비동기로 개선)
    URL에 대한 robots.txt를 확인하여 스크래핑 허용 여부를 상세히 판단합니다.
    """
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(robots_url, timeout=10.0)

            if 400 <= response.status_code < 500:
                return True, "robots.txt 없음 (기본 허용)"
            response.raise_for_status() # 2xx가 아니면 예외 발생

            robots_content = response.text
            path_reason = analyze_robots_paths(robots_content, parsed_url.path)

            # RobotFileParser는 전체적인 허용 여부를 판단하는 데 여전히 유용
            from urllib.robotparser import RobotFileParser
            rp = RobotFileParser()
            rp.parse(robots_content.splitlines())
            can_fetch = rp.can_fetch("*", url)

            if can_fetch:
                return True, path_reason or "robots.txt 허용"
            else:
                return False, path_reason or "robots.txt 금지"

    except (httpx.TimeoutException, httpx.RequestError):
        return True, "robots.txt 타임아웃/오류 (기본 허용)"
    except Exception as e:
        return True, f"robots.txt 확인 실패: {type(e).__name__}"

# --- [ 신규 추가 영역 종료 ] ---


async def _scrape_url_with_playwright(url: str) -> list[LangChainDocument]:
    """
    (개선) Playwright로 동적 페이지를 렌더링하고, 정교한 robots.txt 분석을 통과한 경우에만 처리합니다.
    """
    # 1. 스크래핑 전, 정교한 robots.txt 규칙 확인
    is_allowed, reason = await check_robots_txt(url)
    if not is_allowed:
        print(f"[{url}] 스크래핑 건너뜀 (사유: {reason})")
        return []
    print(f"[{url}] 스크래핑 시작 (사유: {reason})")

    # 2. Playwright를 사용해 동적 웹페이지 처리
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=15000)
            title = await page.title()
            html_content = await page.content()
            await browser.close()

        # 3. BeautifulSoup으로 기본 정제 (메타 데이터 등 제거)
        soup = bs4.BeautifulSoup(html_content, "lxml")
        for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer, link, meta"):
            element.decompose()
        
        # 4. 정제된 HTML을 바탕으로 핵심 콘텐츠 추출
        body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""

        if not body_text:
            return []

        return [LangChainDocument(page_content=body_text, metadata={"source": url, "title": title or "제목 없음"})]

    except Exception as e:
        print(f"(!) Playwright 처리 실패 [{url}]: {type(e).__name__} - {e}")
        return []


async def _get_documents_from_urls_async(urls: list[str]) -> list[LangChainDocument]:
    """
    Playwright를 사용하여 여러 URL을 병렬로 크롤링하고 문서를 생성합니다. (기존과 동일)
    """
    tasks = [_scrape_url_with_playwright(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_documents = []
    for res in results:
        if isinstance(res, list):
            all_documents.extend(res)
        elif isinstance(res, Exception):
            print(f"URL 처리 중 예외 발생: {res}")

    return all_documents


async def load_documents(source_type: str, source_input) -> list[LangChainDocument]:
    """
    소스 타입(URL 또는 파일)에 따라 문서를 로드하고 정제합니다. (기존과 동일)
    """
    documents = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            print("입력된 URL이 없습니다.")
            return []
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작 (Playwright + Advanced Robots.txt 사용)...")
        documents = await _get_documents_from_urls_async(urls)

    elif source_type == "Files":
        # 파일 처리 로직은 기존과 동일
        txt_files = [f for f in source_input if f.name.endswith('.txt')]
        other_files = [f for f in source_input if not f.name.endswith('.txt')]

        for txt_file in txt_files:
            try:
                content = txt_file.getvalue().decode('utf-8')
                doc = LangChainDocument(page_content=content, metadata={"source": txt_file.name, "title": txt_file.name})
                documents.append(doc)
            except Exception as e:
                print(f"Error reading .txt file {txt_file.name}: {e}")

        if other_files:
            print(f"{len(other_files)}개의 파일(PDF, DOCX 등)을 LlamaParse로 분석합니다...")
            llama_documents = get_documents_from_files(other_files)
            if llama_documents:
                langchain_docs = [LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents]
                documents.extend(langchain_docs)

    return documents

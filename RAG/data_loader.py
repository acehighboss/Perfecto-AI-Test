import asyncio
import bs4
from playwright.async_api import async_playwright
from langchain_core.documents import Document as LangChainDocument

from file_handler import get_documents_from_files

async def _scrape_url_with_playwright(url: str) -> list[LangChainDocument]:
    """
    Playwright를 사용하여 단일 URL의 동적 콘텐츠를 비동기적으로 스크래핑합니다.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # 네트워크가 안정될 때까지 기다려 동적 콘텐츠 로딩 보장
            await page.goto(url, wait_until="networkidle") 
            
            # 페이지 제목과 렌더링된 HTML 콘텐츠 가져오기
            title = await page.title()
            html_content = await page.content()
            
            await browser.close()

            # BeautifulSoup으로 HTML 정제
            soup = bs4.BeautifulSoup(html_content, "lxml")
            
            # 불필요한 태그 제거 (기존 로직 재사용)
            for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer"):
                element.decompose()

            # [수정] 메인 콘텐츠 영역을 더 구체적인 선택자(selector)로 탐색
            # 일반적인 뉴스/블로그 사이트에서 본문이 위치할 가능성이 높은 영역을 순서대로 탐색합니다.
            content_selectors = [
                "article", 
                "main", 
                ".post-content", 
                ".entry-content",
                "#article-view-content-div", # 사이언스타임즈의 실제 본문 ID
                ".article_body",
                "#content",
                "body" # 최후의 수단
            ]
            
            content_container = None
            for selector in content_selectors:
                container = soup.select_one(selector)
                if container:
                    content_container = container
                    # print(f"'{url}'에서 '{selector}' 선택자로 콘텐츠 영역을 찾았습니다.") # 디버깅용
                    break
            
            # 텍스트 추출 (개행을 유지하며 불필요한 공백 제거)
            if content_container:
                lines = [line.strip() for line in content_container.get_text(separator='\n').splitlines() if line.strip()]
                cleaned_text = "\n".join(lines)
            else:
                cleaned_text = ""

            if cleaned_text:
                return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title or "제목 없음"})]
            return []
    except Exception as e:
        print(f"Playwright로 URL 처리 실패 {url}: {e}")
        return []

async def _get_documents_from_urls_async(urls: list[str]) -> list[LangChainDocument]:
    """
    Playwright를 사용하여 여러 URL을 병렬로 크롤링하고 문서를 생성합니다.
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
    소스 타입(URL 또는 파일)에 따라 문서를 로드하고 정제합니다.
    """
    documents = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            print("입력된 URL이 없습니다.")
            return []
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작 (Playwright 사용)...")
        # Playwright 기반의 새 함수 호출
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

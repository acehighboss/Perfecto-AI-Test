import asyncio
import bs4
from langchain_core.documents import Document as LangChainDocument
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

from file_handler import get_documents_from_files

# ▼▼▼ [수정] 실제 브라우저처럼 보이게 할 User-Agent 문자열 정의 ▼▼▼
REALISTIC_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


async def _scrape_url_with_playwright(url: str) -> list[LangChainDocument]:
    """
    Playwright와 stealth 플러그인을 사용하여 봇 탐지를 우회하며 동적 콘텐츠를 스크래핑합니다.
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            
            # ▼▼▼ [수정] User-Agent를 설정하여 브라우저 컨텍스트 생성 ▼▼▼
            context = await browser.new_context(user_agent=REALISTIC_USER_AGENT)
            page = await context.new_page()

            # ▼▼▼ [수정] 페이지에 stealth 속성 적용 ▼▼▼
            await stealth_async(page)
            
            # 타임아웃을 30초로 설정하여 너무 오래 기다리지 않도록 함
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            title = await page.title()
            html_content = await page.content()
            
            await browser.close()

            soup = bs4.BeautifulSoup(html_content, "lxml")
            
            for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer"):
                element.decompose()

            content_container = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("body")
            cleaned_text = content_container.get_text(separator="\n", strip=True) if content_container else ""

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
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작 (Playwright Stealth 모드 사용)...")
        documents = await _get_documents_from_urls_async(urls)

    elif source_type == "Files":
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

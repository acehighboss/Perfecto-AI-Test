import asyncio
import bs4
import re
from newspaper import Article, ArticleException
from langchain_core.documents import Document as LangChainDocument

# [추가] 셀레니움 관련 라이브러리 import
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from file_handler import get_documents_from_files

# --- 셀레니움 웹드라이버 설정 (앱 실행 시 한 번만 설정) ---
try:
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("✅ 셀레니움 웹드라이버가 성공적으로 설정되었습니다.")
except Exception as e:
    print(f"❌ 셀레니움 웹드라이버 설정 중 오류 발생: {e}")
    driver = None
# ---------------------------------------------------------


def _fetch_html_with_selenium(url: str) -> str:
    """
    [신규] 셀레니움을 사용하여 자바스크립트가 렌더링된 최종 HTML을 가져옵니다.
    """
    if not driver:
        raise Exception("셀레니움 드라이버가 초기화되지 않았습니다.")
        
    driver.get(url)
    # 페이지 로딩 및 JS 실행을 위해 잠시 대기 (필요 시 시간 조정)
    asyncio.run(asyncio.sleep(3)) 
    return driver.page_source


async def _process_url(url: str) -> list:
    """
    [개선] 셀레니움으로 HTML을 가져온 뒤, BeautifulSoup으로 본문을 추출합니다.
    """
    html_content = ""
    title = "제목 없음"
    try:
        # 1. 셀레니움으로 최종 HTML 가져오기
        loop = asyncio.get_running_loop()
        html_content = await loop.run_in_executor(None, _fetch_html_with_selenium, url)

        # 2. BeautifulSoup으로 본문 추출
        soup = bs4.BeautifulSoup(html_content, "lxml")
        
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer, form, button, input, noscript"):
            element.decompose()
        
        content_container = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find(role="main") or 
            soup.find("div", class_=re.compile("content|post|body|article|entry", re.I)) or
            soup.find("div", id=re.compile("content|post|body|article|entry", re.I)) or
            soup.find("body")
        )
        
        cleaned_text = ""
        if content_container:
            cleaned_text = content_container.get_text(separator="\n", strip=True)

        if cleaned_text and len(cleaned_text) > 100:
            print(f"[성공] 셀레니움/BeautifulSoup으로 '{url}' 기사 추출 성공.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
        else:
            return f"[실패] '{url}'에서 유의미한 콘텐츠를 추출하지 못했습니다."

    except Exception as e:
        return f"URL 처리 중 오류 발생 '{url}': {e}"


async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
    # 셀레니움은 한 번에 하나의 페이지만 처리하므로, 병렬 처리 대신 순차 처리로 변경
    all_documents = []
    error_messages = []
    for url in urls:
        result = await _process_url(url)
        if isinstance(result, list):
            all_documents.extend(result)
        elif isinstance(result, str):
            error_messages.append(result)
            
    return all_documents, error_messages


async def load_documents(source_type: str, source_input) -> tuple[list, list]:
    documents = []
    errors = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            errors.append("입력된 URL이 없습니다.")
            return [], errors
        print(f"총 {len(urls)}개의 URL 순차적 분석 시작 (셀레니움 사용)...")
        documents, url_errors = await _get_documents_from_urls_async(urls)
        errors.extend(url_errors)

    elif source_type == "Files":
        txt_files = [f for f in source_input if f.name.endswith('.txt')]
        other_files = [f for f in source_input if not f.name.endswith('.txt')]

        for txt_file in txt_files:
            try:
                content = txt_file.getvalue().decode('utf-8')
                doc = LangChainDocument(page_content=content, metadata={"source": txt_file.name, "title": txt_file.name})
                documents.append(doc)
            except Exception as e:
                errors.append(f"'.txt' 파일 처리 중 오류: {txt_file.name} - {e}")
        
        if other_files:
            try:
                print(f"{len(other_files)}개의 파일(PDF, DOCX 등)을 LlamaParse로 분석합니다...")
                llama_documents = get_documents_from_files(other_files)
                if llama_documents:
                    langchain_docs = [LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents]
                    documents.extend(langchain_docs)
            except Exception as e:
                errors.append(f"LlamaParse 파일 분석 중 오류: {e}")
    
    return documents, errors

import asyncio
import bs4
import re
import requests  # [변경] requests 라이브러리 사용
import streamlit as st  # [추가] Streamlit 라이브러리 import
from langchain_core.documents import Document as LangChainDocument
from file_handler import get_documents_from_files

# --- ScrapingBee API 호출 함수 ---
def _fetch_html_with_scrapingbee(url: str) -> str:
    """
    [신규] ScrapingBee API를 사용하여 자바스크립트가 렌더링된 최종 HTML을 가져옵니다.
    """
    try:
        # Streamlit secrets에서 API 키를 안전하게 가져옵니다.
        api_key = st.secrets["SCRAPINGBEE_API_KEY"]
    except KeyError:
        # secrets에 키가 없을 경우, 명확한 에러를 발생시켜 문제를 바로 알 수 있게 합니다.
        raise KeyError("Streamlit secrets에 'SCRAPINGBEE_API_KEY'를 설정해주세요.")

    params = {
        "api_key": api_key,
        "url": url,
        "render_js": "true",  # 자바스크립트 렌더링 활성화 (핵심 기능)
        "premium_proxy": "true",  # 더 높은 성공률을 위해 프리미엄 프록시 사용
        "block_ads": "true",  # 광고 차단
        "block_resources": "false"  # CSS, 이미지 등은 로드하여 레이아웃 유지
    }
    
    # API 요청 (타임아웃을 60초로 넉넉하게 설정)
    response = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=60)
    
    # 요청 실패 시 예외 발생
    response.raise_for_status()
    
    # 성공 시, 렌더링된 HTML 텍스트를 반환
    return response.text


async def _process_url(url: str) -> list:
    """
    [개선] ScrapingBee로 HTML을 가져온 뒤, BeautifulSoup으로 본문을 추출합니다.
    """
    title = "제목 없음"
    try:
        # 1. ScrapingBee로 최종 HTML 가져오기
        loop = asyncio.get_running_loop()
        # 동기 함수인 requests를 비동기 환경에서 실행
        html_content = await loop.run_in_executor(None, _fetch_html_with_scrapingbee, url)

        # 2. BeautifulSoup으로 본문 추출 (기존의 강력한 예비 로직 재사용)
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
            print(f"✅ [성공] ScrapingBee로 '{url}' 기사 추출 성공.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
        else:
            return f"⚠️ [실패] ScrapingBee가 '{url}'에서 유의미한 콘텐츠를 추출하지 못했습니다."

    except Exception as e:
        # API 요청 실패 또는 다른 예외 발생 시 에러 메시지 반환
        return f"❌ URL 처리 중 오류 발생 '{url}': {e}"


async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
    # 한 번에 여러 URL을 비동기로 처리
    tasks = [_process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)

    all_documents = []
    error_messages = []
    for res in results:
        if isinstance(res, list):
            all_documents.extend(res)
        elif isinstance(res, str):
            error_messages.append(res)
            
    return all_documents, error_messages


async def load_documents(source_type: str, source_input) -> tuple[list, list]:
    documents = []
    errors = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            errors.append("입력된 URL이 없습니다.")
            return [], errors
        print(f"총 {len(urls)}개의 URL 분석 시작 (ScrapingBee API 사용)...")
        documents, url_errors = await _get_documents_from_urls_async(urls)
        errors.extend(url_errors)

    elif source_type == "Files": # 파일 처리 로직은 기존과 동일
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

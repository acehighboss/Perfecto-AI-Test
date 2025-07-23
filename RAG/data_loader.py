import asyncio
import bs4
import re
import requests
import streamlit as st
import json # [추가] 자바스크립트 시나리오를 위해 json 모듈 import
from langchain_core.documents import Document as LangChainDocument
from file_handler import get_documents_from_files

# --- ScrapingBee API 호출 함수 ---
def _fetch_html_with_scrapingbee(url: str) -> str:
    """
    [최종 개선] 자바스크립트 시나리오를 사용하여 동적 상호작용 후 최종 HTML을 가져옵니다.
    """
    try:
        api_key = st.secrets["SCRAPINGBEE_API_KEY"]
    except KeyError:
        raise KeyError("Streamlit secrets에 'SCRAPINGBEE_API_KEY'를 설정해주세요.")

    # [신규] 클릭, 스크롤 등 복잡한 상호작용을 위한 JS 시나리오 정의
    js_scenario = {
        "instructions": [
            {"wait": 2000},  # 페이지 로딩 후 2초 대기
            # 만약 쿠키 배너가 있다면 클릭 (CSS 선택자는 사이트마다 다를 수 있음)
            # 예시: {"click": "#cookie-accept-button"}, 
            {"wait_for": "body"}, # body 태그가 완전히 로딩될 때까지 대기
        ]
    }

    params = {
        "api_key": api_key,
        "url": url,
        "render_js": "true",
        "premium_proxy": "true", # 프리미엄 프록시로 차단 회피율 상승
        "js_scenario": json.dumps(js_scenario), # 시나리오를 JSON 문자열로 전달
    }
    
    # 타임아웃을 90초로 더욱 넉넉하게 설정
    response = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=90)
    response.raise_for_status()
    return response.text


async def _process_url(url: str) -> list:
    """
    [개선] ScrapingBee로 HTML을 가져온 뒤, BeautifulSoup으로 본문을 추출합니다.
    """
    title = "제목 없음"
    try:
        loop = asyncio.get_running_loop()
        html_content = await loop.run_in_executor(None, _fetch_html_with_scrapingbee, url)

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
            print(f"✅ [성공] ScrapingBee(JS 시나리오)로 '{url}' 기사 추출 성공.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
        else:
            return f"⚠️ [실패] ScrapingBee가 '{url}'에서 유의미한 콘텐츠를 추출하지 못했습니다."

    except Exception as e:
        return f"❌ URL 처리 중 오류 발생 '{url}': {e}"


# --- 아래 함수들은 기존과 동일합니다 ---

async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
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
        print(f"총 {len(urls)}개의 URL 분석 시작 (ScrapingBee API + JS 시나리오 사용)...")
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

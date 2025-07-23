import asyncio
import bs4
import re
import streamlit as st
from langchain_core.documents import Document as LangChainDocument
from langchain_scraperapi import ScraperAPITool

from file_handler import get_documents_from_files

def _get_document_from_url(url: str) -> list:
    """
    [최종 개선] langchain-scraperapi의 ScraperAPITool을 사용하여 URL 콘텐츠를 가져옵니다.
    """
    title = "제목 없음"
    try:
        # 1. ScraperAPITool을 초기화합니다.
        # Streamlit secrets의 SCRAPERAPI_API_KEY를 자동으로 사용합니다.
        scraper_tool = ScraperAPITool(output_format="html")

        # 2. 도구를 실행하여 HTML 콘텐츠를 가져옵니다.
        # .invoke() 메서드는 이제 딕셔너리 형태의 입력을 받습니다.
        html_content = scraper_tool.invoke({"url": url})
        
        # 3. BeautifulSoup으로 본문을 추출합니다.
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
            print(f"✅ [성공] ScraperAPITool로 '{url}' 기사 추출 성공.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
        else:
            return f"⚠️ [실패] ScraperAPITool이 '{url}'에서 유의미한 콘텐츠를 추출하지 못했습니다."

    except Exception as e:
        return f"❌ URL 처리 중 오류 발생 '{url}': {e}"


async def _process_url_async(url: str) -> list:
    """
    비동기 환경에서 동기 함수인 _get_document_from_url를 실행하기 위한 래퍼 함수입니다.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _get_document_from_url, url)
    return result


async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
    tasks = [_process_url_async(url) for url in urls]
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
        print(f"총 {len(urls)}개의 URL 분석 시작 (langchain-scraperapi 사용)...")
        documents, url_errors = await _get_documents_from_urls_async(urls)
        errors.extend(url_errors)

    elif source_type == "Files":
        txt_files = [f for f in source_input if f.name.endswith('.txt')]
        other_files = [f for f in source_input if not f.name.endswith('.txt')]

        for txt_file in txt_file:
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

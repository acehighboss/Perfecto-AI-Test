import asyncio
import bs4
import re
import requests
import streamlit as st
from langchain_core.documents import Document as LangChainDocument
from file_handler import get_documents_from_files

# --- Oxylabs API 호출 함수 ---

# ▼▼▼ [수정 제안] 안정성과 오류 진단을 개선한 함수 ▼▼▼
def _get_document_from_url(url: str) -> list | str:
    """
    [개선] requests 라이브러리와 세분화된 오류 처리를 사용하여 Oxylabs API를 호출합니다.
    """
    title = "제목 없음"
    try:
        # 1. Streamlit secrets에서 사용자 인증 정보를 안전하게 가져옵니다.
        username = st.secrets["OXYLABS_USERNAME"]
        password = st.secrets["OXYLABS_PASSWORD"]

        # 2. Oxylabs Web Scraper API의 엔드포인트와 payload를 설정합니다.
        #    - user_agent: 실제 브라우저처럼 보이도록 설정하여 안정성 향상
        #    - render: 'html'로 설정하여 JavaScript 기반 동적 콘텐츠 렌더링
        payload = {
            "source": "universal",
            "url": url,
            "render": "html",
            "user_agent_type": "desktop", # user_agent 추가
        }
        
        # 3. requests를 이용해 API를 직접 호출합니다. (HTTP Basic Auth 사용)
        response = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(username, password),
            json=payload,
            timeout=120
        )
        # HTTP 오류(4xx, 5xx) 발생 시 예외를 발생시킵니다.
        response.raise_for_status()
        
        response_json = response.json()

        # 4. [개선] Oxylabs 응답 구조를 더 안전하게 파싱합니다.
        if not response_json.get("results") or not response_json["results"][0].get("content"):
            return f"⚠️ [실패] Oxylabs가 '{url}'에 대한 콘텐츠를 반환하지 않았습니다. (응답 구조 오류)"

        result_html = response_json["results"][0]["content"]
        
        # 5. BeautifulSoup으로 본문을 추출합니다.
        soup = bs4.BeautifulSoup(result_html, "lxml")
        
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
            print(f"✅ [성공] Oxylabs로 '{url}' 기사 추출 성공.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
        else:
            return f"⚠️ [실패] Oxylabs가 '{url}'에서 유의미한 콘텐츠를 추출하지 못했습니다."

    # 6. [개선] 오류 유형에 따라 더 구체적인 예외 메시지를 제공합니다.
    except KeyError as e:
        return f"❌ '{url}' 처리 중 설정 오류 발생: Streamlit secrets에 '{e}'가 설정되었는지 확인해주세요."
    except requests.exceptions.HTTPError as e:
        return f"❌ '{url}' 처리 중 HTTP 오류 발생: {e.response.status_code} {e.response.reason}. (API 키 또는 URL 확인 필요)"
    except requests.exceptions.RequestException as e:
        return f"❌ '{url}' 처리 중 네트워크 오류 발생: {e}"
    except (IndexError, KeyError) as e:
        return f"❌ '{url}' 처리 중 API 응답 파싱 오류: {e}. (Oxylabs 응답 형식이 변경되었을 수 있습니다)"
    except Exception as e:
        return f"❌ '{url}' 처리 중 예기치 않은 오류 발생: {e}"


async def _process_url_async(url: str) -> list:
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
        print(f"총 {len(urls)}개의 URL 분석 시작 (Oxylabs API 사용)...")
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

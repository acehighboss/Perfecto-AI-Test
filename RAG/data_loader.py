import asyncio
import bs4
import aiohttp # aiohttp import
from newspaper import Article
from langchain_core.documents import Document as LangChainDocument

from file_handler import get_documents_from_files

async def _process_url(url: str, session) -> list:
    """
    [수정] 단일 URL을 비동기적으로 처리하며, 안티 크롤링 등 예외 상황을 처리합니다.
    성공 시 Document 리스트를, 실패 시 에러 메시지(str)를 반환합니다.
    """
    try:
        # 10초 이상 응답이 없으면 타임아웃 처리
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, timeout=timeout) as response:
            # 4xx, 5xx 에러 발생 시, 크롤링 실패로 간주하고 명확한 에러 메시지 반환
            if response.status >= 400:
                return f"URL 처리 실패 '{url}', 상태 코드: {response.status}. (웹사이트의 보안 정책일 수 있습니다)"
            
            html_content = await response.text()
            
            # newspaper3k로 1차 파싱 시도
            loop = asyncio.get_running_loop()
            article = await loop.run_in_executor(None, lambda: Article(url=url, language='ko'))
            await loop.run_in_executor(None, article.download, input_html=html_content)
            await loop.run_in_executor(None, article.parse)
            title = article.title
            cleaned_text = article.text

            # newspaper3k가 콘텐츠 추출에 실패했을 경우, BeautifulSoup으로 2차 시도
            if len(cleaned_text) < 100: # 콘텐츠가 너무 짧으면 실패로 간주
                soup = bs4.BeautifulSoup(html_content, "lxml")
                for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer"):
                    element.decompose()
                content_container = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("body")
                cleaned_text = content_container.get_text(separator="\n", strip=True) if content_container else ""

            if cleaned_text:
                return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title or "제목 없음"})]
            return [] # 빈 리스트 반환
            
    except asyncio.TimeoutError:
        return f"URL 처리 시간 초과: '{url}'"
    except Exception as e:
        return f"URL 처리 중 알 수 없는 오류 발생 '{url}': {e}"


async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
    """[수정] 여러 URL을 병렬로 처리하고, 성공한 문서와 실패/오류 메시지를 분리하여 반환합니다."""
    async with aiohttp.ClientSession() as session:
        tasks = [_process_url(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    all_documents = []
    error_messages = []
    for res in results:
        if isinstance(res, list): # 성공한 경우
            all_documents.extend(res)
        elif isinstance(res, str): # _process_url에서 반환한 에러 메시지인 경우
            error_messages.append(res)
        elif isinstance(res, Exception): # gather 자체에서 예외가 발생한 경우
            error_messages.append(f"비동기 작업 처리 중 예외 발생: {res}")
            
    return all_documents, error_messages


async def load_documents(source_type: str, source_input) -> tuple[list, list]:
    """[수정] 소스 타입에 따라 문서를 로드하고, 성공 리스트와 에러 리스트를 튜플로 반환합니다."""
    documents = []
    errors = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            errors.append("입력된 URL이 없습니다.")
            return [], errors
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작...")
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

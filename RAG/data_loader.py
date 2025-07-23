import asyncio
import bs4
import aiohttp
import re  # 정규 표현식 모듈 추가
from newspaper import Article, ArticleException
from langchain_core.documents import Document as LangChainDocument

from file_handler import get_documents_from_files

async def _process_url(url: str, session) -> list:
    """
    [개선] newspaper3k 실패 시, BeautifulSoup으로 더욱 강력하게 본문을 추출합니다.
    """
    html_content = ""
    title = "제목 없음"
    try:
        # 타임아웃을 15초로 늘려 안정성 확보
        timeout = aiohttp.ClientTimeout(total=15)
        headers = { # 일반적인 브라우저처럼 보이도록 User-Agent 추가
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with session.get(url, timeout=timeout, headers=headers) as response:
            if response.status >= 400:
                return f"URL 처리 실패 '{url}', 상태 코드: {response.status}. (웹사이트의 보안 정책일 수 있습니다)"
            html_content = await response.text()

        # 1. newspaper3k로 1차 시도
        try:
            article = Article(url=url, language='ko')
            article.html = html_content
            article.parse()
            cleaned_text = article.text
            title = article.title

            # 추출된 텍스트가 너무 짧으면 실패로 간주하고 예비 로직으로 강제 전환
            if len(cleaned_text or "") < 150:
                raise ArticleException()
            
            print(f"[성공] newspaper3k가 '{url}'의 기사를 성공적으로 추출했습니다.")
            return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title or "제목 없음"})]

        # 2. newspaper3k 실패 시, BeautifulSoup 예비 로직 발동
        except ArticleException:
            print(f"[정보] newspaper3k가 '{url}' 추출에 실패하여 BeautifulSoup으로 재시도합니다.")
            soup = bs4.BeautifulSoup(html_content, "lxml")
            
            # 제목을 더 확실하게 추출
            if soup.title and soup.title.string:
                title = soup.title.string
            elif soup.find("h1"):
                title = soup.find("h1").get_text()

            # 불필요한 태그 제거 (더 공격적으로 제거)
            for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer, form, button, input, noscript"):
                element.decompose()
            
            # 다양한 형태의 본문 컨테이너 탐색
            content_container = (
                soup.find("main") or 
                soup.find("article") or 
                soup.find(role="main") or 
                soup.find("div", class_=re.compile("content|post|body|article|entry", re.I)) or
                soup.find("div", id=re.compile("content|post|body|article|entry", re.I))
            )
            
            # 최후의 수단으로 body 전체 사용
            if not content_container:
                content_container = soup.find("body")
            
            cleaned_text = ""
            if content_container:
                cleaned_text = content_container.get_text(separator="\n", strip=True)

            if cleaned_text:
                print(f"[성공] BeautifulSoup이 '{url}'의 기사를 성공적으로 추출했습니다.")
                return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title})]
            else:
                return f"[실패] BeautifulSoup으로도 '{url}'의 기사를 추출하지 못했습니다."

    except asyncio.TimeoutError:
        return f"URL 처리 시간 초과: '{url}'"
    except Exception as e:
        return f"URL 처리 중 알 수 없는 오류 발생 '{url}': {e}"


async def _get_documents_from_urls_async(urls: list[str]) -> tuple[list, list]:
    async with aiohttp.ClientSession() as session:
        tasks = [_process_url(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    all_documents = []
    error_messages = []
    for res in results:
        if isinstance(res, list):
            all_documents.extend(res)
        elif isinstance(res, str):
            error_messages.append(res)
        elif isinstance(res, Exception):
            error_messages.append(f"비동기 작업 처리 중 예외 발생: {res}")
            
    return all_documents, error_messages


async def load_documents(source_type: str, source_input) -> tuple[list, list]:
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

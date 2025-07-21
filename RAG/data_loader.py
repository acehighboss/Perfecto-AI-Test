import asyncio
import bs4
from newspaper import Article
from langchain_core.documents import Document as LangChainDocument

from file_handler import get_documents_from_files

async def _process_url(url: str, session) -> list[LangChainDocument]:
    """단일 URL을 비동기적으로 처리하여 정제된 Document 리스트를 반환합니다."""
    try:
        loop = asyncio.get_running_loop()
        article = await loop.run_in_executor(None, lambda: Article(url=url, language='ko'))
        await loop.run_in_executor(None, article.download)
        await loop.run_in_executor(None, article.parse)
        title = article.title
        
        async with session.get(url) as response:
            html_content = await response.text()
            soup = bs4.BeautifulSoup(html_content, "lxml")
            
            for element in soup.select("script, style, nav, footer, aside, .ad, .advertisement, .banner, .menu, .header, .footer"):
                element.decompose()

            content_container = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("body")
            cleaned_text = content_container.get_text(separator="\n", strip=True) if content_container else article.text

            if cleaned_text:
                return [LangChainDocument(page_content=cleaned_text, metadata={"source": url, "title": title or "제목 없음"})]
            return []
    except Exception as e:
        print(f"URL 처리 실패 {url}: {e}")
        return []

async def _get_documents_from_urls_async(urls: list[str]) -> list[LangChainDocument]:
    """여러 URL을 병렬로 크롤링하고 문서를 생성합니다."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [_process_url(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    all_documents = []
    for res in results:
        if isinstance(res, list):
            all_documents.extend(res)
        elif isinstance(res, Exception):
            print(f"URL 처리 중 예외 발생: {res}")
            
    return all_documents

async def load_documents(source_type: str, source_input) -> list[LangChainDocument]:
    """소스 타입(URL 또는 파일)에 따라 문서를 로드하고 정제합니다."""
    documents = []
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            print("입력된 URL이 없습니다.")
            return []
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작...")
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
            llama_documents = await get_documents_from_files(other_files)
            if llama_documents:
                langchain_docs = [LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents]
                documents.extend(langchain_docs)
    
    return documents

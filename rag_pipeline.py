import nest_asyncio
nest_asyncio.apply()

import asyncio
import time
import bs4
import nltk
from newspaper import Article
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker

from file_handler import get_documents_from_files

# --- NLTK 데이터 자동 다운로드 로직 ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' 모델이 이미 존재합니다.")
except LookupError:
    print("NLTK 'punkt' 모델을 찾을 수 없어 다운로드를 시작합니다...")
    nltk.download('punkt')
    print("다운로드가 완료되었습니다.")


# --- 파라미터 튜닝을 위한 설정 클래스 ---
class RAGConfig:
    # 3순위 (고정 권장)
    CHUNK_SIZE = 400
    BM25_K1 = 1.2
    BM25_B = 0.75

    # 2순위 (중간 영향)
    BM25_TOP_K = 50
    RERANK_1_TOP_N = 20
    FAISS_TOP_K = 15
    RERANK_2_TOP_N = 5

    # 1순위 (성능에 가장 큰 영향)
    RERANK_1_THRESHOLD = 0.5
    RERANK_2_THRESHOLD = 0.7
    FINAL_DOCS_COUNT = 5
    
    # 임베딩 배치 설정
    EMBEDDING_BATCH_SIZE = 250

# --- 핵심 RAG 파이프라인 ---

async def process_url(url: str, session) -> list[LangChainDocument]:
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

async def get_documents_from_urls_async(urls: list[str]) -> list[LangChainDocument]:
    """여러 URL을 병렬로 크롤링하고 문서를 생성합니다."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
    all_documents = []
    for res in results:
        if isinstance(res, list):
            all_documents.extend(res)
        elif isinstance(res, Exception):
            print(f"URL 처리 중 예외 발생: {res}")
            
    return all_documents

async def sentence_split_and_embed_async(query: str, compression_retriever_1, embeddings):
    """(비동기) 1, 2단계 필터링 및 문장 분할, 임베딩, 최종 Rerank를 수행합니다."""
    print(f"\n사용자 질문으로 1/2단계 필터링 실행: {query}")
    reranked_chunks = compression_retriever_1.invoke(query)
    print(f"1차 Rerank 후 {len(reranked_chunks)}개 청크 선별 완료.")

    sentences = []
    for chunk in reranked_chunks:
        sents = nltk.sent_tokenize(chunk.page_content)
        for i, sent in enumerate(sents):
            metadata = chunk.metadata.copy()
            metadata["chunk_location"] = f"chunk_{i+1}"
            sentences.append(LangChainDocument(page_content=sent, metadata=metadata))
    print(f"총 {len(sentences)}개의 문장으로 분할 완료.")
    if not sentences: return []

    print("문장 임베딩 및 FAISS 인덱싱 시작...")
    sentence_texts = [s.page_content for s in sentences]
    
    async def embed_in_batches():
        all_embeddings = []
        for i in range(0, len(sentence_texts), RAGConfig.EMBEDDING_BATCH_SIZE):
            batch = sentence_texts[i:i + RAGConfig.EMBEDDING_BATCH_SIZE]
            print(f"임베딩 배치 {i // RAGConfig.EMBEDDING_BATCH_SIZE + 1} 처리 중 ({len(batch)}개 문장)...")
            batch_embeddings = await embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    embedded_vectors = await embed_in_batches()
    
    text_embedding_pairs = list(zip(sentence_texts, embedded_vectors))
    faiss_index = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings, metadatas=[s.metadata for s in sentences])
    
    print(f"FAISS 검색 (상위 {RAGConfig.FAISS_TOP_K}개 문장 선별)...")
    faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": RAGConfig.FAISS_TOP_K})
    faiss_results = faiss_retriever.invoke(query)

    print(f"2차 Cohere Rerank (최종 {RAGConfig.RERANK_2_TOP_N}개 문장 선별, Threshold: {RAGConfig.RERANK_2_THRESHOLD})...")
    cohere_compressor_2 = CohereRerank(model="rerank-multilingual-v3.0", top_n=RAGConfig.RERANK_2_TOP_N)
    final_reranker = cohere_compressor_2.compress_documents(documents=faiss_results, query=query)
    
    final_docs = [
        doc for doc in final_reranker 
        if doc.metadata.get('relevance_score', 0) >= RAGConfig.RERANK_2_THRESHOLD
    ][:RAGConfig.FINAL_DOCS_COUNT]

    print(f"최종 {len(final_docs)}개 문장 선별 완료.")
    return final_docs

async def get_retriever_from_source_async(source_type, source_input):
    """
    URL 리스트 또는 파일로부터 문서를 로드하고, 다단계 필터링 Retriever를 생성합니다.
    """
    start_time = time.time()
    documents = []

    print("\n[1단계: 관련 콘텐츠 추출 시작]")
    if source_type == "URL":
        urls = [url.strip() for url in source_input.splitlines() if url.strip()]
        if not urls:
            print("입력된 URL이 없습니다.")
            return None
        print(f"총 {len(urls)}개의 URL 병렬 크롤링 시작...")
        documents = await get_documents_from_urls_async(urls)

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

    if not documents:
        print("처리할 문서를 찾지 못했습니다.")
        return None
    
    print(f"콘텐츠 추출 완료. (소요 시간: {time.time() - start_time:.2f}초)")
    
    print("\n의미적 경계 기반 청크화 시작...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        min_chunk_size=100,
        buffer_size=1,
        breakpoint_threshold_amount=95
    )
    chunks = text_splitter.split_documents(documents)
    print(f"총 {len(chunks)}개의 청크 생성 완료.")
    if not chunks: return None

    print("\n[2단계: 청크 단위 1차 필터링 시작]")
    print(f"BM25 검색 (상위 {RAGConfig.BM25_TOP_K}개 선별)...")
    bm25_retriever = BM25Retriever.from_documents(chunks, k=RAGConfig.BM25_TOP_K, bm25_params={'k1': RAGConfig.BM25_K1, 'b': RAGConfig.BM25_B})
    
    print(f"1차 Cohere Rerank (상위 {RAGConfig.RERANK_1_TOP_N}개 압축, Threshold: {RAGConfig.RERANK_1_THRESHOLD})...")
    cohere_compressor_1 = CohereRerank(model="rerank-multilingual-v3.0", top_n=RAGConfig.RERANK_1_TOP_N)
    compression_retriever_1 = ContextualCompressionRetriever(
        base_compressor=cohere_compressor_1, base_retriever=bm25_retriever
    )
    
    print("\n[3단계: Retriever 구성 완료]")
    
    def sync_retriever_wrapper(query: str):
        return asyncio.run(sentence_split_and_embed_async(query, compression_retriever_1, embeddings))
    
    return RunnableLambda(sync_retriever_wrapper)


def get_retriever_from_source(source_type, source_input):
    """
    비동기 함수인 get_retriever_from_source_async를 실행하고 결과를 반환합니다.
    """
    try:
        return asyncio.run(get_retriever_from_source_async(source_type, source_input))
    except Exception as e:
        print(f"Retriever 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_conversational_rag_chain(retriever, system_prompt):
    """
    최종적으로 생성된 문장 단위의 출처를 사용하여 답변을 생성하는 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    rag_prompt_template = f"""{system_prompt}

Answer the user's request based *only* on the provided "Context".
If the context does not contain the answer, say you don't know.
Do not use any prior knowledge.

**Context:**
{{context}}

**User's Request:**
{{input}}

**Answer (in Korean):**
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    def format_docs_with_metadata(docs: list[LangChainDocument]) -> str:
        """문서 리스트를 LLM 프롬프트 형식에 맞게 변환합니다."""
        if not docs:
            return "No context provided."
        
        sources = {}
        for doc in docs:
            source_url = doc.metadata.get("source", "Unknown Source")
            title = doc.metadata.get("title", "No Title")
            key = (source_url, title)
            if key not in sources:
                sources[key] = []
            sources[key].append(doc.page_content)

        formatted_string = ""
        for (source_url, title), sentences in sources.items():
            formatted_string += f"\n--- Source: {title} ({source_url}) ---\n"
            formatted_string += "\n".join(f"- {s}" for s in sentences)
        return formatted_string.strip()

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_metadata), "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def get_default_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()

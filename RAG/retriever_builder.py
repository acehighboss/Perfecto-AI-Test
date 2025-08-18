# RAG/retriever_builder.py

from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever
# ★★★ 새로운 텍스트 분할기 임포트 ★★★
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .rag_config import RAGConfig
from .redis_cache import get_from_cache, set_to_cache, create_cache_key

def build_retriever(documents: list[LangChainDocument]):
    """
    문서를 의미 있는 청크로 분할하고, 하이브리드 검색 및 Rerank를 수행하는
    RAG 파이프라인을 구성합니다.
    """
    if not documents:
        return None

    # ★★★ 1. RecursiveCharacterTextSplitter를 사용한 문서 분할 ★★★
    # 기존의 문장 단위 분할 대신, 맥락을 유지하는 청크 단위로 분할합니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        print("분할된 청크가 없어 Retriever를 생성할 수 없습니다.")
        return None
    print(f"총 {len(chunks)}개의 청크 생성 완료.")

    # 2. 임베딩 및 벡터 저장소(FAISS) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RAGConfig.FAISS_TOP_K})
    except Exception as e:
        print(f"FAISS 인덱스 생성 실패: {e}")
        return None

    # 3. 키워드 기반 검색(BM25) Retriever 생성
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RAGConfig.BM25_TOP_K

    # 4. 하이브리드 검색을 위한 EnsembleRetriever 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4]  # 키워드 검색에 약간 더 가중치 부여
    )

    # 5. Cohere Rerank 압축기 설정
    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=RAGConfig.RERANK_TOP_N)

    # 6. 최종 파이프라인 체인 구성
    def get_cached_or_run_pipeline(query: str):
        cache_key = create_cache_key("final_rag_result_openai", query)
        
        cached_docs = get_from_cache(cache_key)
        if cached_docs is not None:
            return cached_docs

        # 하이브리드 검색 실행
        retrieved_docs = ensemble_retriever.invoke(query)
        
        # Reranker를 통해 관련성 높은 순으로 정렬 및 압축
        reranked_docs = cohere_reranker.compress_documents(documents=retrieved_docs, query=query)
        
        # 관련성 점수가 임계값 이상인 문서만 필터링
        final_docs = [
            doc for doc in reranked_docs 
            if doc.metadata.get('relevance_score', 0) >= RAGConfig.RERANK_THRESHOLD
        ][:RAGConfig.FINAL_DOCS_COUNT]

        # 만약 필터링 후 문서가 하나도 없다면, Rerank 결과의 최상위 문서를 사용 (Fallback)
        if not final_docs and reranked_docs:
            final_docs = reranked_docs[:RAGConfig.FINAL_DOCS_COUNT_FALLBACK]

        print(f"최종 {len(final_docs)}개 청크 선별 완료.")
        
        set_to_cache(cache_key, final_docs)
        
        return final_docs

    return RunnableLambda(get_cached_or_run_pipeline)

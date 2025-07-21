import asyncio
import spacy
import re
from langdetect import detect, LangDetectException
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.cache import RedisCache
from redis import Redis

from .rag_config import RAGConfig

try:
    nlp_ko = spacy.load("ko_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
    print("spaCy 한국어 및 영어 모델 로딩 완료!")
except OSError:
    print("spaCy 모델을 찾을 수 없습니다. requirements.txt를 확인하고 다시 설치해주세요.")
    nlp_ko, nlp_en = None, None

async def _sentence_split_and_embed_async(query: str, compression_retriever_1, embeddings):
    """(비동기) 1, 2단계 필터링 및 문장 분할, 임베딩, 최종 Rerank를 수행합니다."""
    print(f"\n사용자 질문으로 1/2단계 필터링 실행: {query}")
    reranked_chunks = compression_retriever_1.invoke(query)
    print(f"1차 Rerank 후 {len(reranked_chunks)}개 청크 선별 완료.")

    sentences = []
    for chunk in reranked_chunks:
        try:
            lang = detect(chunk.page_content)
        except LangDetectException:
            lang = 'en'
            
        if lang == 'ko' and nlp_ko:
            doc = nlp_ko(chunk.page_content)
        elif nlp_en:
            doc = nlp_en(chunk.page_content)
        else:
            sents = re.split(r'(?<=[.?!])\s+', chunk.page_content)
            # doc.sents를 사용하지 않으므로 직접 sents를 순회합니다.
            for i, sent in enumerate(sents):
                if sent.strip():
                    metadata = chunk.metadata.copy()
                    metadata["chunk_location"] = f"chunk_{i+1}"
                    sentences.append(LangChainDocument(page_content=sent.strip(), metadata=metadata))
            continue # spaCy 처리를 건너뛰고 다음 청크로 넘어갑니다.

        # spaCy 처리 결과
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
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

def build_retriever(documents: list[LangChainDocument]):
    """문서 리스트를 받아 다단계 Retriever를 구성하고 반환합니다."""
    print("\n의미적 경계 기반 청크화 시작...")

    try:
        import streamlit as st
        redis_url = st.secrets.get("REDIS_URL", "redis://localhost:6379")
        print("Redis 캐시를 연결합니다...")
        redis_client = Redis.from_url(redis_url)
        cache = RedisCache(redis_client)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            cache=cache
        )
        print("Redis 캐시 연결 성공!")
    except Exception as e:
        print(f"Redis 캐시 연결 실패: {e}. 캐시 없이 진행합니다.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        buffer_size=1
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
        return asyncio.run(_sentence_split_and_embed_async(query, compression_retriever_1, embeddings))
    
    return RunnableLambda(sync_retriever_wrapper)

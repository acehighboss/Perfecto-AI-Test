import asyncio
import spacy
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_experimental.text_splitter import SemanticChunker

from .rag_config import RAGConfig
from .redis_cache import get_from_cache, set_to_cache, create_cache_key

# spaCy 언어 모델 로드 (앱 실행 시 한 번만 로드)
try:
    nlp_korean = spacy.load("ko_core_news_sm")
    nlp_english = spacy.load("en_core_web_sm")
    print("✅ spaCy language models loaded successfully.")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 'requirements.txt'에 모델이 포함되었는지 확인하세요.")
    # 모델 로드 실패 시 기본값으로 None 설정
    nlp_korean, nlp_english = None, None


async def _sentence_split_and_embed_async(query: str, compression_retriever_1, embeddings):
    """(비동기) 1, 2단계 필터링 및 문장 분할, 임베딩, 최종 Rerank를 수행 (Redis, spaCy 적용)."""
    
    # 1. 쿼리를 기반으로 고유한 캐시 키 생성
    cache_key = create_cache_key("rag_result", query)
    
    # 2. Redis에서 캐시된 결과 확인
    cached_docs = get_from_cache(cache_key)
    if cached_docs is not None:
        # 캐시가 존재하면 즉시 반환
        return cached_docs

    print(f"\n[Cache Miss] 사용자 질문으로 1/2단계 필터링 실행: {query}")
    reranked_chunks = compression_retriever_1.invoke(query)
    print(f"1차 Rerank 후 {len(reranked_chunks)}개 청크 선별 완료.")

    sentences = []
    
    # spaCy를 이용한 지능형 문장 분할
    if nlp_korean and nlp_english: # 모델이 정상적으로 로드되었을 때만 실행
        for chunk in reranked_chunks:
            # 먼저 한국어 모델로 시도
            doc = nlp_korean(chunk.page_content)
            # 휴리스틱: 한국어 문장이 거의 없으면(알파벳 비율이 높으면) 영어 모델 재시도
            if len(doc.sents) <= 1 and sum(c.isalpha() and 'a' <= c.lower() <= 'z' for c in chunk.page_content) / len(chunk.page_content) > 0.5:
                doc = nlp_english(chunk.page_content)

            sents = [sent.text.strip() for sent in doc.sents]

            for i, sent_text in enumerate(sents):
                if sent_text:
                    metadata = chunk.metadata.copy()
                    metadata["chunk_location"] = f"chunk_{i+1}"
                    sentences.append(LangChainDocument(page_content=sent_text, metadata=metadata))
    else:
        print("spaCy 모델이 없어 문장 분할을 건너뜁니다. 청크를 그대로 사용합니다.")
        sentences = reranked_chunks # spaCy 실패 시 청크 단위를 그대로 사용


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

    # 최종 결과를 Redis에 저장
    set_to_cache(cache_key, final_docs)

    print(f"최종 {len(final_docs)}개 문장 선별 완료.")
    return final_docs


def build_retriever(documents: list[LangChainDocument]):
    """문서 리스트를 받아 다단계 Retriever를 구성하고 반환합니다."""
    print("\n의미적 경계 기반 청크화 시작...")
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

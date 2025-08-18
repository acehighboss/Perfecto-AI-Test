import spacy
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever

from .rag_config import RAGConfig
from .redis_cache import get_from_cache, set_to_cache, create_cache_key

# spaCy 언어 모델 로드
try:
    nlp_korean = spacy.load("ko_core_news_sm")
    nlp_english = spacy.load("en_core_web_sm")
    print("✅ spaCy language models loaded successfully.")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 'requirements.txt'를 확인하세요.")
    nlp_korean, nlp_english = None, None

def _split_documents_into_sentences(documents: list[LangChainDocument]) -> list[LangChainDocument]:
    """문서 리스트를 spaCy를 이용해 문장 단위로 분할합니다."""
    sentences = []
    if not nlp_korean or not nlp_english:
        return documents

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            continue
        
        # 언어 감지 휴리스틱 (한국어/영어)
        try:
            is_korean = sum('가' <= c <= '힣' for c in doc.page_content[:200]) > 10
            nlp = nlp_korean if is_korean else nlp_english
            nlp_doc = nlp(doc.page_content)
        except Exception:
            continue

        for sent in nlp_doc.sents:
            if sent.text.strip():
                sentences.append(LangChainDocument(page_content=sent.text.strip(), metadata=doc.metadata.copy()))
    
    return sentences


def build_retriever(documents: list[LangChainDocument]):
    """
    문서를 문장 단위로 분해하고, 하이브리드 검색(BM25 + FAISS) 및 Rerank를 수행하는
    전체 RAG 파이프라인을 구성합니다.
    """
    if not documents:
        return None

    # 1. 문서 전체를 문장으로 분할
    sentences = _split_documents_into_sentences(documents)
    if not sentences:
        return None
    print(f"총 {len(sentences)}개의 문장 생성 완료.")

    # 2. 임베딩 및 벡터 저장소(FAISS) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 
    try:
        vectorstore = FAISS.from_documents(sentences, embeddings)
        # FAISS가 가져오는 문서 수를 늘림
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RAGConfig.FAISS_TOP_K})
    except Exception as e:
        print(f"FAISS 인덱스 생성 실패: {e}")
        return None
    
    # 3. 키워드 기반 검색(BM25) Retriever 생성
    bm25_retriever = BM25Retriever.from_documents(sentences)
    bm25_retriever.k = RAGConfig.BM25_TOP_K

    # 4. 하이브리드 검색을 위한 EnsembleRetriever 생성
    # ★★ BM25(키워드) 가중치를 높여 질문의 핵심 단어가 포함된 문장을 더 잘 찾도록 수정 ★★
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3] # BM25 가중치 상향
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
        
        # ★★ 필터링 로직 수정 ★★
        # 관련성 점수가 임계값 이상인 문서만 선택하되,
        # 만약 그런 문서가 하나도 없다면 가장 점수가 높은 상위 문서를 최소한으로 포함
        high_relevance_docs = [
            doc for doc in reranked_docs 
            if doc.metadata.get('relevance_score', 0) >= RAGConfig.RERANK_THRESHOLD
        ]
        
        if not high_relevance_docs and reranked_docs:
            # 임계값을 넘는 문서가 없으면, 가장 점수 높은 문서를 최소한으로 포함
            final_docs = reranked_docs[:RAGConfig.FINAL_DOCS_COUNT_FALLBACK]
        else:
            # 임계값을 넘는 문서가 있으면, 그 중에서 최종 개수만큼 선택
            final_docs = high_relevance_docs[:RAGConfig.FINAL_DOCS_COUNT]

        print(f"최종 {len(final_docs)}개 문장 선별 완료.")
        
        set_to_cache(cache_key, final_docs)
        
        return final_docs

    return RunnableLambda(get_cached_or_run_pipeline)

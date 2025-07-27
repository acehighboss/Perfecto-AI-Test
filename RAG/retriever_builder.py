import asyncio
import spacy
from langchain_core.documents import Document as LangChainDocument
from langchain_core.runnables import RunnableLambda
# ▼▼▼ [수정] OpenAIEmbeddings 임포트 ▼▼▼
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever

from .rag_config import RAGConfig
from .redis_cache import get_from_cache, set_to_cache, create_cache_key

# spaCy 언어 모델 로드 (앱 실행 시 한 번만 로드)
try:
    nlp_korean = spacy.load("ko_core_news_sm")
    nlp_english = spacy.load("en_core_web_sm")
    print("✅ spaCy language models loaded successfully.")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 'requirements.txt'에 모델이 포함되었는지 확인하세요.")
    nlp_korean, nlp_english = None, None

def _split_documents_into_sentences(documents: list[LangChainDocument]) -> list[LangChainDocument]:
    """문서 리스트를 spaCy를 이용해 문장 단위로 분할합니다."""
    sentences = []
    if not nlp_korean or not nlp_english:
        print("spaCy 모델이 로드되지 않아 문장 분할을 건너뜁니다.")
        return documents

    for doc in documents:
        if not doc.page_content or not doc.page_content.strip():
            continue

        # 먼저 한국어 모델로 시도
        nlp_doc = nlp_korean(doc.page_content)
        # 휴리스틱: 한국어 문장이 거의 없으면(알파벳 비율이 높으면) 영어 모델 재시도
        try:
            sents_list = list(nlp_doc.sents)
            if len(sents_list) <= 1 and sum(c.isalpha() and 'a' <= c.lower() <= 'z' for c in doc.page_content) / len(doc.page_content) > 0.5:
                nlp_doc = nlp_english(doc.page_content)
        except ZeroDivisionError: # 빈 page_content에 대한 예외 처리
            continue

        for sent in nlp_doc.sents:
            if sent.text.strip():
                sentences.append(LangChainDocument(page_content=sent.text.strip(), metadata=doc.metadata.copy()))

    # ▼▼▼ [디버깅 코드] ▼▼▼
    print("\n\n" + "="*50, flush=True)
    print("🕵️ 2. [retriever_builder] 문장 분할 결과 확인", flush=True)
    print(f"총 {len(sentences)}개의 문장으로 분할되었습니다.", flush=True)
    eps_sentences = [s.page_content for s in sentences if "EPS" in s.page_content]
    if eps_sentences:
        print("✅ EPS 관련 문장이 성공적으로 분할되었습니다:", flush=True)
        for sent in eps_sentences:
            print(f"   - {sent}", flush=True)
    else:
        print("🚨 경고: 문장 분할 후 EPS 관련 정보를 찾을 수 없습니다.", flush=True)
    print("="*50 + "\n\n", flush=True)
    # ▲▲▲ [디버깅 코드] ▲▲▲
    
    return sentences


def build_retriever(documents: list[LangChainDocument]):
    """
    문서를 문장 단위로 분해하고, 하이브리드 검색(BM25 + FAISS) 및 Rerank를 수행하는
    전체 RAG 파이프라인을 구성합니다.
    """
    if not documents:
        return None

    # 1. 문서 전체를 문장으로 분할
    print("\n[1단계: 문서 전체를 문장 단위로 분할]")
    sentences = _split_documents_into_sentences(documents)
    if not sentences:
        print("분할된 문장이 없어 Retriever를 생성할 수 없습니다.")
        return None
    print(f"총 {len(sentences)}개의 문장 생성 완료.")

    # ▼▼▼ [수정] Google 임베딩을 OpenAI 임베딩으로 교체 ▼▼▼
    # 2. 임베딩 및 벡터 저장소(FAISS) 생성
    print("\n[2단계: 문장 임베딩 및 벡터 저장소 생성 (OpenAI)]")
    # 모델 이름은 필요에 따라 변경 가능 (예: "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 
    try:
        vectorstore = FAISS.from_documents(sentences, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RAGConfig.BM25_TOP_K})
    except Exception as e:
        print(f"FAISS 인덱스 생성 실패: {e}")
        return None
    
    # 3. 키워드 기반 검색(BM25) Retriever 생성
    print("\n[3단계: 키워드 기반 BM25 Retriever 생성]")
    bm25_retriever = BM25Retriever.from_documents(sentences)
    bm25_retriever.k = RAGConfig.BM25_TOP_K

    # 4. 하이브리드 검색을 위한 EnsembleRetriever 생성
    print("\n[4단계: 하이브리드 Ensemble Retriever 구성]")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    # 5. Cohere Rerank 압축기 설정
    print("\n[5단계: Cohere Reranker 구성]")
    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=RAGConfig.RERANK_1_TOP_N)

    # 6. 최종 파이프라인 체인 구성
    def get_cached_or_run_pipeline(query: str):
        cache_key = create_cache_key("final_rag_result_openai", query) # 캐시 키 변경
        
        cached_docs = get_from_cache(cache_key)
        if cached_docs is not None:
            return cached_docs

        print(f"\n[Cache Miss] 질문 '{query}'에 대한 RAG 파이프라인 실행 (OpenAI)")
        
        retrieved_docs = ensemble_retriever.invoke(query)
        print(f"하이브리드 검색 후 {len(retrieved_docs)}개 문장 선별 완료.")

        reranked_docs = cohere_reranker.compress_documents(documents=retrieved_docs, query=query)
        print(f"Cohere Rerank 후 {len(reranked_docs)}개 문장 선별 완료.")
        
        final_docs = [
            doc for doc in reranked_docs 
            if doc.metadata.get('relevance_score', 0) >= RAGConfig.RERANK_2_THRESHOLD
        ][:RAGConfig.FINAL_DOCS_COUNT]

        print(f"최종 {len(final_docs)}개 문장 선별 완료.")

        # ▼▼▼ [디버깅 코드] ▼▼▼
        print(f"\n\n{'='*50}")
        print(f"🕵️ 3. [retriever_builder] RAG 파이프라인 단계별 결과 확인")
        print(f"질문: {query}")
        print(f"{'-'*50}")

        retrieved_docs = ensemble_retriever.invoke(query)
        print(f"➡️ [단계 1] 하이브리드 검색(BM25+FAISS) 결과: {len(retrieved_docs)}개 문서")
        eps_in_retrieved = [doc for doc in retrieved_docs if "EPS" in doc.page_content]
        if eps_in_retrieved:
            print(f"✅ 이 중 {len(eps_in_retrieved)}개 문서에 EPS 정보가 포함되어 있습니다.")
        else:
            print("🚨 경고: 하이브리드 검색 결과에서 EPS 관련 문서를 찾지 못했습니다.")
        print(f"{'-'*50}")

        reranked_docs = cohere_reranker.compress_documents(documents=retrieved_docs, query=query)
        print(f"➡️ [단계 2] Cohere Rerank 결과: {len(reranked_docs)}개 문서")
        eps_in_reranked = [doc for doc in reranked_docs if "EPS" in doc.page_content]
        if eps_in_reranked:
            print(f"✅ Rerank 후에도 {len(eps_in_reranked)}개 문서에 EPS 정보가 남아있습니다.")
            for doc in eps_in_reranked:
                 print(f"   - (점수: {doc.metadata.get('relevance_score', 'N/A')}) {doc.page_content}")
        else:
            print("🚨 경고: Rerank 과정에서 EPS 관련 문서가 필터링되었습니다.")
        print(f"{'-'*50}")
        
        final_docs = [
            doc for doc in reranked_docs 
            if doc.metadata.get('relevance_score', 0) >= RAGConfig.RERANK_2_THRESHOLD
        ][:RAGConfig.FINAL_DOCS_COUNT]

        print(f"➡️ [단계 3] 최종 필터링(점수 >= {RAGConfig.RERANK_2_THRESHOLD}) 결과: {len(final_docs)}개 문서")
        eps_in_final = [doc for doc in final_docs if "EPS" in doc.page_content]
        if eps_in_final:
            print(f"✅ 최종 문서에 EPS 정보가 포함되어 있습니다!")
        else:
            print("🚨 문제 지점: 최종 필터링 과정에서 EPS 관련 문서가 제거되었습니다. RERANK_2_THRESHOLD 값을 낮춰보는 것을 고려해보세요.")
        print(f"{'='*50}\n\n")
        # ▲▲▲ [디버깅 코드] ▲▲▲
        
        set_to_cache(cache_key, final_docs)
        
        return final_docs

    return RunnableLambda(get_cached_or_run_pipeline)

from __future__ import annotations
from typing import List, Optional, Sequence
import os

from langchain_core.documents import Document

# 벡터스토어/임베딩
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# SmartRetriever 래퍼
from RAG.smart_retriever import SmartRetriever, SmartRetrieverConfig


# 임베딩 & 벡터스토어 유틸
def create_default_embeddings() -> OpenAIEmbeddings:
    """
    다국어 성능이 필요한 경우 OpenAI 임베딩을 기본값으로 사용.
    필요 시 환경변수/설정으로 모델명을 바꾸세요.
    """
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    return OpenAIEmbeddings(model=model)


def build_faiss_vectorstore(
    docs: Sequence[Document],
    embeddings: Optional[OpenAIEmbeddings] = None,
) -> FAISS:
    """
    업로드/크롤링된 Document들을 받아 FAISS 벡터스토어를 구성.
    문서의 metadata에는 최소한 'source' (URL/파일경로), 'page'(있으면) 가 들어있길 권장.
    """
    if embeddings is None:
        embeddings = create_default_embeddings()

    # 빈 입력 방어
    safe_docs = [d for d in docs if (d and (d.page_content or "").strip())]
    if not safe_docs:
        # 빈 인덱스를 돌려주면 후단에서 폴백 로직이 동작하게 됨
        return FAISS.from_texts(texts=[""], embedding=embeddings)

    return FAISS.from_documents(safe_docs, embeddings)


# similarity 유지 + SmartRetriever 래핑
def get_retriever_from_source(
    vectorstore: FAISS,
    all_docs: Optional[List[Document]] = None,
    debug: bool = False,
) -> SmartRetriever:
    """
    기존과 동일하게 search_type='similarity'를 유지하면서,
    SmartRetriever가 외곽에서 동적 힌트, BM25 보강, (가능시) 리랭킹을 수행.
    """
    # NOTE: 여기서 search_type을 바꾸지 않습니다.
    base = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6,                 # SmartRetriever가 내부에서 k를 (6,10,16)로 점증 시도
            "score_threshold": None # 과도한 필터링 방지
        },
    )

    cfg = SmartRetrieverConfig(
        top_n=4,
        escalate_k=(6, 10, 16),
        use_bm25=True,
        use_reranker=True,
        min_domain_hits=1,
        debug=debug,
    )
    return SmartRetriever(
        base_retriever=base,
        all_docs=all_docs or [],
        config=cfg,
        logger=None,  # Streamlit logger를 넣고 싶으면 st.sidebar 등 전달 가능
    )

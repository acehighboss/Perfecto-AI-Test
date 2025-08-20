from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

def _split_docs(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Document]:
    """문서를 청크 단위로 분할."""
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _get_embeddings(model: Optional[str] = None):
    """OpenAI 임베딩 래퍼. 환경변수 EMBEDDINGS_MODEL을 우선 사용."""
    model_name = model or os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name)


def _build_vectorstore_with_chroma(
    docs: List[Document],
    embeddings,
    persist_directory: Optional[str] = None,
):
    """Chroma 기반 벡터스토어 구성."""
    persist_dir = persist_directory or os.environ.get("CHROMA_PERSIST_DIR")
    # persist_dir가 주어지면 영속 모드, 없으면 메모리 모드
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )


def _apply_search_kwargs(
    retriever: VectorStoreRetriever,
    *,
    search_type: str,
    top_k: Optional[int],
    score_threshold: Optional[float],
):
    """Retriever에 search_kwargs 적용 (k, score_threshold 등)."""
    search_kwargs: Dict[str, Any] = dict(retriever.search_kwargs or {})
    if top_k is not None:
        try:
            search_kwargs["k"] = int(top_k)
        except Exception:
            # 정수 변환 실패 시 무시 (안전성 우선)
            pass

    # similarity_score_threshold 모드일 때 score_threshold 사용
    # 일부 버전에서는 search_type="similarity"에서도 score_threshold를 지원
    if score_threshold is not None and search_type in (
        "similarity_score_threshold",
        "similarity",
    ):
        try:
            search_kwargs["score_threshold"] = float(score_threshold)
        except Exception:
            pass

    retriever.search_kwargs = search_kwargs
    return retriever


# --------------------------- 외부 공개 API ---------------------------

def get_retriever_from_source(
    docs: List[Document],
    *,
    # 문서 청크 설정
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    # 임베딩/백엔드 설정
    embedding_model: Optional[str] = None,
    vectorstore_backend: str = "chroma",  # 현재 chroma만 사용 (의존성 안전)
    persist_directory: Optional[str] = None,
    # 검색 설정
    search_type: str = "similarity",  # "similarity" | "similarity_score_threshold" | "mmr" 등
    top_k: Optional[int] = None,      # ★ main.py에서 넘긴 top_k 안전 지원
    score_threshold: Optional[float] = None,
    # 향후 확장 인자 흡수 (예상 못한 키워드가 들어와도 오류 없이 무시)
    **_: Any
) -> VectorStoreRetriever:
    """
    main.py에서 호출하는 RAG 진입점.

    Parameters
    ----------
    docs : List[Document]
        업로드/크롤링 등으로 얻은 문서 리스트
    chunk_size, chunk_overlap : int
        청크 분할 설정
    embedding_model : Optional[str]
        OpenAI 임베딩 모델명 (기본: 환경변수 EMBEDDINGS_MODEL 또는 text-embedding-3-small)
    vectorstore_backend : str
        현재는 'chroma'만 지원 (추가 의존성 회피)
    persist_directory : Optional[str]
        Chroma 영속 디렉터리. None이면 메모리 모드
    search_type : str
        'similarity', 'similarity_score_threshold', 'mmr' 등
    top_k : Optional[int]
        검색 개수 제한. None이면 백엔드 기본값
    score_threshold : Optional[float]
        similarity_score_threshold 모드에서 임계값
    **_ : Any
        예기치 못한 키워드 인자를 흡수하여 TypeError 방지

    Returns
    -------
    VectorStoreRetriever
        LangChain 호환 Retriever 객체
    """
    # 1) 문서 분할
    splits = _split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2) 임베딩 로드
    embeddings = _get_embeddings(embedding_model)

    # 3) 벡터스토어 구성 (기본 Chroma)
    backend = (vectorstore_backend or "chroma").lower()
    if backend != "chroma":
        # 안전을 위해 현재는 chroma만 허용
        backend = "chroma"

    vs = _build_vectorstore_with_chroma(splits, embeddings, persist_directory=persist_directory)

    # 4) Retriever로 래핑 + 검색 파라미터 적용
    retriever = vs.as_retriever(search_type=search_type)
    retriever = _apply_search_kwargs(
        retriever,
        search_type=search_type,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    return retriever

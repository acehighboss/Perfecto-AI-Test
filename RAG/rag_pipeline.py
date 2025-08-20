from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Union

import os
import logging

logger = logging.getLogger(__name__)

# ---- LangChain 타입 호환 ------------------------------------------------------
try:
    from langchain_core.documents import Document
except Exception:  # 구버전 호환
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# ---- 선택적 의존성 -----------------------------------------------------------
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None  # type: ignore

# 벡터스토어: FAISS 우선, 불가하면 Chroma로 폴백 시도
FAISS = None
CHROMA = None
try:
    from langchain_community.vectorstores import FAISS as _FAISS  # type: ignore
    FAISS = _FAISS
except Exception:
    pass

try:
    from langchain_community.vectorstores import Chroma as _Chroma  # type: ignore
    CHROMA = _Chroma
except Exception:
    pass

# 텍스트 분할기
RecursiveCharacterTextSplitter = None
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCTS  # type: ignore
    RecursiveCharacterTextSplitter = _RCTS
except Exception:
    pass


# -----------------------------------------------------------------------------
# 내부 유틸
# -----------------------------------------------------------------------------
def _ensure_documents(
    docs: Optional[List[Document]] = None,
    texts: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> List[Document]:
    """입력으로부터 Document 리스트를 표준화."""
    if docs and len(docs) > 0:
        return docs

    if texts:
        metas = metadatas or [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError("texts와 metadatas 길이가 일치해야 합니다.")
        return [Document(page_content=t or "", metadata=m or {}) for t, m in zip(texts, metas)]

    raise ValueError("docs 또는 texts(및 metadatas)를 제공해야 합니다.")


def _split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
) -> List[Document]:
    """문서를 청크로 분할. 분할기가 없으면 원문 그대로 사용."""
    if not docs:
        return []

    if RecursiveCharacterTextSplitter is None:
        # 의존성이 없으면 분할 생략
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


def _get_default_embeddings(embeddings: Any = None):
    """OpenAI 임베딩 기본 제공. 없으면 에러."""
    if embeddings is not None:
        return embeddings
    if OpenAIEmbeddings is None:
        raise ImportError(
            "OpenAIEmbeddings를 사용할 수 없습니다. "
            "pip install langchain-openai 후 OPENAI_API_KEY를 설정하거나 embeddings 인자를 주입하세요."
        )
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _build_vectorstore(
    docs: List[Document],
    embeddings: Any,
    prefer: str = "faiss",
    persist_dir: Optional[str] = None,
):
    """FAISS 우선, 불가시 Chroma 사용."""
    if prefer == "faiss" and FAISS is not None:
        return FAISS.from_documents(docs, embeddings)
    if CHROMA is not None:
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            return CHROMA.from_documents(docs, embeddings, persist_directory=persist_dir)
        return CHROMA.from_documents(docs, embeddings)
    if FAISS is not None:
        return FAISS.from_documents(docs, embeddings)

    raise ImportError(
        "사용 가능한 벡터스토어가 없습니다. pip install langchain-community faiss-cpu (또는 chromadb)를 설치하세요."
    )


# -----------------------------------------------------------------------------
# 공개 API
# -----------------------------------------------------------------------------
def get_retriever_from_source(
    docs: Optional[List[Document]] = None,
    *,
    texts: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embeddings: Any = None,
    k: int = 6,
    fetch_k: int = 20,
    search_type: str = "mmr",  # "similarity" | "mmr"
    prefer_store: str = "faiss",
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
    persist_dir: Optional[str] = None,
):
    """
    업로드/크롤링한 소스로부터 retriever 생성.
    - docs 또는 (texts+metadatas) 입력 지원
    - FAISS/Chroma 중 가용한 백엔드 자동 선택
    """
    base_docs = _ensure_documents(docs=docs, texts=texts, metadatas=metadatas)
    chunks = _split_documents(base_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("분할된 문서가 없습니다. 입력 내용을 확인하세요.")

    emb = _get_default_embeddings(embeddings)
    vs = _build_vectorstore(chunks, emb, prefer=prefer_store, persist_dir=persist_dir)

    retriever = vs.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    return retriever


# === Backward-compat shim for UI import in main.py ============================
def create_retriever(*args, **kwargs):
    """
    main.py가 기대하는 이름. 기존 get_retriever_from_source가 있으면 그걸 호출하고,
    실패 시 docs 기반의 간단 빌드로 폴백.
    """
    try:
        return get_retriever_from_source(*args, **kwargs)
    except TypeError:
        # 다른 시그니처로 호출된 경우를 대비해 폴백 구현
        # 예: create_retriever(docs, k=6, search_type="mmr", fetch_k=20, embeddings=None)
        docs = kwargs.get("docs") if "docs" in kwargs else (args[0] if args else None)
        if docs is None:
            raise
        k = kwargs.get("k", 6)
        search_type = kwargs.get("search_type", "mmr")
        fetch_k = kwargs.get("fetch_k", 20)
        embeddings = kwargs.get("embeddings", None)
        return get_retriever_from_source(
            docs=docs,
            embeddings=embeddings,
            k=k,
            fetch_k=fetch_k,
            search_type=search_type,
        )

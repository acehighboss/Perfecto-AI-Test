# RAG/rag_pipeline.py
# ------------------------------------------------------------------------------
# - chromadb 미설치/오류 시에도 절대 예외로 죽지 않도록 완전 폴백
# - Chroma import를 모듈 상단이 아닌 "함수 내부 try"에서만 수행
# - top_k / score_threshold 안전 반영
# - 파일/폴더 구조 변경 없음
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
import os
import importlib.util
from typing import List, Optional, Dict, Any, Tuple

from pydantic import Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever


# =========================== 인메모리 폴백 리트리버 ============================

class SimpleEmbeddingRetriever(BaseRetriever):
    """chromadb가 없어도 동작하는 간단한 인메모리 임베딩 리트리버."""
    texts: List[str] = Field(default_factory=list)
    metadatas: List[Dict[str, Any]] = Field(default_factory=list)
    vectors: List[List[float]] = Field(default_factory=list)
    embeddings: Any = Field(default=None)
    k: int = 4
    score_threshold: Optional[float] = None
    search_type: str = "similarity"  # "similarity" | "similarity_score_threshold"

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-12
        nb = math.sqrt(sum(y * y for y in b)) or 1e-12
        return dot / (na * nb)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        qvec = self.embeddings.embed_query(query)
        scored = [(i, self._cosine_sim(qvec, v)) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda t: t[1], reverse=True)

        out: List[Document] = []
        k = max(1, int(self.k or 4))
        for idx, sc in scored:
            if (
                self.score_threshold is not None
                and self.search_type in ("similarity", "similarity_score_threshold")
                and sc < self.score_threshold
            ):
                continue
            meta = dict(self.metadatas[idx]) if self.metadatas and idx < len(self.metadatas) else {}
            meta["_score"] = sc
            out.append(Document(page_content=self.texts[idx], metadata=meta))
            if len(out) >= k:
                break
        return out

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return self._get_relevant_documents(query)


# ================================ 내부 유틸 ================================

def _split_docs(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def _get_embeddings(model: Optional[str] = None):
    model_name = model or os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name)


def _try_build_chroma(docs: List[Document], embeddings, persist_directory: Optional[str]):
    """
    chromadb가 설치된 경우에만 Chroma를 시도한다.
    여기서 발생하는 ImportError/기타 예외는 상위에서 폴백 처리한다.
    """
    has_chromadb = importlib.util.find_spec("chromadb") is not None
    if not has_chromadb:
        raise ImportError("chromadb not installed")

    # 내부 import (모듈 상단이 아닌 여기서만 시도)
    from langchain_community.vectorstores import Chroma  # noqa: WPS433

    persist_dir = persist_directory or os.environ.get("CHROMA_PERSIST_DIR")
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    return Chroma.from_documents(documents=docs, embedding=embeddings)


def _build_inmemory_retriever(
    docs: List[Document],
    embeddings,
    *,
    top_k: Optional[int],
    score_threshold: Optional[float],
    search_type: str,
) -> SimpleEmbeddingRetriever:
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectors = embeddings.embed_documents(texts) if texts else []
    return SimpleEmbeddingRetriever(
        texts=texts,
        metadatas=metadatas,
        vectors=vectors,
        embeddings=embeddings,
        k=int(top_k) if top_k is not None else 4,
        score_threshold=float(score_threshold) if score_threshold is not None else None,
        search_type=search_type,
    )


def _apply_search_kwargs_to_retriever(
    retriever: VectorStoreRetriever,
    *,
    search_type: str,
    top_k: Optional[int],
    score_threshold: Optional[float],
):
    search_kwargs: Dict[str, Any] = dict(retriever.search_kwargs or {})
    if top_k is not None:
        try:
            search_kwargs["k"] = int(top_k)
        except Exception:
            pass
    if score_threshold is not None and search_type in ("similarity_score_threshold", "similarity"):
        try:
            search_kwargs["score_threshold"] = float(score_threshold)
        except Exception:
            pass
    retriever.search_kwargs = search_kwargs
    return retriever


# ============================== 외부 공개 API ==============================

def get_retriever_from_source(
    docs: List[Document],
    *,
    # 문서 청크
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    # 임베딩/백엔드
    embedding_model: Optional[str] = None,
    vectorstore_backend: str = "chroma",   # "chroma" 권장, 없으면 자동 폴백
    persist_directory: Optional[str] = None,
    # 검색 설정
    search_type: str = "similarity",
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    # 확장 인자 흡수
    **_: Any
):
    """
    - 기본은 Chroma 사용을 시도 (chromadb 없거나 실패 시 인메모리 폴백)
    - top_k / score_threshold 적용
    """
    splits = _split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = _get_embeddings(embedding_model)

    backend = (vectorstore_backend or "chroma").lower()

    # 1) Chroma 우선 시도
    if backend == "chroma":
        try:
            vs = _try_build_chroma(splits, embeddings, persist_directory)
            retriever = vs.as_retriever(search_type=search_type)
            retriever = _apply_search_kwargs_to_retriever(
                retriever, search_type=search_type, top_k=top_k, score_threshold=score_threshold
            )
            return retriever
        except Exception as e:
            print(f"[WARN] Chroma 사용 불가({e!r}). 인메모리 리트리버로 폴백합니다.")

    # 2) 폴백: 인메모리
    return _build_inmemory_retriever(
        splits, embeddings, top_k=top_k, score_threshold=score_threshold, search_type=search_type
    )

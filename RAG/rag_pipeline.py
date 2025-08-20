# RAG/rag_pipeline.py
# ------------------------------------------------------------------------------
# RAG 파이프라인: main.py에서 호출하는 진입점(get_retriever_from_source)을 제공.
# - top_k 인자를 안전하게 지원 (기존 TypeError 해결)
# - chromadb 미설치 시 자동 폴백: 간단한 인메모리 임베딩 기반 Retriever
# - 파일/폴더 이름 및 구조는 변경 없음
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
import os
from typing import List, Optional, Dict, Any, Tuple

from pydantic import Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever

# Chroma 클래스는 가져올 수 있으나, 실제 인스턴스화 시 chromadb 없으면 ImportError 발생
from langchain_community.vectorstores import Chroma


# =========================== 인메모리 폴백 리트리버 ============================

class SimpleEmbeddingRetriever(BaseRetriever):
    """chromadb가 없는 환경에서도 동작하는 간단한 임베딩 기반 Retriever."""
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

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        qvec = self.embeddings.embed_query(query)
        scores: List[Tuple[int, float]] = []
        for i, v in enumerate(self.vectors):
            scores.append((i, self._cosine_sim(qvec, v)))
        scores.sort(key=lambda t: t[1], reverse=True)

        out: List[Document] = []
        for idx, sc in scores:
            if (
                self.score_threshold is not None
                and self.search_type in ("similarity", "similarity_score_threshold")
                and sc < self.score_threshold
            ):
                continue
            meta = dict(self.metadatas[idx]) if self.metadatas and idx < len(self.metadatas) else {}
            meta["_score"] = sc
            out.append(Document(page_content=self.texts[idx], metadata=meta))
            if len(out) >= max(1, int(self.k or 4)):
                break
        return out

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any = None
    ) -> List[Document]:
        # 간단히 동기 구현을 재사용
        return self._get_relevant_documents(query)


# ================================ 내부 유틸 ================================

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
    """Chroma 기반 벡터스토어 구성. chromadb 미설치 시 ImportError 발생 -> 상위에서 캐치."""
    persist_dir = persist_directory or os.environ.get("CHROMA_PERSIST_DIR")
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


def _build_inmemory_retriever(
    docs: List[Document],
    embeddings,
    *,
    top_k: Optional[int],
    score_threshold: Optional[float],
    search_type: str,
) -> SimpleEmbeddingRetriever:
    """외부 벡터 DB 없이 인메모리로 동작하는 폴백 리트리버 구성."""
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
    """VectorStoreRetriever 타입에 search_kwargs 적용."""
    search_kwargs: Dict[str, Any] = dict(retriever.search_kwargs or {})
    if top_k is not None:
        try:
            search_kwargs["k"] = int(top_k)
        except Exception:
            pass

    # similarity_score_threshold / similarity 모드에서 score_threshold 사용
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
    # 문서 청크 설정
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    # 임베딩/백엔드 설정
    embedding_model: Optional[str] = None,
    vectorstore_backend: str = "chroma",  # 기본 chroma 시도 후 실패 시 인메모리 폴백
    persist_directory: Optional[str] = None,
    # 검색 설정
    search_type: str = "similarity",  # "similarity" | "similarity_score_threshold" | "mmr" 등
    top_k: Optional[int] = None,      # ★ main.py에서 넘긴 top_k 안전 지원
    score_threshold: Optional[float] = None,
    # 향후 확장 인자 흡수 (예상 못한 키워드가 들어와도 오류 없이 무시)
    **_: Any
):
    """
    main.py에서 호출하는 RAG 진입점.

    - chroma 사용 시 chromadb가 없으면 자동으로 인메모리 리트리버로 폴백
    - top_k / score_threshold 적용
    """
    # 1) 문서 분할
    splits = _split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2) 임베딩 로드
    embeddings = _get_embeddings(embedding_model)

    # 3) 우선 'chroma' 시도, 실패하면 인메모리로 폴백
    backend = (vectorstore_backend or "chroma").lower()

    if backend == "chroma":
        try:
            vs = _build_vectorstore_with_chroma(splits, embeddings, persist_directory=persist_directory)
            retriever = vs.as_retriever(search_type=search_type)
            retriever = _apply_search_kwargs_to_retriever(
                retriever,
                search_type=search_type,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            return retriever
        except ImportError:
            # chromadb 미설치 -> 폴백
            # (Streamlit 경고는 라이브러리 의존 제거를 위해 print만 사용)
            print("[WARN] chromadb가 없어 Chroma 백엔드를 사용할 수 없습니다. 인메모리 리트리버로 폴백합니다.")
        except Exception as e:
            # 기타 에러도 폴백 (운영 안정성 우선)
            print(f"[WARN] Chroma 초기화 중 예외 발생: {e}. 인메모리 리트리버로 폴백합니다.")

    # 4) 인메모리 폴백 (또는 명시적으로 backend != chroma 인 경우)
    return _build_inmemory_retriever(
        splits,
        embeddings,
        top_k=top_k,
        score_threshold=score_threshold,
        search_type=search_type,
    )

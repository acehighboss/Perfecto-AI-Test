from __future__ import annotations

from typing import List, Dict, Any, Iterable, Optional
import re
from dataclasses import dataclass

from langchain_core.documents import Document


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _doc_to_sentences(doc: Document) -> List[Dict[str, Any]]:
    """
    (유지) 문장 단위 쪼개기. 여기서는 문장화는 하지 않고 메타만 세팅 예시를 남겨둡니다.
    실제 문장 추출은 chain_builder.extract_relevant_sentences 에서 수행합니다.
    """
    # === 패치 포인트 ===
    # source/url/page/timecode 메타 필드를 표준화해 downstream에서 안정적으로 접근 가능
    items = [{
        "text": _normalize_space(doc.page_content or ""),
        "source": doc.metadata.get("source") or doc.metadata.get("url") or "",
        "page": doc.metadata.get("page"),
        "timecode": doc.metadata.get("timecode"),
    }]
    return items


# ===== 간단한 텍스트 리트리버 =====

@dataclass
class SimpleTextRetriever:
    docs: List[Document]
    top_k: int = 6

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", (text or "").lower().strip())
        text = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+", " ", text)
        return [t for t in text.split() if t]

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0
        qset = set(query_tokens)
        overlap = sum(1 for t in doc_tokens if t in qset)
        return overlap / (1.0 + len(doc_tokens) ** 0.25)

    def get_relevant_documents(self, query: str) -> List[Document]:
        q = self._tokenize(query)
        scored = []
        for d in self.docs:
            tokens = self._tokenize(d.page_content or "")
            s = self._score(q, tokens)
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.top_k]]


def get_retriever_from_source(source: Iterable[Document], top_k: int = 6) -> SimpleTextRetriever:
    """
    레포 호환용. 벡터DB가 없더라도 간단히 동작하도록 텍스트 기반 리트리버 제공.
    - source: Document iterable
    """
    docs = list(source)
    return SimpleTextRetriever(docs=docs, top_k=top_k)


# 과거 import 호환 (예: from RAG.rag_pipeline import create_retriever)
create_retriever = get_retriever_from_source

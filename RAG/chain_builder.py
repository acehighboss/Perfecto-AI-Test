from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import math

import numpy as np

# ---- LangChain 타입 호환 ------------------------------------------------------
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document  # type: ignore
    except Exception:
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

# 선택적 LLM/임베딩
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except Exception:
    OpenAIEmbeddings = None  # type: ignore
    ChatOpenAI = None  # type: ignore


# -----------------------------------------------------------------------------
# 문장 분할 & 포맷 유틸
# -----------------------------------------------------------------------------
def _sent_tokenize(text: str) -> List[str]:
    """언어-무관한 안전 분할: 개행과 문장부호 기준으로 단순 분할."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    # 큰 단위 개행 먼저
    blocks = re.split(r"\n{2,}|\r{2,}", text)
    out: List[str] = []
    for blk in blocks:
        parts = re.split(r"(?<=[\.!?]|[。！？])\s+", blk)
        for p in parts:
            s = p.strip()
            if s:
                out.append(s)
    return out


def _fmt_ts(seconds: Any) -> str:
    try:
        sec = int(float(seconds))
    except Exception:
        return ""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _sentence_rows_from_doc(doc: Document) -> List[Dict[str, Any]]:
    """
    Document → 문장 단위 row 변환
    반환 row: {text, source, where, metadata}
    where: PDF면 'p.N', 자막이면 't=MM:SS'
    """
    rows: List[Dict[str, Any]] = []
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("source") or meta.get("url") or meta.get("path") or ""

    page = meta.get("page") or meta.get("page_number")
    ts = meta.get("start") or meta.get("timestamp") or meta.get("start_time")

    for sent in _sent_tokenize(getattr(doc, "page_content", "") or ""):
        if not sent.strip():
            continue
        where = None
        if page is not None:
            where = f"p.{page}"
        elif ts is not None:
            where = f"t={_fmt_ts(ts)}"
        rows.append(
            {
                "text": sent.strip(),
                "source": src,
                "where": where,
                "metadata": meta,
            }
        )
    return rows


# -----------------------------------------------------------------------------
# 스코어링(임베딩 또는 간이 토큰 중복)
# -----------------------------------------------------------------------------
def _embed_texts(texts: List[str]) -> np.ndarray:
    if OpenAIEmbeddings is None:
        # 폴백: 텍스트 길이를 surrogate feature로 사용 (간이 스코어에 활용)
        return np.array([[len(set(t.lower().split()))] for t in texts], dtype=float)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vecs = emb.embed_documents(texts)
    return np.array(vecs, dtype=float)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# -----------------------------------------------------------------------------
# 공개 함수 1) 질문에 필요한 "최소 문장"만 추출
# -----------------------------------------------------------------------------
def extract_relevant_sentences(
    question: str,
    retrieved_docs: List[Document],
    top_k: int = 8,
    max_per_source: int = 3,
    min_chars: int = 12,
) -> List[Dict[str, Any]]:
    """
    검색된 Document들에서 질문과 가장 관련 높은 문장만 추출.
    반환: [{text, score, source, where, metadata}]
    """
    rows: List[Dict[str, Any]] = []
    for d in retrieved_docs or []:
        rows.extend(_sentence_rows_from_doc(d))

    rows = [r for r in rows if len(r["text"]) >= min_chars]
    if not rows:
        return []

    all_texts = [r["text"] for r in rows]
    if OpenAIEmbeddings is None:
        # 폴백: 질문 토큰과의 중복수 기반 간이 점수
        q_tokens = set(question.lower().split())

        def cheap_score(t: str) -> float:
            s_tokens = set(t.lower().split())
            inter = len(q_tokens & s_tokens)
            return inter / math.sqrt(len(s_tokens) + 1e-6)

        scores = np.array([cheap_score(t) for t in all_texts], dtype=float)
    else:
        q_vec = _embed_texts([question])[0]
        row_vecs = _embed_texts(all_texts)
        scores = np.array([_cosine(q_vec, v) for v in row_vecs], dtype=float)

    for r, s in zip(rows, scores):
        r["score"] = float(s)

    rows.sort(key=lambda x: x["score"], reverse=True)
    per_src: Dict[str, int] = {}
    picked: List[Dict[str, Any]] = []
    for r in rows:
        src = r["source"] or "unknown"
        if per_src.get(src, 0) >= max_per_source:
            continue
        picked.append(r)
        per_src[src] = per_src.get(src, 0) + 1
        if len(picked) >= top_k:
            break

    return picked


# -----------------------------------------------------------------------------
# 공개 함수 2) 추려진 문장만으로 답변 구성
# -----------------------------------------------------------------------------
def build_answer_from_sentences(
    question: str,
    sentences: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    language: str = "ko",
) -> str:
    """
    추려진 문장만 근거로 **과추론 없이** 답변 작성.
    LLM 가능 시 ChatOpenAI 사용, 아니면 간단 요약으로 대체.
    """
    if not sentences:
        return "해당 질문에 답할 수 있는 근거 문장을 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요."

    if ChatOpenAI is not None:
        name = model_name or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
        llm = ChatOpenAI(model=name, temperature=0)
        lines = []
        for i, s in enumerate(sentences, 1):
            where = f" {s['where']}" if s.get("where") else ""
            lines.append(f"[{i}] {s['text']}{where}")
        evidence_block = "\n".join(lines)

        system = (
            "당신은 엄격한 RAG 어시스턴트입니다. 사용자 질문에 대해 아래 '증거 문장'만을 근거로, "
            "추가 지식 없이 한국어로 간결하고 정확하게 답하세요. "
            "모호하면 '문맥 불충분'이라고 답하세요. "
            "숫자/사실은 증거 문장에 나온 값만 사용하세요."
        )
        user_prompt = (
            f"질문: {question}\n\n"
            f"증거 문장들:\n{evidence_block}\n\n"
            "규칙:\n"
            "- 위 문장 안에서만 답을 구성합니다.\n"
            "- 필요시 문장 번호를 대괄호로 인용하세요 (예: [1], [3]).\n"
            "- 마지막 줄에 '근거: [1], [2]'처럼 인용 번호만 표기하세요."
        )
        resp = llm.invoke([{"role": "system", "content": system},
                           {"role": "user", "content": user_prompt}])
        return getattr(resp, "content", str(resp))

    # LLM 불가: 간단 요약 + 근거 라벨
    uniq_tags: List[str] = []
    tag_map: List[str] = []
    for s in sentences:
        tag = (s.get("source") or "").strip()
        where = (s.get("where") or "").strip()
        if where:
            tag = f"{tag} {where}".strip()
        if tag and tag not in uniq_tags:
            uniq_tags.append(tag)
        tag_map.append(tag)

    bullets = "\n".join([f"- {s['text']}" for s in sentences[:8]])
    cite = ", ".join([f"[{i+1}]" for i in range(min(len(sentences), 8))])
    srcs = ", ".join([f"[{i+1}] {t}" for i, t in enumerate(uniq_tags[:6])])
    tail = f"\n\n근거: {cite}"
    if srcs:
        tail += f"\n출처: {srcs}"
    return f"{bullets}{tail}"


# -----------------------------------------------------------------------------
# 기존 API 유지: 체인 빌더 (간단 래퍼)
# -----------------------------------------------------------------------------
class _SimpleRAGChain:
    """UI 코드 호환용: .invoke({'input': 질문}) 형태 지원."""
    def __init__(self, retriever, model_name: Optional[str] = None, language: str = "ko"):
        self.retriever = retriever
        self.model_name = model_name
        self.language = language

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question") or inputs.get("input") or inputs.get("query") or ""
        if not question:
            return {"answer": "질문이 비어있습니다.", "evidence": []}
        docs = self.retriever.get_relevant_documents(question)
        ev = extract_relevant_sentences(question, docs)
        answer = build_answer_from_sentences(question, ev, model_name=self.model_name, language=self.language)
        return {"answer": answer, "evidence": ev}


def get_default_chain(retriever, model_name: Optional[str] = None, language: str = "ko"):
    """
    기본 RAG 체인 반환.
    - 반환 객체는 .invoke({'input': '질문'})로 호출 가능
    - 결과 dict: {'answer': str, 'evidence': List[...]}
    """
    return _SimpleRAGChain(retriever, model_name=model_name, language=language)


def get_conversational_rag_chain(
    retriever,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
    language: str = "ko",
):
    """
    대화형 RAG 체인.
    - chat_history는 [{'role':'user'|'assistant','content':'...'}] 형태(선택)
    - 질문은 .invoke({'input': '질문', 'history': chat_history})로 호출 가능
    """
    class _ConvChain(_SimpleRAGChain):
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            q = inputs.get("question") or inputs.get("input") or ""
            hist = inputs.get("history") or chat_history or []
            # 간단한 재질문 보정(LLM 가능 시 살짝 리라이트)
            if ChatOpenAI is not None and q and hist:
                llm = ChatOpenAI(model=model_name or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini", temperature=0)
                prompt = (
                    "아래 대화 맥락을 고려하여 사용자의 마지막 질문을 검색 친화적으로 간결히 재작성하세요.\n\n"
                    f"대화:\n{hist}\n\n"
                    f"마지막 질문: {q}\n"
                    "재작성:"
                )
                try:
                    q = getattr(llm.invoke(prompt), "content", q) or q
                except Exception:
                    pass

            docs = self.retriever.get_relevant_documents(q or inputs.get("question") or "")
            ev = extract_relevant_sentences(q, docs)
            answer = build_answer_from_sentences(q, ev, model_name=self.model_name, language=self.language)
            return {"answer": answer, "evidence": ev}

    return _ConvChain(retriever, model_name=model_name, language=language)

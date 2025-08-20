# RAG/chain_builder.py
# ------------------------------------------------------------------------------
# - extract_relevant_sentences(): 질문-문장 임베딩 유사도 + 중복 억제로
#   "답변에 필요한 최소 문장 집합"을 자동 선택 (개수 옵션 불필요)
# - YouTube 자막은 시간코드, PDF는 페이지 번호를 출처에 자동 포함
# - build_answer_from_sentences(): 선택된 문장들만으로 간결 답변 생성(또는 LLM에 넘길 컨텍스트 구성)
# - get_conversational_rag_chain/get_default_chain 시그니처는 유지(내부는 경량)
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
import re
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ---------------------------- 문장 토크나이저 ----------------------------

_SENT_SPLIT = re.compile(r"(?<=[\.?!])\s+|\n+")

def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    # 너무 짧은 토큰(예: 한 단어)은 제거
    return [s for s in sents if len(s) >= 3]


# ---------------------------- 유사도 및 랭킹 ----------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def _format_locator(meta: Dict[str, Any]) -> str:
    """
    PDF: meta.get("page") 또는 meta.get("page_number")
    YouTube: meta.get("start") 초 -> mm:ss
    일반 URL: 없음
    """
    # YouTube 시간코드
    if "start" in meta:
        try:
            sec = int(float(meta["start"]))
            mm = sec // 60
            ss = sec % 60
            return f"{mm:02d}:{ss:02d}"
        except Exception:
            pass

    # PDF 페이지
    pg = meta.get("page") or meta.get("page_number")
    if isinstance(pg, int):
        return f"p.{pg}"

    return ""


def extract_relevant_sentences(
    question: str,
    docs: List[Document],
    *,
    min_score: float = 0.43,
    max_per_doc: int = 6,
) -> List[Dict[str, Any]]:
    """
    질문과 관련도가 높은 문장만 "최소 집합"으로 추려서 반환.
    - min_score: 유사도 임계값(낮추면 더 많이 나옴)
    - max_per_doc: 문서당 최대 선택 문장 수(너무 길어지는 것 방지)

    Return: [{ "text": str, "source": str, "locator": str, "score": float }]
    """
    if not docs or not question.strip():
        return []

    embed = OpenAIEmbeddings(model="text-embedding-3-small")
    qv = embed.embed_query(question)

    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for d in docs:
        sents = _split_sentences(d.page_content or "")
        if not sents:
            continue
        sv = embed.embed_documents(sents)
        scored = [(_cosine(qv, v), s, d.metadata or {}) for s, v in zip(sents, sv)]
        # 문서 내 상위 문장 제한
        scored.sort(key=lambda t: t[0], reverse=True)
        top = [t for t in scored if t[0] >= min_score][:max_per_doc]
        candidates.extend(top)

    # 전체 중복 억제(같은/유사 문장 제거)
    # - 간단 중복 기준: 공백 제거 후 30자 이상 동일하면 중복으로 간주
    seen = set()
    unique: List[Tuple[float, str, Dict[str, Any]]] = []
    for sc, sent, meta in sorted(candidates, key=lambda t: t[0], reverse=True):
        key = re.sub(r"\s+", "", sent)
        if len(key) >= 30 and key in seen:
            continue
        seen.add(key)
        unique.append((sc, sent, meta))

    # “최소 집합” 휴리스틱:
    # - 점수 상위부터 채우되, 서로 겹치는 의미(문장 시작~끝의 60% 이상 동일)면 스킵
    final: List[Dict[str, Any]] = []
    for sc, sent, meta in unique:
        if any(_overlap_ratio(sent, x["text"]) >= 0.6 for x in final):
            continue
        locator = _format_locator(meta)
        src = meta.get("source") or meta.get("url") or meta.get("file_path") or ""
        final.append({"text": sent, "source": src, "locator": locator, "score": float(sc)})

        # 전체가 너무 커지는 것 방지(실무적으로 12~18문장 내외면 충분)
        if len(final) >= 16:
            break

    return final


def _overlap_ratio(a: str, b: str) -> float:
    """아주 단순한 겹침률 측정(토큰 미사용)."""
    a_ = re.sub(r"\s+", " ", a.strip())
    b_ = re.sub(r"\s+", " ", b.strip())
    if not a_ or not b_:
        return 0.0
    # 짧은 쪽 길이 대비 공통 prefix 길이
    prefix = 0
    for x, y in zip(a_, b_):
        if x == y:
            prefix += 1
        else:
            break
    return prefix / max(1, min(len(a_), len(b_)))


# ------------------------- 답변 조립 / 체인 헬퍼 -------------------------

def build_answer_from_sentences(question: str, picked: List[Dict[str, Any]]) -> str:
    """
    선택된 문장만으로 간결한 요약/답변 초안 생성.
    (LLM 호출 전 컨텍스트로 써도 되고, 그 자체로 “근거 위주” 답변으로도 사용 가능)
    """
    if not picked:
        return "해당 질문에 직접적으로 답할 수 있는 근거 문장을 찾지 못했습니다."
    # 간단한 압축: 상위 문장 6~8개로 요약 구성
    body = " ".join(p["text"] for p in picked[:8])
    return body.strip()


def get_conversational_rag_chain(*args, **kwargs):
    """
    레포에서 참조하는 시그니처 유지용 헬퍼.
    현재 로직에선 체인 구성 없이도 동작 가능하므로 패스스루 형태로 둡니다.
    필요 시 여기에 LLM + 프롬프트를 얹어 사용하세요.
    """
    return None


def get_default_chain(*args, **kwargs):
    """시그니처 유지용 더미."""
    return None

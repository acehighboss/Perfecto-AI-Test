from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import re
import math
from collections import Counter, defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel


# ===== 프롬프트: 근거 기반 답변 =====
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 한국어 조교입니다. 반드시 제공된 '근거 문장' 안에서만 답하세요. "
     "근거에 없는 정보는 추정하지 마세요. 답변은 간결하게 하되, 질문이 '이름/특징'을 물으면 "
     "1) 한 문단 요약 답변, 2) '핵심 특징' 불릿 2~4개를 제시합니다. "
     "문장 끝에 [S#] 표기를 사용해 인용 근거를 명확히 하세요."),
    ("human",
     "질문: {question}\n\n근거 문장:\n{evidence}\n\n"
     "지침:\n- 5문장 이내 요약\n- 불필요한 수사는 금지\n- [S#]는 실제 근거 문장 번호만 사용")
])

# ===== LLM 이름/키워드 휴리스틱 =====
LLM_NAME_PATTERNS = [
    r"\bA\.?X\b", r"에이닷\s*엑스", r"\bAX\s*\d(?:\.\d)?\b",
    r"Telco\s*-?\s*LLM", r"텔코\s*LLM"
]


def _make_evidence_block(sentences: List[Dict[str, Any]]) -> str:
    """문장 리스트를 [S#] 형식과 출처/페이지/타임코드로 포매팅."""
    lines = []
    for i, s in enumerate(sentences, 1):
        src = s.get("source", "")
        extra = []
        if s.get("page"):
            extra.append(f"p.{s['page']}")
        if s.get("timecode"):
            extra.append(str(s["timecode"]))
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"[S{i}] {s['text']} (출처: {src}{suffix})")
    return "\n".join(lines)


def _contains_answer_terms(sentences: List[Dict[str, Any]], patterns: List[str]) -> bool:
    """근거 문장 덩어리 속에 핵심 정답 키워드가 있는지 휴리스틱 확인."""
    blob = " ".join(s.get("text", "") for s in sentences)
    for pat in patterns:
        if re.search(pat, blob, flags=re.IGNORECASE):
            return True
    return False


def _safe_llm_invoke(llm: Optional[BaseChatModel], prompt: ChatPromptTemplate, inputs: Dict[str, Any]) -> str:
    """LLM이 None이거나 실패 시 안전한 폴백."""
    if llm is None:
        return (
            "⚠️ LLM이 초기화되지 않아 생성 요약을 수행할 수 없습니다. "
            "환경설정(예: API 키, 모델명)을 확인해 주세요."
        )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)


def build_answer_from_sentences(
    llm: Optional[BaseChatModel],
    question: str,
    sentences: List[Dict[str, Any]],
    *,
    allow_generation: bool = True,
    require_terms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    근거 문장들로부터 최종 답변/요약을 생성.
    sentences: [{"text": str, "source": str, "page": int|None, "timecode": str|None}, ...]
    return: {"answer": str, "sentences": [...], "answerable": bool}
    """
    if not allow_generation:
        # 이전보다 사용자 친화적인 문구
        return {
            "answer": "⚠️ 현재 '생성 요약'이 꺼져 있어 근거 문장만 제공합니다. 사이드바에서 켜면 문장형 답변을 생성합니다.",
            "sentences": sentences,
            "answerable": False,
        }

    if not sentences:
        return {
            "answer": "업로드한 출처에서 관련 근거를 찾지 못했습니다. 더 적합한 문서를 업로드해 주세요.",
            "sentences": [],
            "answerable": False,
        }

    # '핵심 키워드 포함 여부' 휴리스틱
    required = require_terms if require_terms is not None else LLM_NAME_PATTERNS
    has_core_terms = _contains_answer_terms(sentences, required)
    evidence = _make_evidence_block(sentences)

    if not has_core_terms:
        return {
            "answer": (
                "업로드한 출처에 **질문의 핵심(LLM의 정확한 이름/특징)** 을 직접적으로 확인할 문장이 없습니다. "
                "아래 근거만으로는 확정 답을 내기 어렵습니다. 필요한 경우 "
                "관련 보도자료/공식 문서를 추가 업로드해 주세요.\n\n"
                "— 근거 요약은 아래를 참고하세요."
            ),
            "sentences": sentences,
            "answerable": False,
        }

    answer = _safe_llm_invoke(
        llm,
        ANSWER_PROMPT,
        {"question": question, "evidence": evidence}
    )
    return {"answer": answer, "sentences": sentences, "answerable": True}


# =========================
# Sentence Ranking Utility
# =========================

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!…]|[。？！]|[\n])\s+")

def _tokenize(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip().lower())
    # 숫자/영문/한글 토큰 유지, 기타 구두점 제거
    text = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+", " ", text)
    return [t for t in text.split() if t]


def _calc_idf(all_sentences: List[List[str]]) -> Dict[str, float]:
    df = defaultdict(int)
    N = len(all_sentences) or 1
    for s in all_sentences:
        for w in set(s):
            df[w] += 1
    return {w: math.log((N + 1) / (df[w] + 0.5)) for w in df}


def _score_sentence(query_tokens: List[str], sent_tokens: List[str], idf: Dict[str, float]) -> float:
    if not sent_tokens:
        return 0.0
    tf = Counter(sent_tokens)
    score = 0.0
    for q in set(query_tokens):
        score += (tf.get(q, 0) * idf.get(q, 0.0))
    # 길이 패널티(너무 긴 문장은 감점 소폭)
    score /= (1.0 + math.log(1 + max(0, len(sent_tokens) - 25)) if len(sent_tokens) > 25 else 1.0)
    return score


def extract_relevant_sentences(
    docs: List[Document],
    question: str,
    k: int = 8,
    dedupe: bool = True
) -> List[Dict[str, Any]]:
    """
    업로드된 Document들에서 질문과 가장 관련 있는 '최소' 문장들을 추출.
    반환 포맷: [{"text": str, "source": str, "page": int|None, "timecode": str|None}, ...]
    """
    question_tokens = _tokenize(question)

    candidate: List[Tuple[float, Dict[str, Any]]] = []
    tokenized_sents: List[List[str]] = []

    # 후보 문장 수집
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("url") or ""
        page = doc.metadata.get("page")
        timecode = doc.metadata.get("timecode")

        # 문장 분할
        for raw_sent in _SENT_SPLIT.split(doc.page_content or ""):
            sent = raw_sent.strip()
            if not sent:
                continue
            tokens = _tokenize(sent)
            tokenized_sents.append(tokens)
            candidate.append((0.0, {"text": sent, "source": src, "page": page, "timecode": timecode, "_tokens": tokens}))

    # 점수화
    idf = _calc_idf([c[1]["_tokens"] for c in candidate]) if candidate else {}
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for _, item in candidate:
        score = _score_sentence(question_tokens, item["_tokens"], idf)
        if score <= 0:
            continue
        cleaned = dict(item)
        cleaned.pop("_tokens", None)
        scored.append((score, cleaned))

    # 정렬 및 상위 k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [it for _, it in scored[: max(k * 2, k)]]  # 여유 확보 후 dedupe

    # dedupe: 동일/유사 문장 제거(간단 문자열 키)
    if dedupe:
        seen = set()
        deduped = []
        for it in top:
            key = re.sub(r"\s+", " ", it["text"]).strip()
            if key not in seen:
                seen.add(key)
                deduped.append(it)
        top = deduped

    return top[:k]


# =========================
# (옵션) 기본 체인 Stubs
# =========================
def get_conversational_rag_chain(llm: BaseChatModel):
    """레포 호환용 더미. 프로젝트에서 별도 사용 시 기존 구현을 유지하세요."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "간결하게 답하세요."),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()


def get_default_chain(llm: BaseChatModel):
    """레포 호환용 더미."""
    return get_conversational_rag_chain(llm)

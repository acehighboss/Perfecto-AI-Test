from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# LLM 팩토리
def _default_llm(model: Optional[str] = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    모델/온도 기본값을 한 곳에서 관리.
    환경변수로 OPENAI_API_KEY가 설정되어 있어야 합니다.
    """
    model = model or "gpt-4o-mini"  # 필요 시 프로젝트 표준 모델로 교체
    return ChatOpenAI(model=model, temperature=temperature)


# 유틸
def _format_docs(docs: List[Document]) -> str:
    """
    RAG 컨텍스트용 텍스트 포맷.
    문서가 PDF/YouTube 등일 때 page/timecode 같은 메타데이터가
    이미 채워져있다면 아래처럼 간단히 붙여도 충분합니다.
    """
    lines = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "")
        page = (d.metadata or {}).get("page", None)
        # page/timecode 등 추가 메타가 있으면 보기 좋게 포함
        tag = f"{src}" + (f" (p.{page})" if page is not None else "")
        content = (d.page_content or "").strip()
        if content:
            lines.append(f"[{i}] Source: {tag}\n{content}")
    return "\n\n".join(lines)


def _collect_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    UI의 '소스 미리보기'에 바로 먹일 수 있도록 간단한 메타 형태로 반환.
    """
    out = []
    for d in docs:
        meta = d.metadata or {}
        out.append({
            "source": meta.get("source"),
            "page": meta.get("page"),
            "title": meta.get("title"),
            "type": meta.get("type"),   # "pdf", "web", "youtube" 등 체계화했다면 표시
            "snippet": (d.page_content or "")[:240],
        })
    return out


# 체인 정의
SYSTEM_RAG = """\
You are a careful research assistant. Use ONLY the provided context to answer.
If the context is thin or incomplete, say so and keep the answer conservative.
Cite specific details from the context where possible (but don't invent citations).
Keep answers concise, structured, and neutral.
"""

PROMPT_RAG = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_RAG),
        MessagesPlaceholder(variable_name="history"),   # 대화형이면 사용
        ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer in Korean."),
    ]
)

SYSTEM_DEFAULT = """\
You are a helpful assistant. If you lack sources, answer with general knowledge,
but clearly state that you did not find enough evidence in the provided documents.
Keep the answer concise and practical. Answer in Korean.
"""

PROMPT_DEFAULT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_DEFAULT),
        ("human", "{question}"),
    ]
)


# 호출 형태를 `.invoke({"question": ...})`로 맞추기 위한 간단한 래퍼
class _SimpleInvoke:
    def __init__(self, fn):
        self._fn = fn
    def invoke(self, inputs: Dict[str, Any]) -> Any:
        return self._fn(inputs)


def get_default_chain(llm: Optional[ChatOpenAI] = None) -> _SimpleInvoke:
    """
    문서 컨텍스트 없이 일반 지식으로 답하는 기본 체인.
    """
    llm = llm or _default_llm()
    def _run(inputs: Dict[str, Any]) -> str:
        q = inputs.get("question") or inputs.get("query") or ""
        msgs = PROMPT_DEFAULT.format_messages(question=q)
        resp = llm.invoke(msgs)
        return resp.content
    return _SimpleInvoke(_run)


def get_conversational_rag_chain(
    retriever,
    llm: Optional[ChatOpenAI] = None,
    return_sources: bool = True,
) -> _SimpleInvoke:
    """
    기존 코드 호환: `.invoke({"question": ...})` 로 호출.
    - retriever는 search_type='similarity' 기반 SmartRetriever 래핑 인스턴스여야 함
    - return_sources=True일 때, {"answer": str, "sources": list} 를 반환
      (False면 answer str 만 반환)
    """
    llm = llm or _default_llm()

    def _run(inputs: Dict[str, Any]):
        question = inputs.get("question") or inputs.get("query") or ""
        history = inputs.get("history", [])  # [HumanMessage, AIMessage, ...] 형태면 그대로 사용

        # 1) 문서 검색
        docs: List[Document] = retriever.get_relevant_documents(question)

        # 2) 문서가 없거나 빈 내용뿐이면 바로 폴백 (과도한 "사과" 방지)
        nonempty_docs = [d for d in docs if (d and (d.page_content or "").strip())]
        if not nonempty_docs:
            default_chain = get_default_chain(llm=llm)
            ans = default_chain.invoke({"question": question})
            ans = "⚠️ 업로드 문서에서 직접적인 근거를 충분히 찾지 못했습니다.\n\n" + (ans or "")
            return {"answer": ans, "sources": []} if return_sources else ans

        # 3) 컨텍스트 생성
        context_text = _format_docs(nonempty_docs)

        # 4) LLM 호출
        msgs = PROMPT_RAG.format_messages(
            history=history,
            question=question,
            context=context_text,
        )
        resp = llm.invoke(msgs)
        answer_text = resp.content

        if return_sources:
            return {
                "answer": answer_text,
                "sources": _collect_sources(nonempty_docs),
            }
        else:
            return answer_text

    return _SimpleInvoke(_run)


# ----------------------------
# 헬퍼: 고수준 폴백 일괄 처리 진입점 (선택 사용)
# ----------------------------
def answer_with_fallback(
    retriever,
    question: str,
    history: Optional[List[Any]] = None,
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    간단 진입점:
    - RAG 컨텍스트가 충분하면 RAG로,
    - 부족하면 default로 자동 폴백.
    """
    rag_chain = get_conversational_rag_chain(retriever, llm=llm, return_sources=True)
    return rag_chain.invoke({"question": question, "history": history or []})

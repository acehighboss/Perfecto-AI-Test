import os
import re
import traceback
import streamlit as st
from typing import List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ---- 내부 모듈 ----
from file_handler import (
    get_documents_from_urls_robust,
    get_documents_from_uploaded_files,
)
from RAG.rag_pipeline import get_retriever_from_source
from RAG.chain_builder import get_conversational_rag_chain


# -----------------------------
# 유틸: 간단한 문장 분할 & 서포트 문장 추출
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?。!?])\s+|[\n]+")

def _split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
    return sents[:2000]  # 과도한 길이 방지

def _score_sentence(sent: str, query: str, answer: str) -> float:
    # 아주 가벼운 키워드 기반 점수(질문+답변 키워드와의 교집합 크기)
    def toks(x: str) -> set:
        return set(re.findall(r"[A-Za-z0-9가-힣]+", (x or "").lower()))
    q = toks(query)
    a = toks(answer)
    s = toks(sent)
    inter = len((q | a) & s)
    # 길이 보정(너무 긴 문장은 감점)
    penalty = max(1.0, len(sent) / 300.0)
    return inter / penalty

def extract_support_sentences(doc_text: str, query: str, answer: str, topk: int = 3) -> List[str]:
    sents = _split_sentences(doc_text)
    scored = sorted(sents, key=lambda s: _score_sentence(s, query, answer), reverse=True)
    uniq, res = set(), []
    for s in scored:
        key = s[:120]
        if key not in uniq and len(s) > 10:
            uniq.add(key)
            res.append(s)
        if len(res) >= topk:
            break
    return res

# ------------------------------------
# Streamlit 기본 설정
# ------------------------------------
st.set_page_config(page_title="Perfecto AI Test (RAG)", page_icon="🧪", layout="wide")
st.title("Perfecto AI Test (RAG)")

# ------------------------------------
# Session State 초기화
# ------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Any] = []
if "docs" not in st.session_state:
    st.session_state.docs: List[Document] = []
if "ready_to_analyze" not in st.session_state:
    st.session_state.ready_to_analyze = False
if "docs_for_citation" not in st.session_state:
    st.session_state.docs_for_citation: List[Document] = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# ------------------------------------
# 레이아웃: 좌측 사이드바(시스템 프롬프트/소스 불러오기)
# ------------------------------------
with st.sidebar:
    st.subheader("시스템 프롬프트(페르소나)")
    system_prompt = st.text_area(
        "모델의 역할/톤/스타일을 정의하세요",
        value=(
            "너는 신뢰할 수 있는 분석가다. 명확하고 간결하게 요약하고, "
            "사실 근거를 밝히며, 불확실하면 추정하지 않는다."
        ),
        height=140,
    )

    st.markdown("---")
    st.subheader("데이터 불러오기")

    url_input = st.text_area(
        "URL (줄바꿈으로 여러 개 입력)",
        value="https://www.mobiinside.co.kr/2025/06/27/ai-news-3/\nhttps://namu.wiki/w/%EC%84%B1%EA%B2%BD",
        height=100,
    )

    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF/DOCX/TXT/MD/CSV/JSON/LOG)",
        type=["pdf", "docx", "txt", "md", "csv", "json", "log"],
        accept_multiple_files=True,
    )

    respect_robots = st.toggle("robots.txt 준수", value=True)
    use_js_render = st.toggle("JS 렌더링(Playwright) 사용", value=False,
                              help="CSR 사이트 대응(느림). 환경에 따라 미동작 가능")
    js_only_when_needed = st.toggle("정적 추출 실패 시에만 JS 사용", value=True)

    if st.button("불러오기", use_container_width=True):
        try:
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            url_docs = get_documents_from_urls_robust(
                urls,
                respect_robots=respect_robots,
                use_js_render=use_js_render,
                js_only_when_needed=js_only_when_needed,
            ) if urls else []

            file_docs = get_documents_from_uploaded_files(uploaded_files) if uploaded_files else []

            docs = (url_docs or []) + (file_docs or [])
            st.session_state.docs = docs
            st.session_state.docs_for_citation = docs
            st.session_state.ready_to_analyze = len(docs) > 0

            if not docs:
                st.warning("불러온 문서가 없습니다. URL 또는 파일을 확인해 주세요.")
            else:
                st.success(f"{len(docs)}개 문서를 불러왔습니다.")
        except Exception as e:
            st.error(f"문서 불러오는 중 오류: {e}")
            st.caption(traceback.format_exc())
            st.session_state.docs = []
            st.session_state.ready_to_analyze = False

# ------------------------------------
# 메인 영역: 상단 Q/A, 하단 입력창
# ------------------------------------
# 상단(왼쪽: 대화/Q&A, 오른쪽: 소스 미리보기)
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("질문 & 답변")

    # 기존 기록 렌더(질문은 오른쪽 영역에, 사이드바가 아닌 본문에 표시)
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            st.chat_message("user").write(m.content)
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").write(m.content)

    # 최신 분석 결과의 출처 JSON(요청된 형태) 렌더
    if st.session_state.last_answer and st.session_state.docs_for_citation:
        question_for_json = None
        # 최근 사용자 메시지(질문) 추출
        for msg in reversed(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                question_for_json = msg.content
                break
        citation_json: Dict[str, Any] = {
            "question": question_for_json or "",
            "answer": st.session_state.last_answer,
            "sources": []
        }
        query = question_for_json or ""
        answer = st.session_state.last_answer

        # source_documents가 체인에서 주어졌다면 그걸 우선 사용하고,
        # 없으면 docs_for_citation(=현재 코퍼스)에서 상위 n개만 샘플로 표시
        preview_docs = st.session_state.docs_for_citation
        max_sources = min(6, len(preview_docs))
        for d in preview_docs[:max_sources]:
            meta = d.metadata or {}
            title = meta.get("title") or meta.get("filename") or "unknown"
            src = meta.get("source") or ""
            support = extract_support_sentences(d.page_content or "", query, answer, topk=3)
            citation_json["sources"].append({
                "title": title,
                "source": src,
                "support": support,
            })

        st.markdown("**출처 (JSON)**")
        st.json(citation_json)

with col_right:
    st.subheader("소스 미리보기")
    preview_docs = st.session_state.docs_for_citation or st.session_state.docs
    if preview_docs:
        for i, d in enumerate(preview_docs[:8], 1):
            src = (d.metadata or {}).get("source", "")
            title = (d.metadata or {}).get("title", src or f"문서 {i}")
            with st.expander(f"[{i}] {title}"):
                if src:
                    st.caption(src)
                body = d.page_content or ""
                st.write(body[:1200] + ("..." if len(body) > 1200 else ""))
    else:
        st.caption("불러온 문서가 없습니다.")

# -----------------------------
# 우측 하단 입력창(사이드바 X)
# -----------------------------
user_query = st.chat_input("여기에 질문을 입력하세요")  # 화면 우측 하단에 고정

if user_query and st.session_state.ready_to_analyze and st.session_state.docs:
    # 질문 메시지 본문 영역에 표시
    st.session_state.messages.append(HumanMessage(content=user_query))

    try:
        # 시스템 프롬프트를 질문 앞에 주입(체인이 system 역할을 직접 받지 않는 경우를 대비)
        composed_query = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_query}"

        retriever = get_retriever_from_source(documents=st.session_state.docs)
        chain = get_conversational_rag_chain(retriever=retriever)

        with st.spinner("분석 중..."):
            result = chain.invoke({"question": composed_query})

        answer_text = None
        source_docs: List[Document] = []

        if isinstance(result, dict):
            answer_text = result.get("answer") or result.get("output") or result.get("text")
            sd = result.get("source_documents") or result.get("sources") or []
            if isinstance(sd, list):
                for d in sd:
                    if isinstance(d, Document):
                        source_docs.append(d)
                    elif isinstance(d, dict) and "page_content" in d:
                        md = d.get("metadata", {}) or {}
                        source_docs.append(Document(page_content=d["page_content"], metadata=md))
        elif isinstance(result, str):
            answer_text = result

        if not answer_text:
            answer_text = "분석 결과가 비어있습니다. 프롬프트/문서 상태를 확인해 주세요."

        # 답변을 본문 영역에 표시
        st.chat_message("assistant").write(answer_text)
        st.session_state.messages.append(AIMessage(content=answer_text))
        st.session_state.last_answer = answer_text

        # 출처 표시용 상태 업데이트
        if source_docs:
            st.session_state.docs_for_citation = source_docs
        else:
            # 체인이 출처를 반환하지 않은 경우엔 현재 코퍼스로 대체
            st.session_state.docs_for_citation = st.session_state.docs

    except Exception as e:
        st.error(f"분석 중 오류: {e}")
        st.caption(traceback.format_exc())

elif user_query and not st.session_state.ready_to_analyze:
    st.warning("먼저 좌측에서 URL/파일을 불러오세요.")


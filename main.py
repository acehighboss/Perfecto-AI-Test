import traceback
import streamlit as st
from typing import List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from file_handler import (
    get_documents_from_urls_robust,
    get_documents_from_uploaded_files,
)
from RAG.rag_pipeline import (
    get_retriever_from_source,      # retriever 빌드만 위임
)
from RAG.chain_builder import (
    get_conversational_rag_chain,   # 체인 빌드만 위임
)

# Streamlit 기본 설정
st.set_page_config(page_title="Perfecto AI Test (RAG)", page_icon="🧪", layout="wide")
st.title("Perfecto AI Test (RAG)")

# Session State (UI 상태만)
if "messages" not in st.session_state:
    st.session_state.messages: List[Any] = []
if "docs" not in st.session_state:
    st.session_state.docs: List[Document] = []
if "ready" not in st.session_state:
    st.session_state.ready = False
if "docs_for_citation" not in st.session_state:
    st.session_state.docs_for_citation: List[Document] = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# ------------------------------------
# 사이드바 (시스템 프롬프트 + 데이터 불러오기)
# ------------------------------------
with st.sidebar:
    st.subheader("시스템 프롬프트(페르소나)")
    system_prompt = st.text_area(
        "모델의 역할/톤/스타일",
        value=(
            "당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다. "
            "항상 친절하고, 정확한 정보를 한국어로 상세하게 전달해주세요. "
            "컨텍스트에 없는 내용은 답변할 수 없다고 솔직하게 말해주세요."
        ),
        height=150,
        help="필요하면 수정 가능합니다.",
    )

    st.markdown("---")
    st.subheader("데이터 불러오기")

    url_input = st.text_area(
        "URL (줄바꿈으로 여러 개 입력)",
        value="",
        height=240,   # ★ 높이 확장
        help="예) 각 줄에 1개 URL 입력",
    )
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF/DOCX/TXT/MD/CSV/JSON/LOG)",
        type=["pdf", "docx", "txt", "md", "csv", "json", "log"],
        accept_multiple_files=True,
    )

    # 크롤링 관련 토글 (동작은 백엔드로 위임)
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
            st.session_state.ready = len(docs) > 0

            if not docs:
                st.warning("불러온 문서가 없습니다. URL 또는 파일을 확인해 주세요.")
            else:
                st.success(f"{len(docs)}개 문서를 불러왔습니다.")
        except Exception as e:
            st.error(f"문서 불러오는 중 오류: {e}")
            st.caption(traceback.format_exc())
            st.session_state.docs = []
            st.session_state.docs_for_citation = []
            st.session_state.ready = False

# 메인: Q/A 및 출처 JSON
col_main, col_sources = st.columns([3, 2])

with col_main:
    st.subheader("질문 & 답변")

    # 이전 대화 렌더(우측 영역에 표시)
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            st.chat_message("user").write(m.content)
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").write(m.content)

    # 최신 결과의 출처 JSON (UI 렌더만)
    if st.session_state.last_answer and st.session_state.docs_for_citation:
        st.markdown("**출처 (JSON)**")
        # 체인이 source_documents를 반환하는 경우 메타데이터만 표시
        # (근거 문장/스팬 등은 백엔드에서 메타로 넣어주면 그대로 노출)
        citation_obj: Dict[str, Any] = {
            "question": st.session_state.last_question,
            "answer": st.session_state.last_answer,
            "sources": [],
        }
        for d in st.session_state.docs_for_citation[:10]:
            meta = d.metadata or {}
            citation_obj["sources"].append({
                "title": meta.get("title") or meta.get("filename") or "unknown",
                "source": meta.get("source") or "",
                # 백엔드가 문장/스팬을 넣어줬다면 그대로 표출 (없으면 생략/빈 리스트)
                "support": meta.get("support") or meta.get("spans") or [],
            })
        st.json(citation_obj)

with col_sources:
    st.subheader("소스 미리보기")
    preview_docs = st.session_state.docs_for_citation or st.session_state.docs
    if preview_docs:
        for i, d in enumerate(preview_docs[:8], 1):
            meta = d.metadata or {}
            src = meta.get("source", "")
            title = meta.get("title", src or f"문서 {i}")
            with st.expander(f"[{i}] {title}"):
                if src:
                    st.caption(src)
                body = d.page_content or ""
                st.write(body[:1200] + ("..." if len(body) > 1200 else ""))
    else:
        st.caption("불러온 문서가 없습니다.")

# 우측 하단 입력창
user_query = st.chat_input("여기에 질문을 입력하세요")

if user_query:
    if not st.session_state.ready or not st.session_state.docs:
        st.warning("먼저 좌측에서 URL/파일을 불러오세요.")
    else:
        # Q 표시
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.last_question = user_query

        try:
            # 백엔드에 시스템 프롬프트는 별도 필드로 넘길 수 있으면 가장 좋음.
            # 체인 시그니처가 'system'을 받지 않는다면, 백엔드에서 반영되도록 구성해주세요.
            retriever = get_retriever_from_source(documents=st.session_state.docs)
            chain = get_conversational_rag_chain(retriever=retriever)

            with st.spinner("분석 중..."):
                # 체인의 입력 규약에 맞춰 전달 (예: {"question": "...", "system": "..."})
                # 체인이 system을 지원하지 않는다면, 내부에서 반영되도록 백엔드에서 처리 권장
                result = chain.invoke({"question": user_query, "system": system_prompt})

            # 최소 파싱: answer와 source_documents만 사용 (나머지는 백엔드 책임)
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
                answer_text = "분석 결과가 비어있습니다."

            # A 표시
            st.chat_message("assistant").write(answer_text)
            st.session_state.messages.append(AIMessage(content=answer_text))
            st.session_state.last_answer = answer_text

            # 출처 JSON 표시에 사용할 우선순위: 체인에서 준 source_documents
            st.session_state.docs_for_citation = source_docs or st.session_state.docs

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            st.caption(traceback.format_exc())

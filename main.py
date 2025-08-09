import os
import traceback
import streamlit as st
from typing import List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ---- 내부 모듈 ----
from file_handler import (
    get_documents_from_urls_robust,
    get_documents_from_uploaded_files
)
from RAG.rag_pipeline import (
    get_retriever_from_source,
    get_conversational_rag_chain,
)

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

# ------------------------------------
# Sidebar (URL + 파일 업로드 + 옵션)
# ------------------------------------
with st.sidebar:
    st.subheader("URL 업로드")
    url_input = st.text_area(
        "하나 이상의 URL을 줄바꿈으로 입력하세요",
        value="https://www.mobiinside.co.kr/2025/06/27/ai-news-3/\nhttps://namu.wiki/w/%EC%84%B1%EA%B2%BD",
        height=120,
        help="예: 각 줄에 1개 URL",
    )

    st.subheader("파일 업로드")
    uploaded_files = st.file_uploader(
        "PDF / DOCX / TXT / MD / CSV 지원",
        type=["pdf", "docx", "txt", "md", "csv", "json", "log"],
        accept_multiple_files=True,
        help="여러 파일을 한 번에 업로드할 수 있습니다."
    )

    # 크롤링 옵션
    respect_robots = st.toggle(
        "robots.txt 준수", value=True,
        help="해제 시 차단 경로도 시도하지만, 실제로는 서버에서 거부될 수 있습니다."
    )
    use_js_render = st.toggle(
        "JS 렌더링(Playwright) 사용", value=False,
        help="CSR 페이지 대응. 느리고, 호스팅 환경에 따라 동작하지 않을 수 있습니다."
    )
    js_only_when_needed = st.toggle(
        "정적 추출 실패/부족 시에만 JS 사용", value=True
    )

    # 문서 불러오기 버튼
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
            st.session_state.ready_to_analyze = len(docs) > 0

            if not docs:
                st.warning("불러온 문서가 없습니다. URL 또는 파일을 확인해 주세요.")
            else:
                st.success(f"{len(docs)}개 문서를 불러왔습니다.")

        except Exception as e:
            st.error(f"문서를 불러오는 중 오류가 발생했습니다: {e}")
            st.caption(traceback.format_exc())
            st.session_state.docs = []
            st.session_state.ready_to_analyze = False

    st.divider()

    # 분석 프롬프트
    user_query = st.text_input(
        "질문/분석 요청",
        value="요약과 핵심 포인트를 알려줘. 출처도 표시해줘."
    )

    analyze_clicked = st.button(
        "분석 시작",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.ready_to_analyze
    )

# ------------------------------------
# 본문 레이아웃
# ------------------------------------
col_chat, col_right = st.columns([3, 2])

# 좌측: 대화/분석
with col_chat:
    st.subheader("대화")
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            st.chat_message("user").write(m.content)
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").write(m.content)
        else:
            role = "assistant" if isinstance(m, dict) and m.get("role") == "assistant" else "user"
            st.chat_message(role).write(str(m))

    if analyze_clicked and st.session_state.ready_to_analyze and st.session_state.docs:
        st.session_state.messages.append(HumanMessage(content=user_query))
        try:
            retriever = get_retriever_from_source(
                documents=st.session_state.docs
            )
            chain = get_conversational_rag_chain(retriever=retriever)

            with st.spinner("분석 중..."):
                result = chain.invoke({"question": user_query})

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

            st.chat_message("assistant").write(answer_text)
            st.session_state.messages.append(AIMessage(content=answer_text))

            if source_docs:
                st.session_state.docs_for_citation = source_docs
            else:
                st.session_state.docs_for_citation = st.session_state.docs

            st.session_state.ready_to_analyze = False

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            st.caption(traceback.format_exc())
            st.session_state.ready_to_analyze = False

# 우측: 소스 미리보기 & 참고/출처
with col_right:
    st.subheader("소스 미리보기")
    preview_docs = st.session_state.get("docs_for_citation") or st.session_state.docs
    if preview_docs:
        for i, d in enumerate(preview_docs, 1):
            src = (d.metadata or {}).get("source", "")
            title = (d.metadata or {}).get("title", src or f"문서 {i}")
            with st.expander(f"[{i}] {title}"):
                if src:
                    st.caption(src)
                body = d.page_content or ""
                st.write(body[:1200] + ("..." if len(body) > 1200 else ""))

    st.subheader("참고/출처")
    if preview_docs:
        seen = set()
        for d in preview_docs:
            src = (d.metadata or {}).get("source") or ""
            title = (d.metadata or {}).get("title") or src
            key = (title, src)
            if src and key not in seen:
                seen.add(key)
                st.markdown(f"- [{title}]({src})")
    else:
        st.caption("불러온 문서가 없습니다.")

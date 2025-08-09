# main.py

import os
import traceback
import streamlit as st
from typing import List, Any

# LangChain types
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# ---- Repo 내부 모듈 (존재 가정) ----
# robust URL 수집기 (JS 렌더링 토글 포함) - file_handler.py에 구현되어 있어야 합니다.
from file_handler import get_documents_from_urls_robust

# RAG 파이프라인 (기존 레포의 함수 시그니처를 사용)
# - get_retriever_from_source(documents=[...], ...) -> retriever
# - get_conversational_rag_chain(retriever=retriever) -> chain
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

# ------------------------------------
# Sidebar (URL 업로드 + 옵션)
# ------------------------------------
with st.sidebar:
    st.subheader("URL 업로드")
    url_input = st.text_area(
        "하나 이상의 URL을 줄바꿈으로 입력하세요",
        value="https://www.mobiinside.co.kr/2025/06/27/ai-news-3/\nhttps://namu.wiki/w/%EC%84%B1%EA%B2%BD",
        height=120,
        help="예: 각 줄에 1개 URL",
    )

    # 크롤링 동작 토글
    respect_robots = st.toggle(
        "robots.txt 준수", value=True,
        help="해제 시 차단 경로도 시도하지만, 실제로는 서버 단에서 거부될 수 있습니다."
    )
    use_js_render = st.toggle(
        "JS 렌더링(Playwright) 사용", value=False,
        help="React/Vue 등 CSR 페이지 대응. 느리고, 호스팅 환경에 따라 동작하지 않을 수 있습니다."
    )
    js_only_when_needed = st.toggle(
        "정적 추출 실패/부족 시에만 JS 사용", value=True,
        help="정적 파싱으로 충분하면 JS 렌더링을 생략하여 속도를 높입니다."
    )

    # 불러오기 버튼
    if st.button("URL 불러오기", use_container_width=True):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        try:
            docs = get_documents_from_urls_robust(
                urls,
                respect_robots=respect_robots,
                use_js_render=use_js_render,
                js_only_when_needed=js_only_when_needed,
            )
            st.session_state.docs = docs
            st.session_state.ready_to_analyze = len(docs) > 0

            if not docs:
                st.warning(
                    "본문을 추출할 수 없었습니다. (robots 차단/로그인 필요/강한 봇차단/JS 미지원 환경/빈 본문 등)"
                )
            else:
                st.success(f"{len(docs)}개 문서를 불러왔습니다.")
        except Exception as e:
            st.error(f"URL 불러오는 중 오류가 발생했습니다: {e}")
            st.caption(traceback.format_exc())
            st.session_state.docs = []
            st.session_state.ready_to_analyze = False

    st.divider()

    # 분석 요청 프롬프트
    user_query = st.text_input(
        "질문/분석 요청",
        value="두 문서의 핵심 요약, 공통점/차이, 주의할 점을 알려줘. 출처도 표시해줘."
    )

    # 분석 시작 버튼
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

# ------------------------------------
# 좌측: 대화/분석
# ------------------------------------
with col_chat:
    st.subheader("대화")

    # 기존 메시지 렌더
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            st.chat_message("user").write(m.content)
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").write(m.content)
        else:
            # 안전장치(문자열 등)
            role = "assistant" if isinstance(m, dict) and m.get("role") == "assistant" else "user"
            st.chat_message(role).write(str(m))

    # 분석 시작 로직
    if analyze_clicked and st.session_state.ready_to_analyze and st.session_state.docs:
        st.session_state.messages.append(HumanMessage(content=user_query))
        try:
            # 1) 문서 → retriever
            retriever = get_retriever_from_source(
                documents=st.session_state.docs,  # <- 반드시 documents 인자를 받도록 rag_pipeline 수정되어 있어야 함
                # 필요 시 chunk_size, chunk_overlap 등의 파라미터도 전달 가능
            )

            # 2) retriever → 대화형 RAG 체인
            chain = get_conversational_rag_chain(retriever=retriever)

            # 3) 호출
            with st.spinner("분석 중..."):
                result = chain.invoke({"question": user_query})

            # 4) 응답/출처 처리 (체인 구현별로 유연하게 수용)
            answer_text = None
            source_docs: List[Document] = []

            # (A) result가 dict 형태로 answer / source_documents를 제공하는 경우
            if isinstance(result, dict):
                answer_text = result.get("answer") or result.get("output") or result.get("text")
                sd = result.get("source_documents") or result.get("sources") or []
                if isinstance(sd, list):
                    # LangChain Document 타입 또는 유사 dict를 허용
                    for d in sd:
                        if isinstance(d, Document):
                            source_docs.append(d)
                        elif isinstance(d, dict) and "page_content" in d:
                            # dict -> Document 변환
                            md = d.get("metadata", {}) or {}
                            source_docs.append(Document(page_content=d["page_content"], metadata=md))
            # (B) 단순 문자열로 오는 경우
            if answer_text is None and isinstance(result, str):
                answer_text = result

            # 5) 답변 렌더
            if not answer_text:
                answer_text = "분석 결과가 비어있습니다. 프롬프트/문서 상태를 확인해 주세요."

            st.chat_message("assistant").write(answer_text)
            st.session_state.messages.append(AIMessage(content=answer_text))

            # 6) 우측 컬럼의 '참고/출처' 섹션이 source_documents를 렌더할 수 있도록 상태 저장
            #    (여기선 docs 전체/혹은 source_documents 우선)
            if source_docs:
                # source_documents가 있으면 그것 우선 미리보기로 표시할 수 있도록 교체
                st.session_state.docs_for_citation = source_docs
            else:
                st.session_state.docs_for_citation = st.session_state.docs

            # 다음 클릭 전까지 중복 실행 방지
            st.session_state.ready_to_analyze = False

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            st.caption(traceback.format_exc())
            # 실패 시 ready 플래그는 유지하여 재시도 가능하도록 둘 수도 있음
            # 여기서는 안전하게 False로 초기화
            st.session_state.ready_to_analyze = False

# ------------------------------------
# 우측: 소스 미리보기 & 참고/출처
# ------------------------------------
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
                st.write(body[:1200] + ("..." if len(body) > 1200 else ""))  # 길이 제한

    st.subheader("참고/출처")
    if preview_docs:
        # 중복 URL 제거하여 나열
        seen = set()
        for d in preview_docs:
            src = (d.metadata or {}).get("source") or ""
            title = (d.metadata or {}).get("title") or src
            key = (title, src)
            if src and key not in seen:
                seen.add(key)
                st.markdown(f"- [{title}]({src})")
    else:
        st.caption("불러온 문서가 없습니다. 사이드바에서 URL을 입력하고 ‘URL 불러오기’를 눌러주세요.")

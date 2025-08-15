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
        height=240,
        help="예) 각 줄에 1개 URL 입력",
    )
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF/DOCX/TXT/MD/CSV/JSON/LOG)",
        type=["pdf", "docx", "txt", "md", "csv", "json", "log"],
        accept_multiple_files=True,
    )

    use_js_render = st.toggle("JS 렌더링(Playwright) 사용", value=True,
                              help="CSR 사이트 대응. 초기 HTML에 내용이 거의 없으면 자동으로 JS 렌더링을 시도합니다.")

    if st.button("불러오기", use_container_width=True):
        with st.spinner("문서를 불러오는 중입니다..."):
            try:
                urls = [u.strip() for u in url_input.splitlines() if u.strip()]
                url_docs = get_documents_from_urls_robust(
                    urls,
                    use_js_render=use_js_render,
                ) if urls else []

                file_docs = get_documents_from_uploaded_files(uploaded_files) if uploaded_files else []

                docs = (url_docs or []) + (file_docs or [])
                st.session_state.docs = docs
                # 초기 소스 미리보기는 전체 문서를 보여줌
                st.session_state.docs_for_citation = []
                st.session_state.ready = len(docs) > 0

                if not docs:
                    st.warning("불러온 문서가 없습니다. URL 또는 파일을 확인해 주세요.")
                else:
                    st.success(f"{len(docs)}개 문서를 불러왔습니다.")
                    st.rerun() # 소스 미리보기 업데이트
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

    # 이전 대화 렌더
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.write(m.content)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant"):
                st.write(m.content)

    # 최신 결과의 출처 JSON (UI 렌더만)
    # docs_for_citation에 내용이 있을 때만 (즉, 답변이 생성된 후에만) 표시
    if st.session_state.last_answer and st.session_state.docs_for_citation:
        st.markdown("**출처 (JSON)**")
        citation_obj: Dict[str, Any] = {
            "question": st.session_state.last_question,
            "answer": st.session_state.last_answer,
            "sources": [],
        }
        # 답변 생성에 사용된 정확한 문장들만 출처로 표시
        for d in st.session_state.docs_for_citation:
            meta = d.metadata or {}
            citation_obj["sources"].append({
                "source": meta.get("source") or "N/A",
                "title": meta.get("title") or meta.get("filename") or "Unknown",
                "snippet": d.page_content, # 페이지 내용이 곧 문장
            })
        st.json(citation_obj)


with col_sources:
    st.subheader("소스 미리보기")
    # 답변 생성 전에는 전체 문서, 생성 후에는 관련 문서(문장) 표시
    preview_docs = st.session_state.docs_for_citation or st.session_state.docs
    if preview_docs:
        for i, d in enumerate(preview_docs[:8], 1):
            meta = d.metadata or {}
            src = meta.get("source", "N/A")
            title = meta.get("title", src or f"문서 {i}")
            with st.expander(f"[{i}] {title}"):
                st.caption(f"Source: {src}")
                body = d.page_content or ""
                # 내용이 문장이면 짧으므로 전체 표시
                st.write(body)
    else:
        st.caption("불러온 문서가 없습니다.")


# 하단 입력창
if user_query := st.chat_input("여기에 질문을 입력하세요"):
    if not st.session_state.ready or not st.session_state.docs:
        st.warning("먼저 좌측에서 URL/파일을 불러오세요.")
    else:
        # UI에 질문 표시
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.last_question = user_query
        
        with st.chat_message("user"):
            st.write(user_query)

        try:
            # Retriever와 Chain 생성
            retriever = get_retriever_from_source(st.session_state.docs)
            chain = get_conversational_rag_chain(retriever, system_prompt)

            with st.spinner("답변을 생성하는 중입니다..."):
                result = chain.invoke(user_query) # 이제 질문만 넘김

            # 결과 파싱 (answer와 source_documents를 포함하는 dict 기대)
            answer_text = result.get("answer", "답변을 생성하지 못했습니다.")
            source_docs = result.get("source_documents", [])

            # UI에 답변 표시
            st.session_state.messages.append(AIMessage(content=answer_text))
            st.session_state.last_answer = answer_text
            st.session_state.docs_for_citation = source_docs

            # 페이지 새로고침하여 출처 및 답변 업데이트
            st.rerun()

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            st.caption(traceback.format_exc())


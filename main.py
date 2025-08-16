import traceback
import streamlit as st
from typing import List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from file_handler import (
    get_documents_from_urls_robust,
    get_documents_from_uploaded_files,
)
# 수정한 함수를 임포트합니다.
from RAG.rag_pipeline import get_retriever_from_documents
from RAG.chain_builder import get_conversational_rag_chain

# Streamlit 기본 설정
st.set_page_config(page_title="Perfecto AI Test (RAG)", page_icon="🧪", layout="wide")
st.title("Perfecto AI Test (RAG)")

# Session State
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

# 사이드바
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
    )

    st.markdown("---")
    st.subheader("데이터 불러오기")

    url_input = st.text_area("URL (줄바꿈으로 여러 개 입력)", height=240)
    uploaded_files = st.file_uploader(
        "파일 업로드",
        type=["pdf", "docx", "txt", "md", "csv", "json", "log"],
        accept_multiple_files=True,
    )
    use_js_render = st.toggle("JS 렌더링(Playwright) 사용", value=True)

    if st.button("불러오기", use_container_width=True):
        with st.spinner("문서를 불러오는 중입니다..."):
            try:
                urls = [u.strip() for u in url_input.splitlines() if u.strip()]
                url_docs = get_documents_from_urls_robust(urls, use_js_render=use_js_render) if urls else []
                file_docs = get_documents_from_uploaded_files(uploaded_files) if uploaded_files else []

                docs = url_docs + file_docs
                st.session_state.docs = docs
                st.session_state.docs_for_citation = []
                st.session_state.ready = bool(docs)

                if not docs:
                    st.warning("불러온 문서가 없습니다.")
                else:
                    st.success(f"{len(docs)}개 문서를 불러왔습니다.")
                    st.rerun()
            except Exception as e:
                st.error(f"문서 불러오는 중 오류: {e}")
                st.caption(traceback.format_exc())
                st.session_state.ready = False

# 메인 화면
col_main, col_sources = st.columns([3, 2])

with col_main:
    st.subheader("질문 & 답변")
    for m in st.session_state.messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(m.content)

    if st.session_state.last_answer and st.session_state.docs_for_citation:
        st.markdown("**출처 (JSON)**")
        citation_obj = {
            "question": st.session_state.last_question,
            "answer": st.session_state.last_answer,
            "sources": [
                {
                    "source": d.metadata.get("source", "N/A"),
                    "title": d.metadata.get("title", "Unknown"),
                    "snippet": d.page_content,
                }
                for d in st.session_state.docs_for_citation
            ],
        }
        st.json(citation_obj)

with col_sources:
    st.subheader("소스 미리보기")
    preview_docs = st.session_state.docs_for_citation or st.session_state.docs
    if preview_docs:
        for i, d in enumerate(preview_docs[:8], 1):
            meta = d.metadata or {}
            title = meta.get("title", meta.get("source", f"문서 {i}"))
            with st.expander(f"[{i}] {title}"):
                st.caption(f"Source: {meta.get('source', 'N/A')}")
                st.write(d.page_content)
    else:
        st.caption("불러온 문서가 없습니다.")

# 사용자 입력 처리
if user_query := st.chat_input("여기에 질문을 입력하세요"):
    if not st.session_state.ready:
        st.warning("먼저 좌측에서 URL/파일을 불러오세요.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.last_question = user_query
        
        with st.chat_message("user"):
            st.write(user_query)

        try:
            # 오류가 발생했던 부분을 수정된 함수 호출로 변경합니다.
            retriever = get_retriever_from_documents(st.session_state.docs)
            
            if retriever:
                chain = get_conversational_rag_chain(retriever, system_prompt)
                with st.spinner("답변을 생성하는 중입니다..."):
                    # 체인에 사용자 질문만 전달합니다.
                    result = chain.invoke(user_query)
                
                answer_text = result.get("answer", "답변을 생성하지 못했습니다.")
                source_docs = result.get("source_documents", [])
            else:
                answer_text = "Retriever를 생성하는 데 실패했습니다. 문서를 다시 불러와 주세요."
                source_docs = []

            st.session_state.messages.append(AIMessage(content=answer_text))
            st.session_state.last_answer = answer_text
            st.session_state.docs_for_citation = source_docs
            st.rerun()

        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            st.caption(traceback.format_exc())

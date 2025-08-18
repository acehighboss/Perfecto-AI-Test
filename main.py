import traceback
from collections import defaultdict
import streamlit as st
from typing import List, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from file_handler import (
    get_documents_from_urls_robust,
    get_documents_from_uploaded_files,
)
from RAG.rag_pipeline import (
    get_retriever_from_source,
)
from RAG.chain_builder import (
    get_conversational_rag_chain,
)

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
if "source_documents" not in st.session_state:
    st.session_state.source_documents: List[Document] = []
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

    respect_robots = st.toggle("robots.txt 준수", value=True)
    use_js_render = st.toggle("JS 렌더링(Playwright) 사용", value=True,
                              help="CSR 사이트 대응(느림). 고품질 추출을 위해 활성화를 권장합니다.")
    js_only_when_needed = st.toggle("정적 추출 실패 시에만 JS 사용", value=True)

    if st.button("불러오기", use_container_width=True):
        with st.spinner("문서를 불러오는 중입니다..."):
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
                st.session_state.source_documents = [] # 새 문서 로드 시 이전 출처 초기화
                st.session_state.ready = len(docs) > 0

                if not docs:
                    st.warning("불러온 문서가 없습니다. URL 또는 파일을 확인해 주세요.")
                else:
                    st.success(f"{len(docs)}개 문서를 불러왔습니다.")
            except Exception as e:
                st.error(f"문서 불러오는 중 오류: {e}")
                st.caption(traceback.format_exc())
                st.session_state.docs = []
                st.session_state.source_documents = []
                st.session_state.ready = False

# 메인: Q/A 및 출처 JSON
col_main, col_sources = st.columns([3, 2])

with col_main:
    st.subheader("질문 & 답변")

    # 이전 대화 렌더
    for m in st.session_state.messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(m.content)

    # 최신 결과의 출처 JSON (UI 렌더만)
    if st.session_state.last_answer and st.session_state.source_documents:
        st.markdown("**출처 (JSON)**")
        
        # 소스별로 관련 문장(support)을 그룹화
        grouped_sources = defaultdict(lambda: {"source": "", "support": []})
        for doc in st.session_state.source_documents:
            meta = doc.metadata or {}
            title = meta.get("title") or meta.get("filename") or "unknown"
            source_url = meta.get("source") or ""
            
            # 고유한 키로 그룹화 (제목 + 소스 URL)
            key = (title, source_url)
            grouped_sources[key]["source"] = source_url
            # page_content가 바로 관련 문장이므로 support 리스트에 추가
            if doc.page_content not in grouped_sources[key]["support"]:
                 grouped_sources[key]["support"].append(doc.page_content)

        citation_obj: Dict[str, Any] = {
            "question": st.session_state.last_question,
            "answer": st.session_state.last_answer,
            "sources": [
                {
                    "title": title,
                    "source": data["source"],
                    "support": data["support"],
                }
                for (title, _), data in grouped_sources.items()
            ],
        }
        st.json(citation_obj)

with col_sources:
    st.subheader("소스 미리보기")
    preview_docs = st.session_state.docs
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

# 사용자 입력
user_query = st.chat_input("여기에 질문을 입력하세요")

if user_query:
    if not st.session_state.ready or not st.session_state.docs:
        st.warning("먼저 좌측에서 URL/파일을 불러오세요.")
    else:
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.last_question = user_query
        
        with st.chat_message("user"):
            st.write(user_query)

        try:
            with st.spinner("답변을 생성하는 중입니다..."):
                # 1. Retriever 생성
                retriever = get_retriever_from_source(st.session_state.docs)
                
                if not retriever:
                    st.error("Retriever를 생성할 수 없습니다. 문서를 다시 불러와 주세요.")
                    st.stop()
                
                # 2. RAG 체인 생성
                chain = get_conversational_rag_chain(retriever)

                # 3. 체인 실행
                # 체인의 입력 규약에 맞춰 전달
                result = chain.invoke({
                    "question": user_query, 
                    "system": system_prompt,
                    "chat_history": st.session_state.messages[:-1] # 마지막 질문 제외
                })

            # 결과 파싱
            answer_text = result.get("answer", "답변을 생성하지 못했습니다.")
            source_docs = result.get("source_documents", [])

            # UI 업데이트 및 상태 저장
            st.session_state.last_answer = answer_text
            st.session_state.source_documents = source_docs
            st.session_state.messages.append(AIMessage(content=answer_text))

            with st.chat_message("assistant"):
                st.write(answer_text)
            
            # 페이지를 새로고침하여 출처 JSON을 즉시 표시
            st.rerun()

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            st.caption(traceback.format_exc())

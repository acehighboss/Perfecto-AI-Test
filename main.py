import subprocess
import sys
import time
import json
import streamlit as st
from langchain_core.messages import HumanMessage

# Streamlit Cloud 환경에 맞는 Playwright 브라우저 설치
# 시스템 종속성은 packages.txt로 설치되므로, 여기서는 브라우저만 다운로드합니다.
try:
    subprocess.run(
        # --with-deps 옵션 제거
        [f"{sys.executable}", "-m", "playwright", "install"],
        check=True,
        capture_output=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    # 오류 발생 시 로그를 명확하게 출력
    print("Playwright 브라우저 설치 실패. 에러 로그:")
    print(e.stdout)
    print(e.stderr)
    raise

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import time
import json
from RAG.rag_pipeline import get_retriever_from_source
from RAG.graph_builder import get_rag_graph
from RAG.chain_builder import get_default_chain

# --- 페이지 설정 ---
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="⚙️")
st.title("⚙️ Advanced RAG Chatbot")
st.markdown(
    """
    **추가 검색(Self-Correction)** 기능이 적용된 에이전트형 RAG 챗봇입니다.
    답변에 필요한 정보가 부족하다고 판단되면, 스스로 질문을 바꿔 다시 검색합니다.
    """
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# ▼▼▼ [수정] retriever 대신 graph를 세션 상태에 저장합니다. ▼▼▼
if "graph" not in st.session_state:
    st.session_state.graph = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다. 항상 친절하고, 정확한 정보를 한국어로 상세하게 전달해주세요. 컨텍스트에 없는 내용은 답변할 수 없다고 솔직하게 말해주세요."

# --- 사이드바 UI ---
with st.sidebar:
    st.header("⚙️ 설정")
    with st.form("persona_form"):
        st.subheader("🤖 AI 페르소나 설정")
        system_prompt_input = st.text_area(
            "AI의 역할을 설정해주세요.",
            value=st.session_state.system_prompt,
            height=150
        )
        if st.form_submit_button("페르소나 적용"):
            st.session_state.system_prompt = system_prompt_input
            st.success("페르소나가 적용되었습니다!")

    st.divider()
    
    with st.form("source_form"):
        st.subheader("🔎 분석 대상 설정")
        url_input = st.text_area("웹사이트 URL (한 줄에 하나씩 입력)", placeholder="https://news.google.com\nhttps://blog.google/...")
        
        uploaded_files = st.file_uploader(
            "파일 업로드 (PDF, DOCX 등)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )

        if st.form_submit_button("분석 시작"):
            source_type = "URL" if url_input else "Files" if uploaded_files else None
            source_input = url_input or uploaded_files

            if source_type:
                with st.spinner("문서를 분석하고 RAG 워크플로우를 준비 중입니다..."):
                    # ▼▼▼ [수정] retriever를 만들고, 이를 사용해 그래프를 생성합니다. ▼▼▼
                    retriever = get_retriever_from_source(source_type, source_input)
                    if retriever:
                        st.session_state.graph = get_rag_graph(retriever, st.session_state.system_prompt)
                        st.success("분석이 완료되었습니다! 이제 질문해보세요.")
                    else:
                        st.session_state.graph = None
                        st.error("분석에 실패했습니다. API 키나 URL/파일 상태를 확인해주세요.")
            else:
                st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 메인 채팅 화면 ---
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("자세한 출처 보기 (문장 단위)"):
                for source in message["sources"]:
                    st.markdown(f"**- {source['title']}** ([링크]({source['url']}))")
                    for sentence in source['sentences']:
                        st.caption(f"    - {sentence}")
                    st.divider()


if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            # ▼▼▼ [수정] 그래프 워크플로우를 실행합니다. ▼▼▼
            if st.session_state.graph:
                with st.spinner("답변을 생성하고 있습니다... (필요시 추가 검색 수행)"):
                    start_time = time.time()
                    
                    # 그래프 실행을 위한 입력값 설정
                    inputs = {"messages": [HumanMessage(content=user_input)]}
                    final_answer = ""
                    final_sources = []

                    # st.write_stream을 사용하여 그래프의 중간 및 최종 결과를 스트리밍합니다.
                    for output in st.session_state.graph.stream(inputs):
                        for key, value in output.items():
                            if key == "generate": # 최종 답변 생성 단계
                                final_answer = value.get("generation")
                                final_sources = value.get("documents")

                    st.markdown(final_answer)

                    # 출처 표시
                    with st.expander("자세한 출처 보기 (문장 단위)"):
                        sources_by_url = {}
                        for doc in final_sources:
                            url = doc.metadata.get("source", "N/A")
                            title = doc.metadata.get("title", "No Title")
                            sentence = doc.page_content

                            if url not in sources_by_url:
                                sources_by_url[url] = {"url": url, "title": title, "sentences": []}
                            sources_by_url[url]["sentences"].append(sentence)
                        
                        final_sources_list = list(sources_by_url.values())
                        for source in final_sources_list:
                            st.markdown(f"**- {source['title']}** ([링크]({source['url']}))")
                            for sentence in source['sentences']:
                                st.caption(f"    - {sentence}")
                            st.divider()
                    
                    processing_time = time.time() - start_time
                    st.caption(f"답변 생성 완료! (소요 시간: {processing_time:.2f}초)")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "sources": final_sources_list
                    })

            else: # RAG 파이프라인이 없는 경우
                chain = get_default_chain(current_system_prompt)
                ai_answer = st.write_stream(chain.stream({"question": user_input}))
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_answer, "sources": []}
                )

    except Exception as e:
        error_message = f"죄송합니다, 답변 생성 중 오류가 발생했습니다: {e}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})

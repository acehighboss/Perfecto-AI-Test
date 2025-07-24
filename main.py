import subprocess
import sys

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
from RAG.chain_builder import get_conversational_rag_chain, get_default_chain

# --- 페이지 설정 ---
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="⚙️")
st.title("⚙️ Advanced RAG Chatbot")
st.markdown(
    """
    **병렬 크롤링**, **다단계 필터링**, **문장 단위 출처 표시** 기능이 적용된 RAG 챗봇입니다.
    """
)

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 주어진 컨텍스트만을 사용하여 사용자의 질문에 답변하는 AI 어시스턴트입니다. 항상 친절하고, 정확한 정보를 한국어로 상세하게 전달해주세요. 컨텍스트에 없는 내용은 답변할 수 없다고 솔직하게 말해주세요."

# --- 사이드바 UI ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("LLAMA_CLOUD_API_KEY, GOOGLE_API_KEY, COHERE_API_KEY를 Streamlit secrets에 설정해야 합니다.")
    st.divider()
    
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
                with st.spinner("문서를 병렬로 분석하고 RAG 파이프라인을 준비 중입니다..."):
                    st.session_state.retriever = get_retriever_from_source(source_type, source_input)
                
                if st.session_state.retriever:
                    st.success("분석이 완료되었습니다! 이제 질문해보세요.")
                else:
                    st.error("분석에 실패했습니다. API 키나 URL/파일 상태를 확인해주세요.")
            else:
                st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.experimental_rerun()

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

    current_system_prompt = st.session_state.system_prompt

    try:
        with st.chat_message("assistant"):
            if st.session_state.retriever:
                with st.spinner("답변을 생성하고 있습니다..."):
                    processing_start_time = time.time()
                    
                    # 1. Retriever를 사용하여 관련 문서를 가져옵니다.
                    retrieved_docs = st.session_state.retriever.invoke(user_input)
                    
                    # 2. 가져온 문서로 RAG 체인을 실행합니다.
                    rag_chain = get_conversational_rag_chain(
                        retriever=lambda x: retrieved_docs, # 이미 가져온 문서를 그대로 사용
                        system_prompt=current_system_prompt
                    )
                    ai_answer = rag_chain.invoke(user_input)
                    
                    processing_time = time.time() - processing_start_time

                    # --- 요청된 JSON 출력 형식에 맞게 재구성 ---
                    sources_by_url = {}
                    for doc in retrieved_docs:
                        url = doc.metadata.get("source", "N/A")
                        title = doc.metadata.get("title", "No Title")
                        sentence = doc.page_content

                        if url not in sources_by_url:
                            sources_by_url[url] = {"url": url, "title": title, "sentences": []}
                        sources_by_url[url]["sentences"].append(sentence)
                    
                    final_sources = list(sources_by_url.values())

                    # 최종 결과 객체
                    response_json = {
                        "answer": ai_answer,
                        "sources": final_sources,
                        "processing_time": f"{processing_time:.2f}초"
                    }

                    # 화면에 표시
                    st.markdown(response_json["answer"])
                    with st.expander("자세한 출처 보기 (문장 단위)"):
                        st.json(response_json) # 디버깅 및 확인용으로 JSON 전체 출력
                        for source in response_json["sources"]:
                            st.markdown(f"**- {source['title']}** ([링크]({source['url']}))")
                            for sentence in source['sentences']:
                                st.caption(f"    - {sentence}")
                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_json["answer"], 
                        "sources": response_json["sources"]
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

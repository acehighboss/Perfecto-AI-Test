import subprocess
import sys
import time
import json
import streamlit as st

# Streamlit Cloud 환경에 맞는 Playwright 브라우저 설치
try:
    subprocess.run(
        [f"{sys.executable}", "-m", "playwright", "install"],
        check=True,
        capture_output=True,
        text=True
    )
except subprocess.CalledProcessError as e:
    print("Playwright 브라우저 설치 실패. 에러 로그:")
    print(e.stdout)
    print(e.stderr)
    raise

import nest_asyncio
nest_asyncio.apply()

from RAG.rag_pipeline import get_retriever_from_source
from RAG.chain_builder import get_conversational_rag_chain, get_default_chain
from RAG.rag_config import RAGConfig

# --- 페이지 설정 ---
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="⚙️")
st.title("⚙️ Advanced RAG Chatbot")
st.markdown(
    """
    **병렬 크롤링**, **다단계 필터링**, **문장 단위 출처 표시** 기능이 적용된 RAG 챗봇입니다.
    사이드바에서 RAG 파이프라인의 주요 파라미터를 실시간으로 조절하며 성능을 테스트할 수 있습니다.
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
                    st.session_state.retriever = get_retriever_from_source(
                        source_type, 
                        source_input,
                        rag_params=st.session_state.get("rag_params", {}) # 현재 설정된 RAG 파라미터 전달
                    )
                
                if st.session_state.retriever:
                    st.success("분석이 완료되었습니다! 이제 질문해보세요.")
                else:
                    st.error("분석에 실패했습니다. API 키나 URL/파일 상태를 확인해주세요.")
            else:
                st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

    st.divider()

    # --- RAG 파라미터 동적 설정 UI ---
    st.subheader("🔧 RAG 파라미터 조절")
    st.info("파라미터 변경 후, **분석 시작** 버튼을 다시 눌러야 적용됩니다.")

    # 세션 상태에 rag_params가 없으면 기본값으로 초기화
    if "rag_params" not in st.session_state:
        st.session_state.rag_params = {
            "bm25_top_k": RAGConfig.BM25_TOP_K,
            "rerank_top_n": RAGConfig.RERANK_1_TOP_N,
            "final_docs_count": RAGConfig.FINAL_DOCS_COUNT
        }

    st.session_state.rag_params["bm25_top_k"] = st.slider(
        "BM25 검색 문서 수 (1단계)", 10, 100, st.session_state.rag_params["bm25_top_k"],
        help="키워드 검색(BM25)을 통해 1차적으로 가져올 문서(문장)의 개수입니다."
    )
    st.session_state.rag_params["rerank_top_n"] = st.slider(
        "Cohere Rerank 상위 N개 (2단계)", 5, 50, st.session_state.rag_params["rerank_top_n"],
        help="1단계에서 가져온 문서를 Reranker로 재정렬한 후, 상위 몇 개를 선택할지 결정합니다."
    )
    st.session_state.rag_params["final_docs_count"] = st.number_input(
        "최종 컨텍스트 문서 수 (3단계)", 1, 10, st.session_state.rag_params["final_docs_count"],
        help="Reranker를 통과한 문서 중, 최종적으로 LLM에 컨텍스트로 전달할 문서의 개수입니다."
    )

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

    current_system_prompt = st.session_state.system_prompt
    
    try:
        with st.chat_message("assistant"):
            if st.session_state.retriever:
                with st.spinner("관련 문서를 찾고 답변을 생성하고 있습니다..."):
                    processing_start_time = time.time()
                    
                    # 1. Retriever를 사용하여 관련 문서를 가져옵니다. (스트리밍을 위해 invoke 사용)
                    retrieved_docs = st.session_state.retriever.invoke(user_input)
                    
                    # 2. 가져온 문서로 RAG 체인을 실행합니다.
                    rag_chain = get_conversational_rag_chain(
                        retriever=lambda x: retrieved_docs, # 이미 가져온 문서를 그대로 사용
                        system_prompt=current_system_prompt
                    )
                    
                    # 스트리밍 답변 생성 및 출력
                    response_stream = rag_chain.stream(user_input)
                    ai_answer = st.write_stream(response_stream)
                    
                    processing_time = time.time() - processing_start_time

                # --- 출처 정보 재구성 및 표시 ---
                with st.expander("자세한 출처 보기 (문장 단위)"):
                    sources_by_url = {}
                    for doc in retrieved_docs:
                        url = doc.metadata.get("source", "N/A")
                        title = doc.metadata.get("title", "No Title")
                        sentence = doc.page_content

                        if url not in sources_by_url:
                            sources_by_url[url] = {"url": url, "title": title, "sentences": []}
                        sources_by_url[url]["sentences"].append(sentence)
                    
                    final_sources = list(sources_by_url.values())
                    
                    for source in final_sources:
                        st.markdown(f"**- {source['title']}** ([링크]({source['url']}))")
                        for sentence in source['sentences']:
                            st.caption(f"    - {sentence}")
                        st.divider()
                
                st.caption(f"답변 생성 완료! (소요 시간: {processing_time:.2f}초)")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": final_sources
                })

            else: # RAG 파이프라인이 없는 경우
                with st.spinner("답변을 생성하고 있습니다..."):
                    chain = get_default_chain(current_system_prompt)
                    ai_answer = st.write_stream(chain.stream({"question": user_input}))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": ai_answer, "sources": []}
                    )

    except Exception as e:
        error_message = f"죄송합니다, 답변 생성 중 오류가 발생했습니다: {e}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})

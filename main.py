import os
import re
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# --- LangChain types ---
try:
    from langchain_core.documents import Document
except Exception:  # older LC
    from langchain.schema import Document  # type: ignore

# --- LLM (OpenAI Chat) ---
_ChatOpenAI = None
try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI  # >=0.1
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI as _ChatOpenAI  # legacy
    except Exception:
        _ChatOpenAI = None

# --- Project modules (파일명 변경 금지) ---
from RAG.rag_pipeline import create_retriever  # 호환 래퍼가 rag_pipeline.py에 있어야 함
from RAG.chain_builder import (
    extract_relevant_sentences,
    build_answer_from_sentences,
)
from file_handler import get_documents_from_files
from text_scraper import clean_html_parallel

# =========================
# App Config
# =========================
load_dotenv()
st.set_page_config(page_title="Perfecto RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 Perfecto RAG Chatbot")
st.caption("업로드/URL을 인덱싱하고, 질문에 필요한 **최소 문장**만 근거로 답합니다. (YouTube=타임코드, PDF=페이지)")

# =========================
# Sidebar (UI 전용)
# =========================
with st.sidebar:
    st.subheader("⚙️ 설정")
    model_name = st.selectbox("LLM 모델", ["gpt-4o", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"], index=1)
    topk_sent = st.slider("표시할 근거 문장 수", 3, 12, 8, 1)
    generate_answer = st.checkbox("생성형 응답 생성 (LLM 사용)", value=True)
    st.markdown("---")
    if os.environ.get("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY 감지됨")
    else:
        st.warning("OPENAI_API_KEY 미설정 — LLM 응답이 비활성화될 수 있습니다.")

# =========================
# Session State
# =========================
if "docs" not in st.session_state:
    st.session_state.docs: List[Document] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =========================
# Inputs (UI 전용)
# =========================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 🔗 URL 입력")
    url_text = st.text_area(
        "여러 개면 줄바꿈/쉼표로 구분",
        height=120,
        placeholder="https://example.com/article\nhttps://youtu.be/xxxxxx\n...",
    )
    fetch_btn = st.button("불러오기 / 인덱싱", use_container_width=True)

with col_right:
    st.markdown("#### 📄 파일 업로드")
    uploaded_files = st.file_uploader(
        "PDF, TXT, DOCX, MD, SRT/VTT 등 (여러 개 가능)",
        type=["pdf", "txt", "md", "docx", "csv", "json", "srt", "vtt"],
        accept_multiple_files=True,
    )

st.markdown("---")
user_query = st.text_input("❓질문을 입력하세요", placeholder="예) 이 문서에서 제안한 핵심 요지는 무엇인가요?")

# =========================
# Helpers (UI에서 쓰는 최소 유틸)
# =========================
def _normalize_urls(block: str) -> List[str]:
    if not block.strip():
        return []
    raw = [x.strip() for x in re.split(r"[\n,]+", block) if x.strip()]
    return list(dict.fromkeys([u for u in raw if re.match(r"^https?://", u)]))  # dedupe keep-order


def _docs_from_urls(urls: List[str]) -> List[Document]:
    if not urls:
        return []
    docs: List[Document] = []
    cleaned = clean_html_parallel(urls)  # 레포 제공 함수 (반환 포맷에 관대)
    if isinstance(cleaned, list) and all(isinstance(x, str) for x in cleaned):
        for u, txt in zip(urls, cleaned):
            docs.append(Document(page_content=txt or "", metadata={"source": u, "url": u}))
    elif isinstance(cleaned, list) and all(isinstance(x, dict) for x in cleaned):
        for item in cleaned:
            txt = item.get("text") or item.get("content") or ""
            meta: Dict[str, Any] = {k: v for k, v in item.items() if k not in ("text", "content")}
            if "source" not in meta and "url" not in meta:
                src = item.get("url") or item.get("canonical") or item.get("source")
                if src:
                    meta["source"] = src
            docs.append(Document(page_content=txt, metadata=meta))
    else:
        for u in urls:
            docs.append(Document(page_content=str(cleaned), metadata={"source": u, "url": u}))
    return docs


def _docs_from_files(files) -> List[Document]:
    if not files:
        return []
    return get_documents_from_files(files)

def _get_llm():
    if not generate_answer or _ChatOpenAI is None:
        return None
    try:
        return _ChatOpenAI(model=model_name, temperature=0.0)
    except Exception:
        return None

# =========================
# 인덱싱 (UI 트리거)
# =========================
if fetch_btn:
    urls = _normalize_urls(url_text)
    url_docs = _docs_from_urls(urls)
    file_docs = _docs_from_files(uploaded_files)

    all_docs = file_docs + url_docs
    st.session_state.docs = all_docs

    if not all_docs:
        st.warning("인덱싱할 문서가 없습니다. URL 또는 파일을 제공하세요.")
        st.session_state.retriever = None
    else:
        # retriever는 즉시 만들어 세션에 보관 (rag_pipeline의 구현 시그니처에 맞춰 source_documents 키를 우선 시도)
        try:
            st.session_state.retriever = create_retriever(source_documents=all_docs)
        except TypeError:
            st.session_state.retriever = create_retriever(all_docs)
        st.success(f"인덱싱 완료! 문서 수: {len(all_docs)}")

# =========================
# 핵심 변경 블록 적용 (질의 처리)
# =========================
if user_query:
    # 1) 검색: 기존 로직과 동일하게 생성(또는 이미 만든 retriever 사용)
    try:
        retriever = st.session_state.retriever or create_retriever()
    except Exception:
        retriever = st.session_state.retriever  # 없으면 None

    if retriever is None:
        if not st.session_state.docs:
            st.info("먼저 '불러오기 / 인덱싱'을 눌러 문서를 등록하세요.")
        docs_for_search = st.session_state.docs
    else:
        try:
            docs_for_search = retriever.get_relevant_documents(user_query)
        except Exception:
            # retriever가 문서를 내장하지 않는 구현이라면 전체 문서에서 후처리
            docs_for_search = st.session_state.docs

    if not docs_for_search:
        st.warning("검색된 문서가 없습니다.")
    else:
        # 2) 질문에 필요한 '최소' 문장만 추출
        picked = extract_relevant_sentences(user_query, docs_for_search, max_sentences=topk_sent)

        # 3) LLM이 오직 해당 문장들만 근거로 답변 생성
        answer_text = ""
        llm = _get_llm()
        if generate_answer and llm is not None:
            try:
                answer_text = build_answer_from_sentences(llm, user_query, picked)
            except Exception:
                answer_text = ""

        # 4) UI 출력
        st.markdown("### 🧠 답변")
        if answer_text:
            st.write(answer_text)
        else:
            st.write("생성형 응답 비활성화됨. 아래 **근거 문장**을 참고하세요.")

        with st.expander("🔎 출처(질문과 가장 관련 있는 최소 문장)"):
            for i, it in enumerate(picked, 1):
                st.markdown(
                    f"**S{i}.** {it['text']}\n\n"
                    f"<small>출처: {it['citation']}</small>",
                    unsafe_allow_html=True,
                )

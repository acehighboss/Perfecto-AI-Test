from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain types
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# RAG pipeline (search_type="similarity" 유지)
from RAG.rag_pipeline import build_faiss_vectorstore, get_retriever_from_source
from RAG.chain_builder import get_conversational_rag_chain, get_default_chain

# 파일 업로드 → Document 변환 (repo에 존재한다고 가정)
try:
    from file_handler import get_documents_from_files
except Exception:
    get_documents_from_files = None

# URL 크롤링 시도 (repo의 스크레이퍼 우선 사용)
def load_urls_as_documents(urls: List[str]) -> List[Document]:
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        return []

    # 1) 우선 repo의 text_scraper 사용
    try:
        from text_scraper import clean_html_parallel, filter_noise
        rows = clean_html_parallel(urls)  # repo 구현체 가정: [{url, title, text, ...}, ...]
        docs: List[Document] = []
        for row in rows:
            txt = row.get("text") or ""
            txt = filter_noise(txt) if "filter_noise" in globals() else txt
            title = row.get("title") or ""
            source = row.get("url") or row.get("source") or ""
            if not txt.strip():
                continue
            meta = {
                "source": source,
                "title": title,
                "type": "web",
            }
            docs.append(Document(page_content=txt, metadata=meta))
        return docs
    except Exception:
        pass

    # 2) 폴백: requests + BeautifulSoup
    docs: List[Document] = []
    try:
        import requests
        from bs4 import BeautifulSoup
        for u in urls:
            try:
                r = requests.get(u, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                # 기본 텍스트 추출 (간단 버전)
                for s in soup(["script", "style", "noscript"]):
                    s.extract()
                title = soup.title.string.strip() if soup.title and soup.title.string else u
                text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
                if not text:
                    continue
                docs.append(Document(page_content=text, metadata={"source": u, "title": title, "type": "web"}))
            except Exception:
                continue
    except Exception:
        pass

    return docs


# ============ Streamlit App ============

load_dotenv()
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- Session State ---
if "ALL_DOCS" not in st.session_state:
    st.session_state.ALL_DOCS: List[Document] = []
if "VECTORSTORE" not in st.session_state:
    st.session_state.VECTORSTORE = None
if "RETRIEVER" not in st.session_state:
    st.session_state.RETRIEVER = None
if "CHAT_HISTORY" not in st.session_state:
    st.session_state.CHAT_HISTORY = []  # [({"role": "user"/"assistant"}, "content"), ...]
if "SOURCES_CACHE" not in st.session_state:
    st.session_state.SOURCES_CACHE: List[Dict[str, Any]] = []
if "INDEX_BUILT_AT" not in st.session_state:
    st.session_state.INDEX_BUILT_AT = None

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Settings")

# 인덱싱 파라미터
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=400, value=120, step=10)

# RAG Debug
debug_rag = st.sidebar.checkbox("RAG Debug", value=False)

# 인덱스 관리
col_reset1, col_reset2 = st.sidebar.columns(2)
with col_reset1:
    if st.button("🗑️ 인덱스 초기화", use_container_width=True):
        st.session_state.ALL_DOCS = []
        st.session_state.VECTORSTORE = None
        st.session_state.RETRIEVER = None
        st.session_state.SOURCES_CACHE = []
        st.session_state.INDEX_BUILT_AT = None
        st.success("인덱스를 초기화했습니다.")
with col_reset2:
    if st.button("🧹 대화 초기화", use_container_width=True):
        st.session_state.CHAT_HISTORY = []
        st.success("대화 기록을 초기화했습니다.")

st.sidebar.markdown("---")
st.sidebar.caption("현재 문서 수: **{}**".format(len(st.session_state.ALL_DOCS)))
if st.session_state.INDEX_BUILT_AT:

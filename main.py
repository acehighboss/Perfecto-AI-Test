from __future__ import annotations

import os
import re
import requests
from typing import List
import streamlit as st

from langchain_core.documents import Document

# 레포 내부 모듈
from RAG.rag_pipeline import get_retriever_from_source
from RAG.chain_builder import (
    extract_relevant_sentences,
    build_answer_from_sentences,
    get_conversational_rag_chain,  # 레포 호환 (미사용 시 삭제 가능)
    get_default_chain,             # 레포 호환 (미사용 시 삭제 가능)
)

# LLM (환경에 맞게 교체 가능)
try:
    from langchain_openai import ChatOpenAI
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
except Exception:
    llm = None  # 폴백: build_answer_from_sentences에서 안전 처리


st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 출처 기반 RAG 챗봇 (생성 요약 기본 ON)")

# ======================
# Sidebar Controls
# ======================
st.sidebar.header("응답 설정")
ALLOW_GENERATION = st.sidebar.checkbox("생성 요약 켜기", value=True)
SHOW_MINIMAL_EVIDENCE = st.sidebar.checkbox("출처는 최소 문장만 표시", value=True)
TOP_K_SENT = st.sidebar.slider("출처 문장 개수(k)", min_value=3, max_value=15, value=8, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("문서 업로드")
uploaded_files = st.sidebar.file_uploader(
    "텍스트/PDF/자막 등 (여러 개 선택 가능)", accept_multiple_files=True
)

st.sidebar.subheader("URL 추가")
urls_text = st.sidebar.text_area("줄바꿈으로 여러 URL 입력 가능", height=120, placeholder="https://example.com/article-1\nhttps://example.com/article-2")

# ======================
# Load Documents
# ======================

def _load_from_uploaded(files) -> List[Document]:
    docs: List[Document] = []
    for f in files or []:
        name = getattr(f, "name", "uploaded")
        try:
            content_bytes = f.read()
            try:
                text = content_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text = content_bytes.decode("cp949", errors="ignore")
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            docs.append(Document(page_content=text, metadata={"source": f"file://{name}"}))
        except Exception as e:
            st.sidebar.warning(f"파일 로드 실패: {name} ({e})")
    return docs


def _fetch_url_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        # 아주 단순한 텍스트 추출(레포 내 text_scraper가 있다면 교체 사용 권장)
        html = resp.text
        # 스크립트/스타일 제거
        html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        # 태그 제거
        text = re.sub(r"(?is)<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception as e:
        st.sidebar.warning(f"URL 요청 실패: {url} ({e})")
        return ""


def _load_from_urls(urls_raw: str) -> List[Document]:
    docs: List[Document] = []
    urls = [u.strip() for u in (urls_raw or "").splitlines() if u.strip()]
    for u in urls:
        txt = _fetch_url_text(u)
        if not txt:
            continue
        docs.append(Document(page_content=txt, metadata={"source": u, "url": u}))
    return docs


# 문서 합치기 & 리트리버 구성
docs_all: List[Document] = []
docs_all.extend(_load_from_uploaded(uploaded_files))
docs_all.extend(_load_from_urls(urls_text))

if "retriever" not in st.session_state or st.button("🔄 리트리버 재구성"):
    st.session_state["retriever"] = get_retriever_from_source(docs_all, top_k=6)
retriever = st.session_state["retriever"]

# ======================
# Chat Section
# ======================

question = st.text_input("질문을 입력하세요", placeholder="예) SK텔레콤이 자체 개발한 LLM의 이름과 특징은?")

if question:
    # 1) Retrieve
    docs = retriever.get_relevant_documents(question)

    # 2) Extract minimal evidence sentences
    sentences = extract_relevant_sentences(docs, question, k=TOP_K_SENT, dedupe=True)

    # 3) Build answer from sentences
    result = build_answer_from_sentences(
        llm, question, sentences,
        allow_generation=ALLOW_GENERATION
    )

    st.subheader("🧠 답변")
    st.markdown(result["answer"])

    st.subheader("🔎 출처(질문과 가장 관련 있는 최소 문장)")
    if SHOW_MINIMAL_EVIDENCE:
        for i, s in enumerate(result["sentences"], 1):
            extras = []
            if s.get("page"):
                extras.append(f"p.{s['page']}")
            if s.get("timecode"):
                extras.append(str(s["timecode"]))
            suffix = f" ({', '.join(extras)})" if extras else ""
            st.markdown(f"**S{i}.** {s['text']}\n\n출처: {s.get('source','')}{suffix}")
    else:
        st.info("출처 표시가 비활성화되어 있습니다. 사이드바에서 켤 수 있어요.")

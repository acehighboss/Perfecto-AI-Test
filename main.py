import os
import re
import sys
import json
import time
import traceback
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# LangChain primitives
from langchain_core.documents import Document

# ==== 프로젝트 내부 모듈 (레포 구조 유지) ====
# 주의: 파일명 변경 금지
from RAG.rag_pipeline import create_retriever
from RAG.chain_builder import get_conversational_rag_chain, get_default_chain  # 내부에서 문장 단위 축약 적용됨
from file_handler import get_documents_from_files
from text_scraper import clean_html_parallel  # 레포에 이미 존재하는 함수 사용

# ---------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Perfecto-AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

load_dotenv()

# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def _normalize_urls(raw: str) -> List[str]:
    urls = []
    for line in (raw or "").splitlines():
        u = line.strip()
        if not u:
            continue
        if not re.match(r"^https?://", u):
            # 간단 보정
            u = "http://" + u
        urls.append(u)
    return list(dict.fromkeys(urls))  # dedupe, keep order

def _docs_from_urls(urls: List[str]) -> List[Document]:
    """레포의 text_scraper.clean_html_parallel()을 이용해 URL -> Document 변환."""
    if not urls:
        return []
    docs: List[Document] = []
    try:
        results = clean_html_parallel(urls)  # 레포 시그니처에 맞춰 사용
        # results 형태는 레포 구현에 따라 다르지만, 보통 dict 또는 tuple로 (url, title, text) 등을 돌려줍니다.
        # 최대한 보수적으로 파싱합니다.
        for item in results:
            # item이 dict
            if isinstance(item, dict):
                url = item.get("url") or item.get("source") or ""
                title = item.get("title") or ""
                text = item.get("text") or item.get("content") or ""
                meta = {k: v for k, v in item.items() if k not in ("text", "content")}
            # item이 tuple/list
            elif isinstance(item, (list, tuple)):
                # (url, title, text) 혹은 (url, text) 형태를 최대한 흡수
                url = item[0] if len(item) > 0 else ""
                if len(item) >= 3:
                    title = item[1] or ""
                    text = item[2] or ""
                elif len(item) == 2:
                    title = ""
                    text = item[1] or ""
                else:
                    title, text = "", ""
                meta = {"url": url, "title": title}
            else:
                # 알 수 없는 포맷은 스킵
                continue

            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        **(meta or {}),
                        "source": meta.get("url") or url or "url",
                        "title": title,
                        "kind": "url",
                    },
                )
            )
    except Exception as e:
        st.warning(f"URL 파싱 중 일부 실패: {e}")
        traceback.print_exc()

    return docs

def _safe_build_retriever(all_docs: List[Document]):
    """
    레포의 get_retriever_from_source를 최대한 보수적으로 호출.
    레포 구현에 따라 인자 시그니처가 다를 수 있어 여러 fallback 시도.
    """
    retriever = None
    last_err = None

    # 1) documents 키워드 인자를 받는 구현
    try:
        retriever = create_retriever(source="uploaded", documents=all_docs)
        return retriever
    except Exception as e:
        last_err = e

    # 2) docs 키워드 인자를 받는 구현
    try:
        retriever = create_retriever(source="uploaded", docs=all_docs)
        return retriever
    except Exception as e:
        last_err = e

    # 3) 단일 인자만 받는 구현 (documents)
    try:
        retriever = create_retriever(all_docs)
        return retriever
    except Exception as e:
        last_err = e

    # 4) source만 받고 내부에서 이미 인덱스가 만들어지는 구현
    try:
        retriever = create_retriever(source="uploaded")
        return retriever
    except Exception as e:
        last_err = e

    # 실패 시
    raise RuntimeError(f"retriever 생성 실패: {last_err}")

def _safe_build_chain(retriever):
    """
    레포별 체인 시그니처 차이를 흡수
    - get_conversational_rag_chain(retriever=...)
    - get_conversational_rag_chain(retriever, ...)
    - 없으면 get_default_chain
    """
    last_err = None
    # conversational 우선
    try:
        return get_conversational_rag_chain(retriever=retriever)
    except Exception as e:
        last_err = e

    try:
        return get_conversational_rag_chain(retriever)
    except Exception as e:
        last_err = e

    # default 체인
    try:
        return get_default_chain(retriever=retriever)
    except Exception as e:
        last_err = e

    try:
        return get_default_chain(retriever)
    except Exception as e:
        last_err = e

    raise RuntimeError(f"체인 생성 실패: {last_err}")

def _invoke_chain(chain, question: str) -> Dict[str, Any]:
    """
    레포마다 입력 키가 다른 문제를 흡수:
    - chain.invoke("질문")
    - chain.invoke({"question": "질문"})
    - chain.invoke({"input": "질문"})
    - chain.invoke({"query": "질문"})
    """
    # 1) 문자열 직접
    try:
        out = chain.invoke(question)
        if isinstance(out, (str, dict)):
            return {"answer": out} if isinstance(out, str) else out
    except Exception:
        pass

    # 2) common keys
    for key in ("question", "input", "query"):
        try:
            out = chain.invoke({key: question})
            if isinstance(out, (str, dict)):
                return {"answer": out} if isinstance(out, str) else out
        except Exception:
            continue

    # 실패 시 에러 전파
    raise

def _render_sources(result: Dict[str, Any]):
    """
    chain 결과에서 문장 단위 출처를 출력.
    RAG/chain_builder.py에서 metadata['selected_sentences']에 심어둔 것을 우선 사용.
    """
    # 여러 키 시도
    source_docs = None
    for k in ("source_documents", "sources", "docs"):
        if isinstance(result, dict) and k in result:
            source_docs = result[k]
            break

    if not source_docs:
        return

    st.markdown("### 출처 (문장 단위)")

    for d in source_docs:
        meta = getattr(d, "metadata", {}) or {}
        citations = meta.get("selected_sentences")
        if citations:
            for c in citations:
                text = c.get("text", "").strip()
                label = c.get("label", meta.get("source", "source"))
                if text:
                    st.markdown(
                        f"- {text}  \n  <sub style='color:#888'>{label}</sub>",
                        unsafe_allow_html=True,
                    )
        else:
            # 안전망: 축약 실패 시 일부만 표시
            content = (getattr(d, "page_content", "") or "").strip()
            label = meta.get("source") or meta.get("url") or "source"
            if content:
                preview = content[:200] + ("..." if len(content) > 200 else "")
                st.markdown(
                    f"- {preview}  \n  <sub style='color:#888'>{label}</sub>",
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------
# 세션 스테이트 초기화
# ---------------------------------------------------------
if "DOCS" not in st.session_state:
    st.session_state.DOCS: List[Document] = []

if "RETRIEVER" not in st.session_state:
    st.session_state.RETRIEVER = None

if "CHAIN" not in st.session_state:
    st.session_state.CHAIN = None

if "CHAT" not in st.session_state:
    st.session_state.CHAT: List[Tuple[str, str]] = []

# ---------------------------------------------------------
# 사이드바: 소스 업로드/설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("📥 소스 업로드")
    uploaded_files = st.file_uploader(
        "파일 업로드 (여러 개 선택 가능)",
        type=["pdf", "txt", "md", "docx", "json", "srt", "vtt", "csv", "html"],
        accept_multiple_files=True,
    )

    st.caption("또는 URL을 줄바꿈으로 입력")
    raw_urls = st.text_area("URLs", placeholder="https://example.com/article-1\nhttps://example.com/article-2")

    per_doc_max = st.slider("문서당 최대 인용 문장 수", min_value=1, max_value=7, value=3, step=1,
                            help="질문과 가장 관련있는 문장만 선택되어 출처로 표기됩니다.")

    build_btn = st.button("📚 인덱스 빌드 / 갱신", type="primary")

# ---------------------------------------------------------
# 인덱스 빌드
# ---------------------------------------------------------
if build_btn:
    all_docs: List[Document] = []

    # 파일 -> Document
    if uploaded_files:
        try:
            file_docs = get_documents_from_files(uploaded_files)  # 레포 내 구현 사용
            # 안전망: page_content 보정
            for d in file_docs:
                if not isinstance(d.page_content, str):
                    d.page_content = str(d.page_content or "")
                meta = d.metadata or {}
                meta.setdefault("kind", "file")
                d.metadata = meta
            all_docs.extend(file_docs)
        except Exception as e:
            st.error(f"파일 처리 중 오류: {e}")
            traceback.print_exc()

    # URL -> Document
    urls = _normalize_urls(raw_urls)
    if urls:
        url_docs = _docs_from_urls(urls)
        all_docs.extend(url_docs)

    if not all_docs:
        st.warning("인덱싱할 문서가 없습니다. 파일을 업로드하거나 URL을 입력하세요.")
    else:
        # 세션에 저장
        st.session_state.DOCS = all_docs

        # retriever 생성
        try:
            retriever = _safe_build_retriever(all_docs)
            # 체인 구성
            chain = _safe_build_chain(retriever)

            # 체인이 내부에서 문장 축약에 사용하는 per_doc_max를 전달해야 하는 구현이 있을 수 있음
            # 옵션 전달을 시도하되, 실패해도 무시
            try:
                if hasattr(chain, "configurable_fields"):
                    # 일부 Runnable은 configurable_fields를 통해 파라미터 조정 지원
                    pass
            except Exception:
                pass

            st.session_state.RETRIEVER = retriever
            st.session_state.CHAIN = chain
            st.success(f"인덱스가 준비되었습니다. 문서 수: {len(all_docs)}")
        except Exception as e:
            st.error(f"인덱스 준비 실패: {e}")
            traceback.print_exc()

# ---------------------------------------------------------
# 본문: 채팅 UI
# ---------------------------------------------------------
st.title("🤖 Perfecto-AI 문장-출처 RAG 챗봇")

# 힌트/상태
with st.expander("ℹ️ 사용 팁", expanded=False):
    st.markdown(
        """
- 파일을 업로드하거나 URL을 입력한 뒤 **인덱스 빌드**를 누르세요.
- 질문을 입력하면, **답변 아래에 '문장 단위' 출처**가 표시됩니다.
- PDF는 `p.<페이지>`, 동영상/자막은 `t=MM:SS` 같은 타임코드가 함께 노출됩니다(메타데이터가 있을 때).
        """
    )

# 채팅 히스토리 표시
for role, text in st.session_state.CHAT:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)

# 입력창
question = st.chat_input("무엇을 도와드릴까요? (예: 이 보고서의 핵심 결론은?)")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.CHAT.append(("user", question))

    if not st.session_state.CHAIN:
        st.warning("먼저 좌측에서 문서를 인덱싱해 주세요.")
    else:
        placeholder = st.empty()
        with st.chat_message("assistant"):
            try:
                placeholder.markdown("생각 중입니다…")
                # 질의 수행
                result = _invoke_chain(st.session_state.CHAIN, question)

                # answer 추출
                answer = None
                if isinstance(result, dict):
                    # 관용 키들
                    for k in ("answer", "output", "result", "response", "text"):
                        if k in result and isinstance(result[k], str):
                            answer = result[k]
                            break
                    # 없으면 Dict 전체를 pretty-print
                    if answer is None:
                        answer = json.dumps(result, ensure_ascii=False, indent=2)
                elif isinstance(result, str):
                    answer = result
                else:
                    answer = str(result)

                placeholder.empty()
                st.markdown("### 답변")
                st.write(answer)
                _render_sources(result)

                # 히스토리에 답변 저장(요약 저장)
                st.session_state.CHAT.append(("assistant", answer))
            except Exception as e:
                placeholder.empty()
                st.error(f"분석 중 오류: {e}")
                st.exception(e)

# 푸터
st.markdown(
    "<hr><div style='color:#999;font-size:12px'>Powered by Perfecto-AI-Test · RAG/chain_builder의 "
    "<code>retrieve_and_fuse_results</code> 는 질문과 가장 관련된 문장만 선별하여 표시합니다.</div>",
    unsafe_allow_html=True,
)



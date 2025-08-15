from __future__ import annotations
from typing import List, Literal, Any

from langchain_core.documents import Document as LangChainDocument

# 업로드 파일 전용 파서
from file_handler import get_documents_from_uploaded_files

# (선택) URL 인입도 지원하려면 주석 해제
from file_handler import get_documents_from_urls_robust


SourceType = Literal["uploaded_files", "urls"]  # 필요 시 다른 타입 추가


def load_documents(source_type: SourceType, source_input: Any) -> List[LangChainDocument]:
    """
    파이프라인에서 호출하는 통합 로더.
    - source_type == "uploaded_files": Streamlit UploadedFile 리스트를 기대
    - source_type == "urls": 줄바꿈 구분 문자열 또는 URL 문자열 리스트 지원
    반환: List[Document]
    """
    if source_type == "uploaded_files":
        # Streamlit의 st.file_uploader(...) 반환 리스트를 기대
        files = source_input or []
        docs = get_documents_from_uploaded_files(files)
        return docs

    elif source_type == "urls":
        # 문자열 하나/여러 줄 or List[str] 모두 허용
        if isinstance(source_input, str):
            urls = [u.strip() for u in source_input.splitlines() if u.strip()]
        elif isinstance(source_input, (list, tuple)):
            urls = [str(u).strip() for u in source_input if str(u).strip()]
        else:
            urls = []
        if not urls:
            return []
        # 옵션은 기본값으로. 필요하면 호출부에서 넘겨받아 세부 제어 가능
        docs = get_documents_from_urls_robust(
            urls,
            respect_robots=True,
            use_js_render=False,
            js_only_when_needed=True,
        )
        return docs

    else:
        # 알 수 없는 타입이면 빈 리스트
        return []

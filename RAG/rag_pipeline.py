import traceback
from typing import List
from langchain_core.documents import Document
from .retriever_builder import build_retriever

def get_retriever_from_documents(documents: List[Document]):
    """
    이미 로드된 문서 리스트에서 직접 Retriever를 생성합니다.
    """
    if not documents:
        print("처리할 문서를 찾지 못했습니다.")
        return None

    try:
        # 문서 리스트를 retriever_builder로 바로 전달
        retriever = build_retriever(documents)
        return retriever
    except Exception as e:
        print(f"Retriever 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return None

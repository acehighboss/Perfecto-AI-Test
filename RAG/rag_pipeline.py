import nest_asyncio
nest_asyncio.apply()

import asyncio
import time
import traceback

from .data_loader import load_documents
from .retriever_builder import build_retriever

async def get_retriever_from_source_async(source_type, source_input):
    """
    [수정] 소스에서 문서를 로드하고 Retriever와 에러 리스트를 반환합니다.
    """
    start_time = time.time()
    
    print("\n[1단계: 관련 콘텐츠 추출 시작]")
    # [수정] load_documents가 반환하는 (문서, 에러) 튜플을 올바르게 받습니다.
    documents, errors = await load_documents(source_type, source_input)
    
    if errors:
        print("문서 처리 중 다음 오류가 발생했습니다:")
        for error in errors:
            print(f"- {error}")

    if not documents:
        print("처리할 문서를 찾지 못했습니다.")
        # 문서가 없어도 에러 메시지를 UI에 전달하기 위해 빈 retriever와 함께 반환
        return None, errors
    
    print(f"콘텐츠 추출 완료. (소요 시간: {time.time() - start_time:.2f}초)")
    
    retriever = build_retriever(documents)
    
    # [수정] retriever와 함께 errors 리스트를 반환합니다.
    return retriever, errors

def get_retriever_from_source(source_type, source_input):
    """
    [수정] 비동기 함수를 실행하고 (retriever, errors) 튜플을 반환합니다.
    """
    try:
        # 비동기 함수가 (retriever, errors)를 반환하므로 그대로 전달
        return asyncio.run(get_retriever_from_source_async(source_type, source_input))
    except Exception as e:
        print(f"Retriever 생성 중 오류 발생: {e}")
        traceback.print_exc()
        # 예외 발생 시, UI와 형식을 맞추기 위해 (None, 에러 메시지) 형태로 반환
        return None, [f"전체 파이프라인 실행 중 오류가 발생했습니다: {e}"]

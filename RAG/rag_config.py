class RAGConfig:
    """RAG 파이프라인의 모든 설정값을 관리하는 클래스"""

    # ★★★ 1. 문서 분할(Chunking) 설정 ★★★
    CHUNK_SIZE = 800         # 청크 크기 (글자 수). 맥락을 유지하기에 적절한 크기
    CHUNK_OVERLAP = 100      # 청크 간 중첩되는 글자 수. 내용이 잘리지 않도록 보조

    # 2. Retriever가 초기에 가져오는 청크의 수 (후보군 확보)
    BM25_TOP_K = 10
    FAISS_TOP_K = 10

    # 3. Reranker가 재정렬할 청크의 수
    RERANK_TOP_N = 20

    # 4. Reranker 결과 필터링 및 최종 선택
    RERANK_THRESHOLD = 0.2   # 관련성 점수 임계값
    FINAL_DOCS_COUNT = 5     # 최종적으로 LLM에 전달할 최대 청크 수
    FINAL_DOCS_COUNT_FALLBACK = 2 # 필터링 통과 청크가 없을 경우, 최소한으로 전달할 청크 수

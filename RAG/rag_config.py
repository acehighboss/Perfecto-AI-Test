class RAGConfig:
    """RAG 파이프라인의 모든 설정값을 관리하는 클래스"""
    
    # 1. Retriever가 초기에 가져오는 문서(문장)의 수 (후보군 확보)
    BM25_TOP_K = 25       # 키워드 검색 결과 수 (상향)
    FAISS_TOP_K = 25      # 의미 검색 결과 수 (상향)

    # 2. Reranker가 재정렬할 문서의 수
    RERANK_TOP_N = 30     # 하이브리드 검색 결과를 합친 후, 재정렬할 상위 문서 수 (상향)

    # 3. Reranker 결과 필터링 및 최종 선택
    RERANK_THRESHOLD = 0.1   # 관련성 점수 임계값 (하향 조정하여 덜 엄격하게)
    FINAL_DOCS_COUNT = 5     # 최종적으로 LLM에 전달할 최대 문서 수

    # ★★★ 필터링을 통과한 문서가 하나도 없을 경우, 최소한으로 전달할 문서 수 (Fallback) ★★★
    FINAL_DOCS_COUNT_FALLBACK = 2

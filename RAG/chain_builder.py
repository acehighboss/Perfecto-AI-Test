from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from typing import List
from collections import Counter

# retriever_builder에서 retriever 생성 함수를 직접 가져와 사용
from .retriever_builder import build_retriever
from .rag_config import RAGConfig

def format_docs_for_llm(docs: list[Document]) -> str:
    """문서 리스트를 LLM 프롬프트 형식에 맞게 변환합니다."""
    if not docs:
        return "No context provided."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def get_conversational_rag_chain(documents: List[Document], system_prompt: str):
    """
    '2-단계 검색' 파이프라인을 구현하여 정확도를 높인 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    # 1. 모든 문서에 대한 청크를 미리 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,
    )
    all_chunks = text_splitter.split_documents(documents)

    if not all_chunks:
        # 처리할 문서가 없는 경우, 고정된 답변을 반환하는 간단한 체인 생성
        return RunnableLambda(lambda query: {"answer": "로드된 문서가 없거나 내용이 비어있습니다.", "source_documents": []})

    # 2. (1단계용) 모든 청크에 대한 예비 검색기(BM25) 생성
    pre_retriever = BM25Retriever.from_documents(all_chunks)
    pre_retriever.k = 10  # 예비 검색에서는 후보군을 넓게 10개 정도 가져옴

    def route_and_retrieve(query: str):
        # 3. [1단계] 예비 검색 실행: 모든 문서에서 후보 청크 탐색
        candidate_chunks = pre_retriever.invoke(query)

        if not candidate_chunks:
            return {"answer": "질문과 관련된 내용을 찾을 수 없습니다.", "source_documents": []}

        # 4. 후보 청크들의 출처(source)를 분석하여 가장 가능성 높은 '정답 문서' 결정
        source_counts = Counter(doc.metadata.get("source") for doc in candidate_chunks)
        most_common_source = source_counts.most_common(1)[0][0]
        print(f"✅ 1단계 검색 결과: '{most_common_source}' 문서를 정답 문서로 선택")

        # 5. '정답 문서'에 해당하는 원본 문서들만 필터링
        focused_docs = [doc for doc in documents if doc.metadata.get("source") == most_common_source]

        # 6. [2단계] 본 검색 실행: '정답 문서' 안에서만 정교한 검색기(build_retriever)를 동적으로 생성
        main_retriever = build_retriever(focused_docs)
        if not main_retriever:
            return {"answer": "관련 문서에서 정보 검색기를 생성하는 데 실패했습니다.", "source_documents": []}

        # 7. 최종 답변 생성
        rag_prompt = ChatPromptTemplate.from_template(
            f"{system_prompt}\n\n**Context:**\n{{context}}\n\n**User's Question:**\n{{question}}\n\n**Answer (in Korean):**"
        )

        main_rag_chain = (
            RunnableParallel({
                "context": main_retriever | RunnableLambda(format_docs_for_llm),
                "question": RunnablePassthrough(),
                "source_documents": main_retriever
            })
            | {
                "answer": (lambda x: {"context": x["context"], "question": x["question"]}) | rag_prompt | llm | StrOutputParser(),
                "source_documents": lambda x: x["source_documents"]
            }
        )
        
        return main_rag_chain.invoke(query)

    # 최종적으로 실행될 함수를 RunnableLambda로 감싸서 반환
    return RunnableLambda(route_and_retrieve)

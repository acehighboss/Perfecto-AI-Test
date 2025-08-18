from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict

# retriever_builder에서 retriever 생성 함수를 직접 가져와 사용
from .retriever_builder import build_retriever

def format_docs_for_llm(docs: list[Document]) -> str:
    """문서 리스트를 LLM 프롬프트 형식에 맞게 변환합니다."""
    if not docs:
        return "No context provided."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def get_conversational_rag_chain(documents: List[Document], system_prompt: str):
    """
    '질문 라우팅'을 통해 가장 관련 있는 문서를 먼저 선택하고,
    해당 문서 내에서만 검색하여 답변을 생성하는 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    # 1. 각 문서의 고유한 source를 키로 하여 문서를 그룹화
    docs_by_source: Dict[str, List[Document]] = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown_source")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    # 2. 각 문서의 내용을 요약하여 라우팅을 위한 정보 생성
    # (여기서는 간단하게 각 문서의 첫 1000자를 대표로 사용)
    doc_summaries = []
    for source, docs in docs_by_source.items():
        # title을 우선적으로 사용하고, 없으면 source 사용
        title = docs[0].metadata.get("title", source)
        content_preview = " ".join([d.page_content for d in docs])[:1000]
        doc_summaries.append(f"Source: {source}\nTitle: {title}\nContent: {content_preview}...")

    summaries_str = "\n\n===\n\n".join(doc_summaries)

    # 3. 질문 라우팅 체인: LLM을 사용해 가장 적절한 문서를 선택
    routing_prompt = ChatPromptTemplate.from_template(
        "아래는 여러 문서의 요약본입니다. 사용자의 질문에 가장 잘 답변할 수 있는 문서의 'Source'를 하나만 정확히 골라주세요.\n\n"
        "--- 문서 요약본 ---\n{summaries}\n\n"
        "--- 사용자의 질문 ---\n{question}\n\n"
        "가장 관련 있는 문서의 Source:"
    )
    
    routing_chain = (
        {"summaries": lambda x: summaries_str, "question": RunnablePassthrough()}
        | routing_prompt
        | llm
        | StrOutputParser()
    )

    # 4. 메인 RAG 체인 구성
    def retrieve_and_answer(query: str):
        # 4-1. 라우팅을 통해 최적의 문서 source 결정
        chosen_source = routing_chain.invoke(query).strip()
        print(f"질문 라우팅 결과 -> 선택된 문서: {chosen_source}")

        # 4-2. 선택된 문서의 내용으로만 Retriever를 동적으로 생성
        relevant_docs = docs_by_source.get(chosen_source)
        if not relevant_docs:
            return {"answer": "질문에 맞는 문서를 찾지 못했습니다.", "source_documents": []}
        
        retriever = build_retriever(relevant_docs)
        if not retriever:
            return {"answer": "정보 검색기를 생성하는 데 실패했습니다.", "source_documents": []}
            
        # 4-3. 선택된 문서 내에서만 정보 검색 및 답변 생성
        rag_prompt = ChatPromptTemplate.from_template(
            f"{system_prompt}\n\n**Context:**\n{{context}}\n\n**User's Question:**\n{{question}}\n\n**Answer (in Korean):**"
        )

        rag_chain_for_source = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs_for_llm),
                "question": RunnablePassthrough(),
                "source_documents": retriever
            })
            | {
                "answer": (lambda x: {"context": x["context"], "question": x["question"]}) | rag_prompt | llm | StrOutputParser(),
                "source_documents": lambda x: x["source_documents"]
            }
        )
        
        return rag_chain_for_source.invoke(query)

    # 최종적으로 실행될 함수를 반환
    return RunnableLambda(retrieve_and_answer)

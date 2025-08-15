from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

def format_docs_with_metadata(docs: list[Document]) -> str:
    """문서 리스트를 LLM 프롬프트 형식에 맞게 변환합니다."""
    if not docs:
        return "No context provided."
    
    # 문서 내용을 하나의 문자열로 합치되, 각 문서의 내용을 명확히 구분
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def get_conversational_rag_chain(retriever, system_prompt: str):
    """
    Retriever를 사용하여 문맥을 찾고, 이를 기반으로 답변을 생성하는 RAG 체인을 구성합니다.
    최종적으로 답변(answer)과 출처(source_documents)를 딕셔너리 형태로 반환합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    # RAG 프롬프트 템플릿
    rag_prompt_template = f"""{system_prompt}

You must answer the user's question based *only* on the provided "Context".
If the information is not in the context, you must state that you cannot answer.
Provide a detailed and synthesized answer in Korean.

**Context:**
{{context}}

**User's Question:**
{{question}}

**Answer (in Korean):**
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    # 1. Retriever를 사용하여 문맥 검색
    # 2. 검색된 문맥(docs)을 'source_documents' 키로 그대로 전달하고,
    #    'context' 키에는 LLM에 입력할 포맷된 문자열을 전달
    retrieval_chain = RunnableParallel(
        {
            "source_documents": retriever,
            "context": retriever | RunnableLambda(format_docs_with_metadata),
            "question": RunnablePassthrough(),
        }
    )
    
    # 3. 프롬프트, 모델, 파서를 연결하여 최종 답변 생성
    rag_chain = (
        retrieval_chain
        | {
            "answer": rag_prompt | llm | StrOutputParser(),
            "source_documents": lambda x: x["source_documents"], # 문맥을 그대로 전달
          }
    )
    
    return rag_chain

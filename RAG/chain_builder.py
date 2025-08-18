from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

def get_keyword_generation_chain():
    """
    사용자의 질문에서 검색 키워드를 생성하는 LLM 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 키워드 생성을 위한 새로운 프롬프트 템플릿
    keyword_prompt_template = """당신은 사용자의 질문에서 효과적인 검색 키워드를 생성하는 전문 AI 어시스턴트입니다.
아래 사용자의 질문을 분석하여, 질문의 핵심 내용과 가장 관련성이 높은 한국어 키워드 목록을 간결하게 추출해주세요.
검색 범위를 넓히기 위해 동의어와 관련 개념도 고려할 수 있습니다.
키워드는 쉼표로 구분된 리스트 형태로 제공해주세요.

**사용자 질문:**
{question}

**키워드 (쉼표로 구분):**
"""
    keyword_prompt = ChatPromptTemplate.from_template(keyword_prompt_template)
    
    return keyword_prompt | llm | StrOutputParser()

def get_conversational_rag_chain(retriever, system_prompt):
    """
    최종적으로 생성된 문장 단위의 출처를 사용하여 답변을 생성하는 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    rag_prompt_template = f"""{system_prompt}

You are a helpful AI assistant. Your primary goal is to provide a comprehensive and synthesized answer to the user's question based *only* on the provided "Context".
Carefully analyze the user's question to understand its core intent.
Then, thoroughly review all provided context snippets. Synthesize and connect pieces of information, even if they are from different sources, to form a complete picture.
Instead of just listing facts, explain the significance and implications of the information as it relates to the user's question. If the context describes an event, explain its impact.
Do not simply state "the information is not in the context" if a reasonable inference can be drawn from the provided text.
Answer in Korean.

**Context:**
{{context}}

**User's Request:**
{{input}}

**Answer (in Korean):**
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    def format_docs_with_metadata(docs: list[LangChainDocument]) -> str:
        """문서 리스트를 LLM 프롬프트 형식에 맞게 변환합니다."""
        if not docs:
            return "No context provided."
        
        sources = {}
        for doc in docs:
            source_url = doc.metadata.get("source", "Unknown Source")
            title = doc.metadata.get("title", "No Title")
            key = (source_url, title)
            if key not in sources:
                sources[key] = []
            sources[key].append(doc.page_content)

        formatted_string = ""
        for (source_url, title), sentences in sources.items():
            formatted_string += f"\n--- Source: {title} ({source_url}) ---\n"
            formatted_string += "\n".join(f"- {s}" for s in sentences)

        return formatted_string.strip()

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_metadata), "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def get_default_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()

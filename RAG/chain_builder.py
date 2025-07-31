from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

def get_conversational_rag_chain(retriever, system_prompt):
    """
    최종적으로 생성된 문장 단위의 출처를 사용하여 답변을 생성하는 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    rag_prompt_template = f"""{system_prompt}

Answer the user's request based *only* on the provided "Context".
If the context does not contain the answer, say you don't know.
Do not use any prior knowledge.

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

# ▼▼▼ [신규] 문서 관련성 평가용 체인 ▼▼▼
def get_retrieval_grader_chain():
    """
    검색된 문서가 질문과 관련이 있는지 평가하는 체인을 생성합니다.
    'yes' 또는 'no'의 이진 점수를 JSON 형식으로 반환합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, format="json")
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {documents} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
        input_variables=["question", "documents"],
    )
    return prompt | llm | JsonOutputParser()

# ▼▼▼ [신규] 질문 변환(재생성)용 체인 ▼▼▼
def get_query_transformer_chain():
    """
    더 나은 검색 결과를 위해 원래 질문을 변환하는 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    prompt = PromptTemplate(
        template="""You are a query transformation assistant. Your task is to rephrase the user's question to be a better, more specific search query. \n
        For example, if the user asks "Tell me about the differences between Microsoft and Google's on-device AI strategies", a good transformation would be "Microsoft's on-device AI strategy vs Google's on-device AI strategy". \n
        Original question: \n\n {question} \n\n
        Rephrased question:""",
        input_variables=["question"],
    )
    # LLM의 출력을 AIMessage 객체로 변환하여 그래프 상태와 호환되도록 합니다.
    return prompt | llm | RunnableLambda(lambda x: AIMessage(content=x.content))

# ▼▼▼ [신규] 그래프의 최종 답변 생성 노드에서 사용할 체인 ▼▼▼
def get_rag_chain_for_graph(system_prompt):
    """
    그래프용 RAG 체인을 구성합니다. 스트리밍을 지원합니다.
    입력 딕셔너리에서 올바른 값을 선택하여 처리하도록 수정되었습니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, streaming=True)
    rag_prompt_template = f"""{system_prompt}
    Answer the user's request based *only* on the provided "Context".
    If the context does not contain the answer, say you don't know.
    Do not use any prior knowledge.

    **Context:**
    {{context}}

    **User's Request:**
    {{input}}

    **Answer (in Korean):**
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    def format_docs(docs: list[LangChainDocument]) -> str:
        """문서 리스트를 하나의 문자열로 합칩니다."""
        return "\n\n".join(doc.page_content for doc in docs)

    # RunnableParallel({})을 사용하여 각 키에 올바른 입력을 명시적으로 전달합니다.
    rag_chain = (
        {
            # "context" 키에는: 입력(x)에서 "context"를 뽑아 format_docs 함수로 가공한 결과를 할당
            "context": lambda x: format_docs(x["context"]),
            # "input" 키에는: 입력(x)에서 "input"을 뽑아 그대로 할당
            "input": lambda x: x["input"],
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

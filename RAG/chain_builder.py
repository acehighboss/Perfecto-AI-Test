# RAG/chain_builder.py

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from typing import List, Dict

from langchain.text_splitter import SpacyTextSplitter


def format_docs_for_llm(docs: list[Document]) -> str:
    """LLM 프롬프트용으로 문서 내용을 포맷합니다."""
    if not docs:
        return "No context provided."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def get_conversational_rag_chain(retriever, system_prompt: str):
    """
    '다중 쿼리 생성'과 '결과 융합', 그리고 '핵심 문장 추출'을 사용하는 RAG 체인을 구성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
    
    try:
        sentence_splitter = SpacyTextSplitter(pipeline="ko_core_news_sm")
    except Exception:
        sentence_splitter = SpacyTextSplitter()


    # 1. 다중 쿼리 생성 체인
    query_generation_prompt = ChatPromptTemplate.from_template(
        '사용자의 질문을 분석하여, 검색에 효과적인 3개의 다양한 검색어를 JSON 객체 {{"queries": [...]}} 형태로 생성해주세요. '
        "질문의 핵심 키워드를 포함하되, 다른 표현이나 관점을 추가하세요.\n\n"
        "사용자 질문: {question}\n\n"
        "검색어 (JSON):"
    )
    question_to_queries_chain = query_generation_prompt | llm | JsonOutputParser()

    # 2. Retriever를 병렬로 실행하고 결과를 융합하는 함수
    def retrieve_and_fuse_results(queries_dict: Dict) -> List[Document]:
        queries = queries_dict.get("queries", [])
        if not queries: return []
        
        retrieved_docs_lists = retriever.batch(queries)
        
        unique_docs: Dict[str, Document] = {}
        for doc_list in retrieved_docs_lists:
            for doc in doc_list:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        
        return list(unique_docs.values())

    # 3. 답변 생성 RAG 체인
    rag_prompt = ChatPromptTemplate.from_template(
        f"{system_prompt}\n\n**Context:**\n{{context}}\n\n**User's Question:**\n{{question}}\n\n**Answer (in Korean):**"
    )

    # 4. 핵심 근거 문장 추출 체인
    sentence_extraction_prompt = ChatPromptTemplate.from_template(
        "You are an expert at identifying supporting evidence.\n"
        'From the provided "Context", extract the exact, complete, and verbatim sentences that directly support the "Answer" to the "User\'s Question".\n'
        "Combine all extracted sentences into a single block of text, separated by newlines.\n"
        "If no sentences directly support the answer, output an empty string.\n\n"
        "--- Context ---\n{context}\n\n"
        "--- User's Question ---\n{question}\n\n"
        "--- Answer ---\n{answer}\n\n"
        "--- Extracted Sentences ---"
    )
    sentence_extractor_chain = sentence_extraction_prompt | llm | StrOutputParser()

    # 5. 전체 파이프라인 구성
    retrieval_chain = (
        RunnableParallel({
            "generated_queries": question_to_queries_chain,
            "original_question": lambda x: x["question"]
        })
        | RunnableParallel({
            "fused_documents": lambda x: retrieve_and_fuse_results(x["generated_queries"]),
            "original_question": lambda x: x["original_question"]
        })
    )

    rag_chain_with_sources = (
        retrieval_chain
        | RunnableParallel({
            "reranked_documents": lambda x: reranker.compress_documents(
                documents=x["fused_documents"], 
                query=x["original_question"]
            ),
            "original_question": lambda x: x["original_question"]
        })
        | {
            "answer": (
                lambda x: {
                    "context": format_docs_for_llm(x["reranked_documents"]),
                    "question": x["original_question"]
                }
            ) | rag_prompt | llm | StrOutputParser(),
            "source_documents": lambda x: x["reranked_documents"],
            "original_question": lambda x: x["original_question"]
        }
    )

    def extract_and_format_final_sources(rag_output: Dict) -> Dict:
        answer = rag_output["answer"]
        source_chunks = rag_output["source_documents"]
        
        if not source_chunks:
            return {"answer": answer, "final_sources": []}

        context_str = format_docs_for_llm(source_chunks)
        extracted_sentences_str = sentence_extractor_chain.invoke({
            "context": context_str,
            "question": rag_output["original_question"],
            "answer": answer
        })

        if not extracted_sentences_str.strip():
            return {"answer": answer, "final_sources": source_chunks}

        extracted_sentences = sentence_splitter.split_text(extracted_sentences_str)
        
        final_sources = []
        for sentence in extracted_sentences:
            original_metadata = {}
            for chunk in source_chunks:
                if sentence.strip() in chunk.page_content:
                    original_metadata = chunk.metadata
                    break
            if not original_metadata and source_chunks:
                original_metadata = source_chunks[0].metadata

            final_sources.append(Document(page_content=sentence.strip(), metadata=original_metadata))
            
        return {"answer": answer, "final_sources": final_sources}

    final_chain = rag_chain_with_sources | RunnableLambda(extract_and_format_final_sources)
    
    return RunnableLambda(lambda question_str: final_chain.invoke({"question": question_str}))

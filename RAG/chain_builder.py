from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGeneraiAI
from langchain_cohere import CohereRerank
from typing import List, Dict

# ★★★ 근거 문장 추출 후 문장 단위로 분리하기 위한 도구 추가 ★★★
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
    llm = ChatGoogleGeneraiAI(model="gemini-1.5-flash", temperature=0)
    reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
    
    # spaCy 모델 로드 (문장 분리용)
    try:
        sentence_splitter = SpacyTextSplitter(pipeline="ko_core_news_sm")
    except Exception:
        # 모델 로드 실패 시 기본 분리기로 대체
        sentence_splitter = SpacyTextSplitter()


    # 1. 다중 쿼리 생성 체인
    query_generation_prompt = ChatPromptTemplate.from_template(
        '사용자의 질문을 분석하여, 검색에 효과적인 3개의 다양한 검색어를 JSON 객체 {"queries": [...]} 형태로 생성해주세요. '
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

    # 4. ★★★ 핵심 근거 문장 추출 체인 ★★★
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
    
    # 5-1. 질문 확장 및 병렬 검색
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

    # 5-2. 재정렬 및 답변 생성
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

    # 5-3. ★★★ 답변을 기반으로 최종 근거 문장 추출 및 포맷팅 ★★★
    def extract_and_format_final_sources(rag_output: Dict) -> Dict:
        answer = rag_output["answer"]
        source_chunks = rag_output["source_documents"]
        
        if not source_chunks:
            return {"answer": answer, "final_sources": []}

        # 근거 문장 추출 실행
        context_str = format_docs_for_llm(source_chunks)
        extracted_sentences_str = sentence_extractor_chain.invoke({
            "context": context_str,
            "question": rag_output["original_question"],
            "answer": answer
        })

        # 추출된 문장이 없으면, 기존 청크를 그대로 반환 (Fallback)
        if not extracted_sentences_str.strip():
            return {"answer": answer, "final_sources": source_chunks}

        # 추출된 문자열을 개별 문장으로 분리
        extracted_sentences = sentence_splitter.split_text(extracted_sentences_str)
        
        # 각 문장을 Document 객체로 변환하고 원본 메타데이터 유지
        final_sources = []
        for sentence in extracted_sentences:
            # 원본 청크에서 문장을 찾아 메타데이터를 가져옴
            original_metadata = {}
            for chunk in source_chunks:
                if sentence.strip() in chunk.page_content:
                    original_metadata = chunk.metadata
                    break
            # 문장을 찾지 못한 경우, 첫 번째 청크의 메타데이터를 기본값으로 사용
            if not original_metadata and source_chunks:
                original_metadata = source_chunks[0].metadata

            final_sources.append(Document(page_content=sentence.strip(), metadata=original_metadata))
            
        return {"answer": answer, "final_sources": final_sources}

    # 최종 체인: 답변 생성 -> 근거 문장 추출
    final_chain = rag_chain_with_sources | RunnableLambda(extract_and_format_final_sources)
    
    # chain.invoke에 question 딕셔너리가 아닌 문자열을 바로 전달할 수 있도록 래핑
    return RunnableLambda(lambda question_str: final_chain.invoke({"question": question_str}))

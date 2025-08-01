from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from .chain_builder import get_retrieval_grader_chain, get_query_transformer_chain, get_rag_chain_for_graph

# --- 그래프의 상태 정의 ---
# 그래프의 각 노드를 거치면서 이 상태 객체가 업데이트됩니다.
class GraphState(TypedDict):
    messages: List[BaseMessage] # 사용자의 질문
    documents: List[Document]   # 검색된 문서
    generation: str             # 최종 생성된 답변
    recursion_counter: int # 재시도 횟수를 기록할 변수

def get_rag_graph(retriever, system_prompt):
    """
    추가 검색(Self-Correction) 기능이 포함된 RAG 그래프를 생성하고 컴파일합니다.
    """
    # 각 노드에서 사용할 체인을 미리 빌드합니다.
    retrieval_grader = get_retrieval_grader_chain()
    query_transformer = get_query_transformer_chain()
    rag_chain = get_rag_chain_for_graph(system_prompt)

    # --- 그래프 노드 정의 (각 단계에서 수행할 작업) ---

    def retrieve(state: GraphState):
        """문서를 검색하는 노드"""
        print("---노드: 문서 검색---")
        # ▼▼▼ [수정] 첫 실행 시 카운터 초기화 ▼▼▼
        if "recursion_counter" not in state or not state["recursion_counter"]:
            state["recursion_counter"] = 0
            
        question = state["messages"][-1].content
        documents = retriever.invoke(question)
        return {"documents": documents, "messages": state["messages"]}

    def grade_documents(state: GraphState):
        """검색된 문서의 관련성을 평가하는 노드"""
        print("---노드: 문서 관련성 평가---")
        question = state["messages"][-1].content
        documents = state["documents"]
        
        # LLM을 호출하여 관련성 점수를 매깁니다.
        response = retrieval_grader.invoke({"question": question, "documents": documents})
        if response['score'] == "yes":
            print("---판단: 문서 관련성 충분함---")
            return {"documents": documents, "messages": state["messages"]}
        else:
            print("---판단: 문서 관련성 불충분함, 질문 재생성 시도---")
            # ▼▼▼ [수정] 재시도 횟수 1 증가 ▼▼▼
            state["recursion_counter"] += 1
            print(f"재시도 횟수: {state['recursion_counter']}")
            # 관련성이 없다고 판단되면, 다음 단계로 가기 위해 documents를 비웁니다.
            return {"documents": [], "messages": state["messages"]}

    def transform_query(state: GraphState):
        """사용자의 질문을 더 나은 검색어로 변환하는 노드"""
        print("---노드: 질문 변환---")
        question = state["messages"][-1].content
        
        # LLM을 호출하여 새 질문을 생성합니다.
        new_question_msg = query_transformer.invoke({"question": question})
        
        # 그래프의 상태를 새 질문으로 업데이트합니다.
        state["messages"].append(new_question_msg)
        print(f"새로운 검색어: {new_question_msg.content}")
        return {"documents": [], "messages": state["messages"]}
    
    def generate(state: GraphState):
        """최종 답변을 생성하는 노드"""
        print("---노드: 답변 생성---")
        question = state["messages"][-1].content
        documents = state["documents"]
        # ▼▼▼ [수정] 문서가 없을 경우, "정보를 찾을 수 없다"는 답변을 직접 생성 ▼▼▼
        if not documents:
            generation = "죄송합니다, 여러 번 시도했지만 질문에 답변하는 데 필요한 관련 정보를 찾을 수 없었습니다."
        else:
            generation = rag_chain.invoke({"context": documents, "input": question})
        return {"documents": documents, "messages": state["messages"], "generation": generation}

    # --- 그래프 엣지(흐름) 정의 ---

    def decide_to_generate(state: GraphState):
        """
        답변을 생성할지, 아니면 질문을 다시 만들어 검색할지 결정하는 조건부 엣지
        """
        print("---엣지: 답변 생성 또는 추가 검색 결정---")
        # ▼▼▼ [수정] 재시도 횟수 제한 로직 추가 ▼▼▼
        if not state["documents"]:
            if state["recursion_counter"] >= 3: # 3번 이상 시도했으면
                print("---판단: 재시도 횟수 초과, 답변 생성으로 이동---")
                return "generate" # 포기하고 답변 생성 노드로 이동
            else:
                print("---판단: 추가 검색 시도---")
                return "transform_query" # 재시도
        else:
            return "generate" # 성공 시 답변 생성

    # --- 그래프 구성 ---
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    
    # 엣지 연결
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "retrieve", "generate": "generate"}, # 조건에 따라 분기
    )
    workflow.add_edge("generate", END) # 답변 생성이 끝나면 워크플로우 종료

    # 그래프 컴파일
    app = workflow.compile()
    print("RAG 그래프가 성공적으로 컴파일되었습니다.")
    return app

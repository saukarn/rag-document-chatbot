from typing import Any, List, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from backend.config import LLM_MODEL, COHERE_API_KEY, RERANK_MODEL
from backend.vector_store import get_retriever


class RAGState(TypedDict):
    question: str
    documents: List[Any]
    answer: str
    grounded: bool
    attempts: int


def retrieve_and_rerank_node(state: RAGState):
    base_retriever = get_retriever(k=12)

    reranker = CohereRerank(
        model=RERANK_MODEL,
        cohere_api_key=COHERE_API_KEY,
        top_n=4
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )

    docs = compression_retriever.invoke(state["question"])

    print(f"DEBUG: Retrieved and reranked {len(docs)} docs")

    return {
        "documents": docs
    }


def generate_answer_node(state: RAGState):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{doc.page_content}"
            for i, doc in enumerate(state["documents"])
        ]
    )

    system_prompt = """
                    You are a careful RAG assistant.

                    Answer using only the provided context.
                    If the answer is not present in the context, say:
                    "I don't know based on the uploaded documents."

                    Rules:
                    - Do not invent facts.
                    - Do not use outside knowledge.
                    - Cite source numbers when useful.
                    - Keep the answer clear and concise.
                    """

    user_prompt = f"""
                Context:
                {context}

                Question:
                {state["question"]}
                """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "answer": response.content,
        "attempts": state["attempts"] + 1
    }


def grade_answer_node(state: RAGState):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    context = "\n\n".join(
        [doc.page_content for doc in state["documents"]]
    )

    prompt = f"""
            You are a strict RAG evaluator.

            Decide whether the answer is fully supported by the provided context.

            Return only one word:
            SUPPORTED
            or
            UNSUPPORTED

            Context:
            {context}

            Answer:
            {state["answer"]}
            """

    response = llm.invoke([HumanMessage(content=prompt)])

    verdict = response.content.strip().upper()

    print("DEBUG: Grounding verdict:", verdict)

    return {
        "grounded": verdict == "SUPPORTED"
    }


def fallback_node(state: RAGState):
    return {
        "answer": (
            "I don't know based on the uploaded documents. "
            "The retrieved context was not strong enough to produce a grounded answer."
        )
    }


def route_after_grading(state: RAGState) -> Literal["end", "retry", "fallback"]:
    if state["grounded"]:
        return "end"

    if state["attempts"] < 2:
        return "retry"

    return "fallback"


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve_and_rerank")

    graph.add_edge("retrieve_and_rerank", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")

    graph.add_conditional_edges(
        "grade_answer",
        route_after_grading,
        {
            "end": END,
            "retry": "retrieve_and_rerank",
            "fallback": "fallback"
        }
    )

    graph.add_edge("fallback", END)

    return graph.compile()


rag_app = build_rag_graph()


def answer_question(question: str):
    result = rag_app.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "grounded": False,
        "attempts": 0
    })

    sources = []

    for doc in result["documents"]:
        sources.append({
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content[:500]
        })

    return {
        "answer": result["answer"],
        "sources": sources
    }
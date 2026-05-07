from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import LLM_MODEL
from backend.vector_store import get_retriever


class RAGState(TypedDict):
    question: str
    documents: List[Any]
    answer: str


def retrieve_node(state: RAGState):
    retriever = get_retriever(k=4)
    docs = retriever.invoke(state["question"])

    return {
        "documents": docs
    }


def generate_node(state: RAGState):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    context = "\n\n".join(
        [doc.page_content for doc in state["documents"]]
    )

    system_prompt = """
You are a helpful RAG assistant.
Answer the user's question using only the provided context.
If the answer is not in the context, say:
"I don't know based on the uploaded documents."
Do not make up facts.
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
        "answer": response.content
    }


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_app = build_rag_graph()


def answer_question(question: str):
    result = rag_app.invoke({
        "question": question,
        "documents": [],
        "answer": ""
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
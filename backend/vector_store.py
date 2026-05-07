from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config import (
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
)


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store():
    embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )


def add_documents_to_vector_store(chunks):
    print(f"DEBUG: Adding {len(chunks)} chunks to Pinecone")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print("DEBUG: Successfully added chunks")

    return len(chunks)


def get_retriever(k: int = 8):
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
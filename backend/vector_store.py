from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from backend.config import CHROMA_DIR, EMBEDDING_MODEL


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def add_documents_to_vector_store(chunks):
    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    vector_store.add_documents(chunks)

    return len(chunks)


def get_retriever(k: int = 4):
    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
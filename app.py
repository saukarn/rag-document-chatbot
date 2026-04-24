import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚")
st.title("📚 PDF RAG Chatbot")
st.write("Upload PDFs and ask questions from them.")

CHROMA_DIR = "data/chroma_db"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None


def load_pdfs(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())

        os.remove(temp_path)

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def create_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Reading, chunking, and embedding documents..."):
            docs = load_pdfs(uploaded_files)
            chunks = split_docs(docs)
            st.session_state.retriever = create_vector_store(chunks)

        st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(chunks)} chunks.")

question = st.chat_input("Ask a question about your documents")

if question:
    if st.session_state.retriever is None:
        st.warning("Upload and process documents first.")
    else:
        st.chat_message("user").write(question)

        chain = create_chain(st.session_state.retriever)

        result = chain.invoke({"question": question})
        answer = result["answer"]

        st.chat_message("assistant").write(answer)

        with st.expander("Sources"):
            for doc in result["source_documents"]:
                st.write(doc.metadata)
                st.write(doc.page_content[:500])
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def save_uploaded_file(file, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_path


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def ingest_pdf(file, upload_dir: str):
    file_path = save_uploaded_file(file, upload_dir)
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    return chunks
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

print("DEBUG ENV PATH:", ENV_PATH)
print("DEBUG ENV EXISTS:", os.path.exists(ENV_PATH))

load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-production-engine")

print("DEBUG PINECONE LOADED:", bool(PINECONE_API_KEY))

UPLOAD_DIR = "data/uploads"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
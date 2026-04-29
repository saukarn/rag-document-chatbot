import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "data/chroma_db"

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
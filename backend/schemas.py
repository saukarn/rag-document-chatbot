from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str


class SourceDocument(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
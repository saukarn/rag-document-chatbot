# 📚 RAG Document Intelligence Platform

A **production-grade Retrieval-Augmented Generation (RAG) system** for intelligent querying over PDF documents, built with modern LLM orchestration and scalable vector search.

---

## 🚀 Overview

This project enables users to:

- Upload PDF documents
- Perform semantic search over content
- Ask questions and receive **grounded, source-backed answers**

The system goes beyond basic RAG by implementing:

- Two-stage retrieval (vector search + reranking)
- Hallucination mitigation using LLM-based answer grading
- Retry and fallback logic using LangGraph
- Full observability using LangSmith

---

## 🏗️ Architecture

```
Streamlit UI
   ↓
FastAPI Backend
   ↓
LangGraph RAG Workflow
   ↓
Pinecone (Vector DB)
   ↓
OpenAI (LLM + Embeddings)
   ↓
Cohere (Reranking)
```

---

## ⚙️ Tech Stack

- **LLM & Embeddings:** OpenAI  
- **Orchestration:** LangGraph  
- **Backend API:** FastAPI  
- **Vector Database:** Pinecone  
- **Reranking:** Cohere  
- **Frontend:** Streamlit  
- **Observability:** LangSmith  
- **Containerization:** Docker  

---

## 🔑 Key Features

### 📄 Document Ingestion
- PDF parsing and processing
- Semantic chunking with optimized chunk size and overlap
- Metadata enrichment for traceability

### 🔍 Advanced Retrieval
- Vector similarity search using Pinecone
- Top-K retrieval followed by **Cohere reranking**
- Reduced noise and improved context quality

### 🧠 Intelligent Answer Generation
- Context-aware LLM responses using OpenAI
- Strict grounding rules to avoid hallucinations
- Source-backed responses with traceable context

### 🛡️ Reliability & Guardrails
- LLM-based answer grading (SUPPORTED / UNSUPPORTED)
- Retry mechanism for improved retrieval
- Fallback response when context is insufficient

### 📊 Observability
- End-to-end tracing using LangSmith
- Visibility into:
  - Retrieval results
  - LLM prompts and responses
  - Answer grading decisions

---

## 🔄 RAG Workflow

```
User Question
   ↓
Retrieve (Pinecone Top-K)
   ↓
Rerank (Cohere → Top-N)
   ↓
Generate Answer (OpenAI)
   ↓
Grade Answer (LLM)
   ↓
   ├── Supported → Return Answer
   ├── Not Supported → Retry
   └── Retry Failed → Fallback Response
```

---

## 🧪 Example Use Cases

- Enterprise document search
- Knowledge base assistants
- Contract / policy analysis
- Internal support chatbots

---

## 🛠️ Run Locally

### 1. Clone repo

```bash
git clone https://github.com/saukarn/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Setup environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
COHERE_API_KEY=

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=rag-production-engine
```

### 4. Run backend

```bash
uvicorn backend.main:app --reload
```

Swagger UI:
http://localhost:8000/docs

### 5. Run frontend

```bash
streamlit run frontend/streamlit_app.py
```

---

## 📦 API Endpoints

| Endpoint   | Description                 |
|------------|-----------------------------|
| /upload    | Upload and index PDF        |
| /chat      | Ask questions               |
| /health    | Health check                |

---

## 📊 Observability (LangSmith)

To view execution traces:

1. Go to: https://smith.langchain.com  
2. Open project: `rag-production-engine`  
3. Inspect:
   - Retrieval results  
   - Reranking output  
   - LLM prompts  
   - Answer grading decisions  

---

## 🚧 Future Enhancements

- Caching layer (Redis / DynamoDB)
- Evaluation metrics (RAGAS / DeepEval)
- Authentication (JWT / Cognito)
- Multi-document context isolation
- AWS deployment (ECS, S3, API Gateway)

---

## 💡 What makes this project stand out

It demonstrates:

✔ Production-grade RAG architecture  
✔ Multi-step LLM orchestration (LangGraph)  
✔ Retrieval quality optimization (reranking)  
✔ Hallucination mitigation strategies  
✔ Backend + AI system design integration  
✔ Observability and debugging practices  

---


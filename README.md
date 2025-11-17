# 🚀 AI Agent with Web Search (RAG + LangGraph + Groq)

This project is an AI-powered agent that answers questions from **uploaded PDFs** and automatically switches to **web search** when the answer isn't found in the document.  
It uses Retrieval-Augmented Generation (RAG), LangGraph for workflow orchestration, and ultra-fast inference via **Groq API**.

---

## ✨ Features

- 📄 **PDF Upload + Text Extraction**  
  Upload a PDF and query its content using a vector-based retrieval system.

- 🔍 **RAG-Based Document Q&A**  
  Generates dense embeddings and retrieves the most relevant text chunks.

- 🌐 **Fallback Web Search**  
  If the PDF doesn't contain the answer, the agent triggers a web search tool automatically.

- 🧠 **LangGraph Workflow**  
  A step-by-step agent graph that decides when to use RAG vs. when to use web search.

- ⚡ **Groq LLMs**  
  Runs extremely fast inference using Groq's hosted Llama/Mixtral models.

- 🖥️ **Two Interfaces**  
  - **FastAPI backend** (API for chat + PDF upload)  
  - **Streamlit UI** (user-friendly chat interface)

- 🧰 **uv Package Manager**  
  Used for fast dependency installation and environment management.

---

## 🏗️ Tech Stack

- **Core:** Python, LangChain, LangGraph, Groq API  
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **RAG:** Embeddings + Vector Search  
- **Environment:** uv package manager

---


---

## ⚙️ Installation (using uv)

```bash
# Clone the repository
git clone https://github.com/Md-Tauhid101/AI-Agent-with-Web-Search.git

# Install dependencies using uv
uv sync
```

- If you're not using uv:
```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables
- Create a `.env` file in the root:
```bash
GROQ_API_KEY=your_groq_api_key
```

## 🧠 How the Agent Works
1. User uploads a PDF.
2. Text is extracted, chunked, embedded, and stored in a vector database.
3. The user asks a question.
4. LangGraph agent decides:
    - Retrieve answer from PDF using embeddings, or
    - Trigger web search when the answer is missing.
5. Groq LLM synthesizes a final answer.
6. Result returned to UI or API.

## 🚧 Future Improvements
- Multi-PDF support
- Conversation memory
- Vector database optimization
- Support for multiple LLM providers

import os
from dotenv import load_dotenv

load_dotenv()

# vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

#groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Paths (adjust as needed)
DOC_SOURCE_DIR = os.getenv("DOC_SOURCE_DIR", "data")

HUGGINGFACE_EMBEDDINGS = os.getenv("HUGGINGFACEHUB_API_TOKEN")
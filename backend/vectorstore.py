import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PINECONE_API_KEY, HUGGINGFACE_EMBEDDINGS, PINECONE_INDEX_NAME

# Load .env if not already done
load_dotenv()

# Validate env variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in config.py or environment")

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)
INDEX_NAME = PINECONE_INDEX_NAME

# Define embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_retriever():
    """Initializes and returns the Pinecone vector store retriever"""
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print("Creating new index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Created Pinecone index.")

    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vector_store.as_retriever()

def add_document_to_vectorstore(text_content: str):
    """Adds a text document to the Pinecone vector store."""
    if not text_content:
        raise ValueError("Document content not provided")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    documents = text_splitter.create_documents([text_content])
    print("Splitting document into chunks for indexing...")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    vectorstore.add_documents(documents)
    print("Successfully added chunks to Pinecone vectorstore.")

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PINECONE_API_KEY

# set env var for pinecone
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
INDEX_NAME = os.environ['PINECONE_INDEX_NAME']

# initialize pinecone for pinecone
pc = Pinecone(api_key = PINECONE_API_KEY)
# define embedding models
embeddings = HuggingFaceEmbeddings(model_name = "sentance-transformers/all-MiniLM-L6-v2")


# retriever function
def get_retriever():
    """
        Initializes and return the pinecone vector store retriever
    """
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new inddex")
        pc.create_index(
            name = INDEX_NAME,
            dimension = 384,
            metric = 'cosine',
            spec = ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print("Created pinecone index")
    
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embeddings = embeddings)
    return vector_store.as_retriever()

# upload doucuments to vector DB
def add_document_to_vectorstore(text_content:str):
    """
        Adds a single text document to the Pinecone vector store.
        Splits the text into chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content not found")
    
    # Create document chunks
    text_spitter = RecursiveCharacterTextSplitter(
        chunk = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )

    # create langchain document objects from the raw text
    documents = text_spitter.create_documents([text_content])
    print("Splitting document into chunk for indexing...")

    # create vector store instance to add documents
    vectorstore = PineconeVectorStore(index_name = INDEX_NAME, embeddings = embeddings)

    # Add documents to vectorstore
    vectorstore.add_documents(documents)
    print("Successfully added chunks to pinecone vectorstore")
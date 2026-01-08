"""
Vector store module
Handles FAISS vector database operations using Google Generative AI embeddings
"""

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def create_vector_store(chunks, embeddings):
    """
    Create FAISS vector store from text chunks.
    
    Args:
        chunks: List of text chunks (strings or Document objects)
        embeddings: Embedding model (GoogleGenerativeAIEmbeddings)
        
    Returns:
        FAISS: Vector store instance
    """
    # Convert strings to Document objects if needed
    if chunks and isinstance(chunks[0], str):
        documents = [Document(page_content=chunk) for chunk in chunks]
    else:
        documents = chunks
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def search_similar(query: str, vector_store: FAISS, k: int = 5):
    """
    Search for similar chunks in vector store.
    
    Args:
        query: Search query string
        vector_store: FAISS vector store
        k: Number of results to return
        
    Returns:
        list: List of Document objects with similar content
    """
    results = vector_store.similarity_search(query, k=k)
    return results


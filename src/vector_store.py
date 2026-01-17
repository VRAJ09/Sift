"""
Vector store module
Handles FAISS vector database operations using Google Generative AI embeddings
"""

import time
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_vector_store(chunks, embeddings, progress_callback=None, batch_size=10, delay_between_batches=5):
    """
    Create FAISS vector store from text chunks with rate limiting and batched processing.
    
    Args:
        chunks: List of text chunks (strings or Document objects)
        embeddings: Embedding model (GoogleGenerativeAIEmbeddings)
        progress_callback: Optional callback function(status, progress) for progress updates
        batch_size: Number of chunks to embed per batch (default: 10, safe for free tier)
        delay_between_batches: Seconds to wait between batches (default: 5)
        
    Returns:
        FAISS: Vector store instance
    """
    # Convert strings to Document objects if needed
    if chunks and isinstance(chunks[0], str):
        documents = [Document(page_content=chunk) for chunk in chunks]
    else:
        documents = chunks
    
    num_docs = len(documents)
    
    if progress_callback:
        progress_callback(f"Processing {num_docs} chunks in batches of {batch_size}...", 0)
    
    # Process in batches to avoid rate limits
    num_batches = (num_docs + batch_size - 1) // batch_size  # Ceiling division
    vector_store = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_docs)
        batch_docs = documents[start_idx:end_idx]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_metadatas = [doc.metadata for doc in batch_docs]
        
        if progress_callback:
            progress = int((batch_idx / num_batches) * 90)  # 0-90% for batches
            progress_callback(
                f"Embedding batch {batch_idx + 1}/{num_batches} ({start_idx + 1}-{end_idx} of {num_docs})...",
                progress
            )
        
        # Retry logic for each batch
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Create or add to vector store incrementally
                if vector_store is None:
                    # Create initial vector store with first batch
                    vector_store = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
                else:
                    # Add subsequent batches (will embed automatically)
                    vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "429" in error_str or 
                    "RESOURCE_EXHAUSTED" in error_str or 
                    "quota" in error_str.lower() or
                    error_type == "GoogleGenerativeAIError" and ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str)
                )
                
                if is_rate_limit:
                    # Extract retry delay from error message if available
                    if "retry in" in error_str.lower():
                        try:
                            import re
                            match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                            if match:
                                retry_delay = max(float(match.group(1)), 5)
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        if progress_callback:
                            progress_callback(
                                f"⏳ Rate limit hit on batch {batch_idx + 1}. Waiting {int(wait_time)}s...",
                                0
                            )
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed, raise the error
                        raise
                else:
                    # Not a rate limit error, raise immediately
                    raise
        
        # Wait between batches to respect rate limits (except after last batch)
        if batch_idx < num_batches - 1:
            if progress_callback:
                progress_callback(f"Waiting {delay_between_batches}s before next batch...", 0)
            time.sleep(delay_between_batches)
    
    if progress_callback:
        progress_callback("Vector store created successfully!", 100)
    
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


"""
Vector store module
Handles FAISS vector database operations using Google Generative AI embeddings
"""

import time
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_vector_store(chunks, embeddings, progress_callback=None, batch_size=100, delay_between_batches=1):
    """
    Create FAISS vector store from text chunks with optimized batch embedding.
    
    BATCHES MULTIPLE CHUNKS INTO SINGLE API CALL to avoid rate limits.
    For example: 100 chunks in 1 API call instead of 100 separate calls.
    
    Args:
        chunks: List of text chunks (strings or Document objects)
        embeddings: Embedding model (GoogleGenerativeAIEmbeddings)
        progress_callback: Optional callback function(status, progress) for progress updates
        batch_size: Number of chunks to embed per API call (default: 100, much more efficient)
        delay_between_batches: Seconds to wait between batches (default: 1, less needed with batching)
        
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
        progress_callback(f"Processing {num_docs} chunks in batches of {batch_size} (optimized batching)...", 0)
    
    # Extract all texts and metadatas upfront
    all_texts = [doc.page_content for doc in documents]
    all_metadatas = [doc.metadata for doc in documents]
    
    # Batch embeddings: send multiple chunks in single API call
    num_batches = (num_docs + batch_size - 1) // batch_size  # Ceiling division
    all_embeddings = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_docs)
        batch_texts = all_texts[start_idx:end_idx]
        
        if progress_callback:
            progress = int((batch_idx / num_batches) * 90)  # 0-90% for batches
            progress_callback(
                f"Embedding batch {batch_idx + 1}/{num_batches} ({start_idx + 1}-{end_idx} of {num_docs}) in single API call...",
                progress
            )
        
        # Retry logic for each batch
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # THIS IS THE KEY: Batch multiple chunks into ONE API call
                batch_embeddings = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
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
        
        # Wait between batches (less needed with larger batches)
        if batch_idx < num_batches - 1:
            if delay_between_batches > 0:
                if progress_callback:
                    progress_callback(f"Waiting {delay_between_batches}s before next batch...", 0)
                time.sleep(delay_between_batches)
    
    # Create FAISS vector store from all pre-computed embeddings
    if progress_callback:
        progress_callback("Creating vector store from embeddings...", 95)
    
    # Use FAISS.from_embeddings if available, otherwise create manually
    try:
        # Try using from_embeddings method (if available in newer LangChain)
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(all_texts, all_embeddings)),
            embedding=embeddings,
            metadatas=all_metadatas if all_metadatas[0] else None
        )
    except (AttributeError, TypeError):
        # Fallback: Create manually using FAISS internals
        import numpy as np
        from faiss import IndexFlatL2
        
        # Get embedding dimension
        embedding_dim = len(all_embeddings[0])
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index
        index = IndexFlatL2(embedding_dim)
        index.add(embeddings_array)
        
        # Create vector store with a dummy embedding to initialize structure
        vector_store = FAISS.from_texts([all_texts[0]], embeddings)
        
        # Replace index with our pre-computed one
        vector_store.index = index
        
        # Add all documents to docstore
        from langchain_core.documents import Document as LC_Document
        for i, (text, metadata) in enumerate(zip(all_texts, all_metadatas)):
            doc = LC_Document(page_content=text, metadata=metadata if metadata else {})
            vector_store.docstore.add({i: doc})
            vector_store.index_to_docstore_id[i] = i
    
    if progress_callback:
        progress_callback(f"✅ Vector store created! ({num_batches} API calls for {num_docs} chunks)", 100)
    
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


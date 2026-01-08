"""
RAG chain module
Implements the Retrieval-Augmented Generation pipeline using Google Gemini
"""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def create_rag_chain(vector_store, api_key: str, model_name: str = "gemini-3.0-flash-preview"):
    """
    Create LangChain RAG chain using Google Gemini.
    
    Args:
        vector_store: FAISS vector store (retriever)
        api_key: Google Gemini API key
        model_name: Model name (default: gemini-3-flash-preview)
                    Options: gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash
        
    Returns:
        RAG chain
    """
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Create custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def query_rag(question: str, rag_chain):
    """
    Query the RAG chain with a question.
    
    Args:
        question: User question
        rag_chain: RAG chain
        
    Returns:
        dict: Contains 'result' (answer) and 'source_documents'
    """
    result = rag_chain.invoke({"query": question})
    return result

def get_embeddings(api_key: str):
    """
    Get Google Generative AI embeddings model.
    
    Args:
        api_key: Google Gemini API key
        
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )


"""
Sift - RAG-Based Document Assistant
Main Streamlit application
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Sift - Document Assistant",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 Sift - Document Assistant")
    st.markdown("Upload a PDF document and ask questions about its content using AI-powered RAG.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input (or use .env)
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Google Gemini API key or set it in .env file. Get it from https://makersuite.google.com/app/apikey"
        )
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            help="gemini-2.0-flash-exp (experimental, fastest), gemini-1.5-pro (most capable), gemini-1.5-flash (balanced)"
        )
        
        # Chunk size
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for embedding"
        )
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload Document", "💬 Chat"])
    
    with tab1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # TODO: Process PDF and create vector store
            # This will be implemented in feature/vector-store branch
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Placeholder for document processing
                    st.info("Document processing will be implemented in Phase 2")
    
    with tab2:
        st.header("Chat with Document")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # TODO: Generate response using RAG
            # This will be implemented in feature/rag-pipeline branch
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = "RAG pipeline will be implemented in Phase 3. Your question: " + prompt
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


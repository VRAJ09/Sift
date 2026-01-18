"""
Sift - RAG-Based Document Assistant
Main Streamlit application
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Import modules
from src.document_processor import parse_pdf_from_streamlit, chunk_text
from src.vector_store import create_vector_store
from src.rag_chain import get_embeddings, query_document_simple
from src.utils import validate_pdf

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
        
        # Processing mode
        processing_mode = st.selectbox(
            "Processing Mode",
            ["Simple (No Embeddings)", "RAG (With Embeddings)"],
            index=0,
            help="Simple: Sends full document with each question (no embedding API calls, limited by context size). RAG: Uses embeddings for semantic search (more API calls but handles large documents)."
        )
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"],
            index=0,
            help="gemini-2.5-flash (Fast, large context), gemini-2.5-pro (Multipurpose), gemini-3-flash-preview (Superior)"
        )
        
        # Chunk size (only for RAG mode)
        if processing_mode == "RAG (With Embeddings)":
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for embedding"
            )
        else:
            chunk_size = None
    
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
            
            # Check if API key is available
            if not api_key:
                st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to process documents.")
            
            # Process Document button
            if st.button("Process Document", disabled=not api_key):
                try:
                    # Initialize variables for error handling
                    num_chunks = 0
                    text_length = 0
                    
                    with st.spinner("Processing document..."):
                        # Validate PDF
                        if not validate_pdf(uploaded_file):
                            st.error("❌ Invalid PDF file. Please upload a valid PDF.")
                            return
                        
                        # Step 1: Parse PDF
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("📖 Extracting text from PDF...")
                        progress_bar.progress(20)
                        
                        full_text = parse_pdf_from_streamlit(uploaded_file)
                        text_length = len(full_text)
                        
                        if text_length == 0:
                            st.error("❌ No text could be extracted from the PDF. It might be image-based or corrupted.")
                            return
                        
                        # Check context limits for Simple mode (~1M tokens = ~4M characters for gemini-2.5-flash)
                        max_context_chars = 3_500_000  # Conservative limit
                        
                        if processing_mode == "Simple (No Embeddings)":
                            # Simple mode: Just store the text, no embeddings
                            if text_length > max_context_chars:
                                st.warning(f"""
                                ⚠️ **Document too large for Simple Mode**
                                
                                Your document has **{text_length:,} characters** which exceeds the context limit (~{max_context_chars:,} chars).
                                
                                **Solutions:**
                                1. Switch to **RAG (With Embeddings)** mode for large documents
                                2. Use a smaller document
                                
                                The document will still be processed, but may be truncated when answering questions.
                                """)
                            
                            # Store full text (no embeddings needed)
                            st.session_state.document_text = full_text
                            st.session_state.document_processed = True
                            st.session_state.processing_mode = "simple"
                            st.session_state.processed_file_name = uploaded_file.name
                            st.session_state.text_length = text_length
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Document processed successfully!")
                            
                            st.success("✅ Document processed successfully!")
                            st.info(f"""
                            **Processing Summary (Simple Mode):**
                            - 📄 File: {uploaded_file.name}
                            - 📝 Text length: {text_length:,} characters
                            - ⚡ **No embedding API calls** - avoids rate limits!
                            - 📤 Full document will be sent with each question
                            
                            You can now switch to the **Chat** tab to ask questions!
                            """)
                        
                        else:
                            # RAG mode: Chunk and create embeddings
                            # Step 2: Chunk text
                            status_text.text("✂️ Chunking text for embeddings...")
                            progress_bar.progress(40)
                            
                            chunk_overlap = min(200, chunk_size // 5)  # 20% overlap, max 200
                            chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            num_chunks = len(chunks)
                            
                            # Step 3: Create embeddings
                            status_text.text("🔢 Creating embeddings...")
                            progress_bar.progress(60)
                            
                            embeddings = get_embeddings(api_key)
                            
                            # Step 4: Create vector store with progress callback
                            def update_progress(message, progress):
                                status_text.text(message)
                                if progress > 0:
                                    progress_bar.progress(60 + int(progress * 0.3))  # 60-90% range
                            
                            status_text.text("💾 Creating vector store...")
                            progress_bar.progress(60)
                            
                            # Calculate expected API calls for user info
                            num_chunks = len(chunks)
                            batch_size = 100  # Default batch size
                            expected_api_calls = (num_chunks + batch_size - 1) // batch_size
                            
                            # Show diagnostics
                            if num_chunks > 0:
                                st.info(f"📊 **Diagnostics:** {num_chunks} chunks → ~{expected_api_calls} embedding API call(s) (batched)")
                            
                            vector_store = create_vector_store(chunks, embeddings, progress_callback=update_progress)
                            
                            # Step 5: Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.document_processed = True
                            st.session_state.processing_mode = "rag"
                            st.session_state.processed_file_name = uploaded_file.name
                            st.session_state.num_chunks = num_chunks
                            st.session_state.text_length = text_length
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Document processed successfully!")
                            
                            # Display processing results
                            st.success("✅ Document processed successfully!")
                            st.info(f"""
                            **Processing Summary (RAG Mode):**
                            - 📄 File: {uploaded_file.name}
                            - 📝 Text length: {text_length:,} characters
                            - ✂️ Number of chunks: {num_chunks}
                            - 📏 Chunk size: {chunk_size} characters
                            - 🔗 Chunk overlap: {chunk_overlap} characters
                            
                            You can now switch to the **Chat** tab to ask questions about your document!
                            """)
                        
                except ValueError as e:
                    st.error(f"❌ Validation error: {str(e)}")
                except Exception as e:
                    error_str = str(e)
                    
                    # Variables are initialized at start of try block
                    
                    # Check for rate limit errors
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        # Extract retry delay if mentioned
                        retry_info = ""
                        if "retry in" in error_str.lower():
                            try:
                                import re
                                match = re.search(r'retry in ([\d.]+)s', error_str.lower())
                                if match:
                                    wait_time = int(float(match.group(1)))
                                    retry_info = f"\n\n⏰ **Please wait {wait_time} seconds** before trying again."
                            except:
                                pass
                        
                        # Calculate expected API calls (batched)
                        batch_size = 100
                        expected_calls = (num_chunks + batch_size - 1) // batch_size if num_chunks > 0 else 0
                        
                        st.error(f"""
                        ❌ **Rate Limit Exceeded (429 Error)**
                        
                        You've exceeded Google Gemini API's free tier quota for embeddings.
                        
                        **Diagnostics:**
                        - Document: {text_length:,} characters
                        - Chunks created: {num_chunks}
                        - Expected API calls: **{expected_calls}** (batched, chunks processed together)
                        - Status: Quota limit reached ({retry_info if retry_info else "Daily/minute limit exceeded"})
                        
                        **Why this happens:**
                        Even with batching, you've likely:
                        - Hit the **daily quota** (~1,500 embedding requests/day) from previous testing
                        - OR exceeded **requests per minute** limit (~15/min)
                        - Your API key may need quota reset
                        
                        **Solutions (in order):**
                        1. **Wait 5-60 seconds** and retry (minute-based limit)
                        2. **Wait several hours** for daily quota to reset (usually resets at midnight UTC)
                        3. **Use Simple Mode** (no embeddings) - Perfect for small documents like yours!
                        4. **Check your quota** at https://ai.dev/rate-limit
                        5. **Upgrade API plan** at https://ai.google.dev/pricing for higher limits
                        
                        **Free Tier Limits:**
                        - ~1,500 embedding requests per day
                        - ~15 requests per minute
                        """)
                        
                        st.info("💡 **Tip:** Increase the 'Chunk Size' slider in the sidebar to reduce the number of embedding requests needed.")
                    else:
                        st.error(f"❌ Error processing document: {error_str}")
                        with st.expander("Show detailed error"):
                            st.exception(e)
            
            # Show status if document is already processed
            if st.session_state.get("document_processed", False):
                if st.session_state.get("processed_file_name") == uploaded_file.name:
                    st.info(f"""
                    ✅ **Document already processed:** {uploaded_file.name}
                    - 📝 {st.session_state.get('num_chunks', 0)} chunks ready
                    - 💬 Switch to the **Chat** tab to ask questions!
                    """)
                else:
                    # Different file uploaded, reset state
                    st.session_state.document_processed = False
                    st.session_state.vector_store = None
    
    with tab2:
        st.header("Chat with Document")
        
        # Check if document is processed
        if not st.session_state.get("document_processed", False):
            st.info("📄 Please upload and process a document in the **Upload Document** tab first.")
        else:
            processing_mode = st.session_state.get("processing_mode", "rag")
            
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
                
                # Generate response based on mode
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            if processing_mode == "simple":
                                # Simple mode: Send full document text
                                document_text = st.session_state.get("document_text", "")
                                response = query_document_simple(prompt, document_text, api_key, model)
                            else:
                                # RAG mode: Use vector store (Phase 3 - placeholder for now)
                                response = "RAG pipeline will be implemented in Phase 3. Your question: " + prompt
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()


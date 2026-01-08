# Sift - RAG-Based Document Assistant

A modern Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with an AI assistant about the content.

## Features

- PDF document upload and parsing
- Semantic search using vector embeddings
- Interactive chat interface
- RAG-powered question answering
- Easy deployment to Streamlit Cloud

## Tech Stack

- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: FAISS (local) or Pinecone (cloud)
- **LLM**: Google Gemini API (gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash) - **Free!**
- **Embeddings**: Google Generative AI (embedding-001)
- **PDF Processing**: PyPDF2/pdfplumber

## Prerequisites

- Python 3.8+
- Google Gemini API key (free at https://makersuite.google.com/app/apikey)

## 🔧 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Sift

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your free API key from: https://makersuite.google.com/app/apikey

### Running Locally

```bash
streamlit run app.py
```

## Deployment

Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add environment variables in Streamlit Cloud settings
4. Deploy!

## Project Structure

```
sift/
├── app.py                 # Main Streamlit application
├── src/
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── utils.py
├── requirements.txt
├── .env
└── README.md
```

## Contributing

See `PROJECT_PLAN.md` for development workflow and branch strategy.

## License

MIT


# Sift - RAG-Based Document Assistant

A modern Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with an AI assistant about the content.

## 🚀 Features

- 📄 PDF document upload and parsing
- 🔍 Semantic search using vector embeddings
- 💬 Interactive chat interface
- 🧠 RAG-powered question answering
- ☁️ Easy deployment to Streamlit Cloud

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: FAISS (local) or Pinecone (cloud)
- **LLM**: OpenAI API (GPT-4/GPT-3.5-turbo)
- **PDF Processing**: PyPDF2/pdfplumber

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key (or HuggingFace for local models)

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

## ⚙️ Configuration

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your_api_key_here
```

## 🏃 Running Locally

```bash
streamlit run app.py
```

## 📦 Deployment

Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add environment variables in Streamlit Cloud settings
4. Deploy!

## 📁 Project Structure

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

## 🤝 Contributing

See `PROJECT_PLAN.md` for development workflow and branch strategy.

## 📝 License

MIT


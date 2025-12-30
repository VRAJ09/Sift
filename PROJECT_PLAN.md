# RAG-Based Document Assistant - Project Plan

## Deployment Recommendation: **Streamlit** ✅

**Why Streamlit over Vercel:**
- **Built for ML/AI**: Streamlit is specifically designed for data science and ML applications
- **Faster development**: File upload, chat interface, and deployment are built-in
- **Free hosting**: Streamlit Cloud offers free deployment (perfect for portfolio)
- **Simpler architecture**: Single app vs. separate frontend/backend
- **Better for interviews**: Shows you can build end-to-end ML applications quickly

**When to use Vercel:**
- If you want a custom React/Next.js frontend
- If you need more complex UI/UX
- If you're building a production SaaS product

**For this project**: Start with Streamlit, then optionally add a separate frontend later if needed.

---

## Project Architecture

### Option 1: Streamlit Monolith (Recommended for MVP)
```
sift/
├── app.py                 # Main Streamlit app
├── src/
│   ├── document_processor.py  # PDF parsing & chunking
│   ├── vector_store.py        # FAISS/Pinecone integration
│   ├── rag_chain.py           # LangChain RAG pipeline
│   └── utils.py               # Helper functions
├── requirements.txt
└── README.md
```

### Option 2: Separate Backend + Frontend (For scalability)
```
sift/
├── backend/
│   ├── app.py            # FastAPI backend
│   ├── services/
│   │   ├── document_service.py
│   │   ├── vector_service.py
│   │   └── rag_service.py
│   └── requirements.txt
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── requirements.txt
└── README.md
```

**Recommendation**: Start with Option 1, refactor to Option 2 if needed.

---

## Work Breakdown & Branches

### Phase 1: Foundation (Branch: `feature/foundation`)
**Tasks:**
- [ ] Set up project structure
- [ ] Install dependencies (LangChain, FAISS, OpenAI, Streamlit)
- [ ] Create basic Streamlit app with file upload
- [ ] Implement PDF parsing (PyPDF2 or pdfplumber)

### Phase 2: Vector Store (Branch: `feature/vector-store`)
**Tasks:**
- [ ] Implement text chunking strategy (LangChain text splitters)
- [ ] Set up FAISS vector store (local) or Pinecone (cloud)
- [ ] Create embeddings using OpenAI or HuggingFace
- [ ] Test document indexing

### Phase 3: RAG Pipeline (Branch: `feature/rag-pipeline`)
**Tasks:**
- [ ] Implement retrieval logic (similarity search)
- [ ] Build LangChain RAG chain
- [ ] Integrate with OpenAI API (or local model)
- [ ] Add prompt engineering for document Q&A

### Phase 4: Chat Interface (Branch: `feature/chat-interface`)
**Tasks:**
- [ ] Build Streamlit chat UI
- [ ] Implement conversation history
- [ ] Add session state management
- [ ] Handle multiple documents

### Phase 5: Polish & Deploy (Branch: `feature/polish-deploy`)
**Tasks:**
- [ ] Error handling & validation
- [ ] Loading states & progress indicators
- [ ] Environment variable setup
- [ ] Deploy to Streamlit Cloud
- [ ] Write README with setup instructions

---

## Branch Strategy

```
main (production-ready code)
├── develop (integration branch)
│   ├── feature/foundation
│   ├── feature/vector-store
│   ├── feature/rag-pipeline
│   ├── feature/chat-interface
│   └── feature/polish-deploy
├── docs/* (documentation branches)
│   └── docs/readme
└── hotfix/* (urgent fixes)
```

**Workflow:**
1. Create feature branch from `develop`
2. Complete feature, test locally
3. Merge to `develop` (test integration)
4. Merge `develop` to `main` when stable
5. Deploy `main` to Streamlit Cloud

**Documentation Workflow:**
- Use `docs/*` branches for README, documentation, and setup guides
- Can be merged directly to `main` or `develop` as needed
- Example: `docs/readme` for README updates

---

## Tech Stack Details

### Core Libraries
- **LangChain**: Document processing, RAG chains, text splitters
- **FAISS**: Local vector storage (free, fast)
- **Pinecone**: Cloud vector DB (optional, for production)
- **OpenAI API**: GPT-4 or GPT-3.5-turbo for chat
- **Streamlit**: Frontend framework
- **PyPDF2/pdfplumber**: PDF parsing

### Alternative Stack (More Impressive)
- **LlamaIndex**: Alternative to LangChain
- **HuggingFace Transformers**: Local LLM (Llama 2, Mistral)
- **Chroma**: Alternative vector DB
- **Chainlit**: Alternative to Streamlit (more chat-focused)

---

## Development Timeline Estimate

- **Week 1**: Foundation + Vector Store (Phases 1-2)
- **Week 2**: RAG Pipeline + Chat Interface (Phases 3-4)
- **Week 3**: Polish + Deploy (Phase 5)

**Total**: 2-3 weeks for a solid MVP

---

## Next Steps

1. Choose: Streamlit monolith or separate backend/frontend
2. Set up virtual environment
3. Create `develop` branch
4. Start with `feature/foundation` branch
5. Build incrementally, test each phase


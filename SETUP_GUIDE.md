# Setup Guide

## Quick Start

### 1. Initial Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Branch Setup

### Create Development Branches

```bash
# Create and switch to develop branch
git checkout -b develop

# Create feature branches
git checkout -b feature/foundation
git checkout -b feature/vector-store
git checkout -b feature/rag-pipeline
git checkout -b feature/chat-interface
git checkout -b feature/polish-deploy
```

### Development Workflow

```bash
# Start working on a feature
git checkout feature/foundation

# Make changes, commit
git add .
git commit -m "feat: implement PDF parsing"

# Merge to develop
git checkout develop
git merge feature/foundation

# When ready, merge develop to main
git checkout main
git merge develop
```

---

## Testing Each Phase

### Phase 1: Foundation
- [ ] Can upload PDF file
- [ ] PDF is parsed successfully
- [ ] Text is extracted correctly

### Phase 2: Vector Store
- [ ] Text is chunked properly
- [ ] Embeddings are created
- [ ] Vector store is created and searchable

### Phase 3: RAG Pipeline
- [ ] Can retrieve relevant chunks
- [ ] RAG chain generates answers
- [ ] Answers are relevant to document

### Phase 4: Chat Interface
- [ ] Chat history persists
- [ ] Multiple questions work
- [ ] UI is responsive

### Phase 5: Deploy
- [ ] App runs on Streamlit Cloud
- [ ] Environment variables are set
- [ ] README is complete

---

## Troubleshooting

### Import Errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

### API Key Issues
- Check `.env` file exists
- Verify API key is correct
- Check OpenAI account has credits

### PDF Parsing Issues
- Try different PDF files
- Some PDFs may be image-based (need OCR)

### Vector Store Issues
- Check FAISS is installed correctly
- Verify embeddings are being created
- Check chunk size is appropriate

---

## Next Steps

1. Start with `feature/foundation` branch
2. Implement PDF parsing
3. Test locally
4. Move to next phase


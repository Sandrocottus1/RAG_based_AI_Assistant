Pluang Internal Knowledge Assistant (RAG)
========================================

Overview
--------
Streamlit-based internal knowledge assistant using a FAISS vector index and
Hugging Face Router APIs for embeddings and LLM inference.

Requirements
------------
- Python 3.9+
- Hugging Face API token in `HUGGINGFACEHUB_API_TOKEN`

Setup
-----
1) Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

2) Add documents to `data/raw` (txt or pdf).

3) Set the environment variable:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Run
---
```
streamlit run main.py
```

Notes
-----
- Click "Re-Index Knowledge Base" in the sidebar after adding documents.
- The FAISS index is stored in `faiss_index`.

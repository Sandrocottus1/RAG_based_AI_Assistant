AI Knowledge Assistant (RAG)
============================

Overview
--------
Streamlit-based knowledge assistant using a FAISS vector index and
Hugging Face Router APIs for embeddings and LLM inference.

Deployed App
-------------
Access the live application here: https://rag-agent-tps6p.ondigitalocean.app/

Requirements
------------
- Python 3.9+
- Hugging Face API token in `HUGGINGFACEHUB_API_TOKEN`
- A Hugging Face Router-supported chat model (configurable)

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

Optional model configuration:

```
HUGGINGFACE_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
HUGGINGFACE_LLM_FALLBACK_MODELS=Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3
```

If you use a fine-grained token, make sure it has permission to call
Inference Providers.

Run
---
```
streamlit run main.py
```

Notes
-----
- Click "Re-Index Knowledge Base" in the sidebar after adding documents.
- The FAISS index is stored in `faiss_index`.

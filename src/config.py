# Cfg: central configuration for models, chunking, and paths.
import os

class Cfg:
    # Centralized config keeps tuning and environment choices consistent.
    pg_title = "AI Knowledge Base"
    # Embedding model optimized for speed/quality tradeoff in RAG.
    mdl_nm = "sentence-transformers/all-MiniLM-L6-v2"
    # Chat model chosen for instruction-following in policy Q&A.
    llm_model = "HuggingFaceH4/zephyr-7b-beta"
    # Chunking tuned to keep context focused while preserving meaning.
    ch_sz = 500
    ch_ol = 50
    # Retrieve top-k chunks to balance recall and prompt length.
    k_ret = 3
    # Ingestion + index paths are relative so the app is portable.
    d_path = "data/raw"
    v_path = "data/vector_store"
    idx_path = "faiss_index"
    # Chat history persistence to keep conversations stateful across reruns.
    chat_hist_path = "data/chat_history.json"
    # Keep last N turns (user+assistant pairs) to control prompt length.
    hist_max_turns = 6

    
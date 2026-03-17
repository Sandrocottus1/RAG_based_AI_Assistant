# Cfg: central configuration for models, chunking, and paths.
import os

class Cfg:
    # Centralized config keeps tuning and environment choices consistent.
    pg_title = "BMW Assistant Knowledge Base"
    # Embedding model optimized for speed/quality tradeoff in RAG.
    mdl_nm = "sentence-transformers/all-MiniLM-L6-v2"
    # Chat model can be overridden with HUGGINGFACE_LLM_MODEL.
    llm_model = os.getenv("HUGGINGFACE_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    # Optional comma-separated fallback models, used when the primary is unsupported.
    _fallback_models_env = os.getenv(
        "HUGGINGFACE_LLM_FALLBACK_MODELS",
        "Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3",
    )
    llm_fallback_models = tuple(
        model.strip() for model in _fallback_models_env.split(",") if model.strip()
    )
    # Chunking tuned to keep context focused while preserving meaning.
    ch_sz = 500
    ch_ol = 50
    # Retrieve top-k chunks to balance recall and prompt length.
    k_ret = 3
    # Ingestion + index paths are relative so the app is portable.
    d_path = "data/raw"
    v_path = "data/vector_store"
    idx_path = "faiss_index"
    # Only ingest files whose names contain one of these keywords.
    # Override with SOURCE_FILENAME_KEYWORDS (comma-separated), e.g. "bmw,cars".
    _source_keywords_env = os.getenv("SOURCE_FILENAME_KEYWORDS", "bmw")
    source_filename_keywords = tuple(
        k.strip().lower() for k in _source_keywords_env.split(",") if k.strip()
    )
    # Chat history persistence to keep conversations stateful across reruns.
    chat_hist_path = "data/chat_history.json"
    # Keep last N turns (user+assistant pairs) to control prompt length.
    hist_max_turns = 6

    
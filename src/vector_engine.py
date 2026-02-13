# VecEng: embedding adapter and FAISS index create/load.
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

from src.config import Cfg

class ManualHFEmbeddings(Embeddings):
    def __init__(self):
        # Read token from env so local dev and deployment use the same mechanism.
        self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.client = InferenceClient(api_key=self.api_token) if self.api_token else None

    # Implement __call__ so FAISS can treat the embedding object like a function.
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._call_api([text])
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return []

    def _call_api(self, texts):
        try:
            if not texts:
                return []
            if not self.api_token:
                # Return zero vectors so the pipeline doesn't crash in demo mode.
                print("Embedding API Error: Missing HUGGINGFACEHUB_API_TOKEN")
                return [[0.0] * 384 for _ in texts]
            if not self.client:
                # Same fallback to keep the UI responsive even if HF client fails.
                print("Embedding API Error: Missing InferenceClient")
                return [[0.0] * 384 for _ in texts]

            payload = texts if len(texts) > 1 else texts[0]
            data = self.client.feature_extraction(payload, model=Cfg.mdl_nm)

            if isinstance(data, list) and data and isinstance(data[0], list):
                return data
            if isinstance(data, list):
                return [data]
            # Final fallback: ensure downstream FAISS always receives a vector list.
            return [[0.0] * 384 for _ in texts]
        except Exception as e:
            # Fail gracefully so indexing does not take down the whole app.
            print(f"Embedding Connection Error: {str(e)}")
            return [[0.0] * 384 for _ in texts]

class VecEng:
    def __init__(self):
        self.hf = ManualHFEmbeddings()
        self.vector_store = None

    def crt_idx(self, chunks):
        # Build a fresh index when the admin re-ingests documents.
        self.vector_store = FAISS.from_documents(chunks, self.hf)
        self.vector_store.save_local(Cfg.idx_path)
        return self.vector_store

    def ld_idx(self):
        if os.path.exists(Cfg.idx_path):
            self.vector_store = FAISS.load_local(
                Cfg.idx_path,
                self.hf, 
                # Local FAISS uses pickle; allow only because we own the index file.
                allow_dangerous_deserialization=True
            )
        return self.vector_store
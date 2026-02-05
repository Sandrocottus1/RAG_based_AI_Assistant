import os

from huggingface_hub import InferenceClient

from src.config import Cfg

class RAGBot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.repo_id = Cfg.llm_model
        self.client = InferenceClient(api_key=self.api_token) if self.api_token else None

    def get_chn(self):
        return self

    def invoke(self, input_dict):
        query = input_dict["query"]

        # 1. Retrieve Docs
        docs = self.vector_store.similarity_search(query, k=Cfg.k_ret)
        
        # 2. Build messages for chat-completions
        context_text = "\n\n".join([d.page_content for d in docs])
        system_msg = (
            "You are a helpful assistant. Use the context provided to answer the user's question. "
            "If you don't know, say \"I don't know\". Do not make up facts.\n\n"
            f"Context:\n{context_text}"
        )

        if not self.api_token:
            return {
                "result": "Missing HUGGINGFACEHUB_API_TOKEN in environment.",
                "source_documents": docs,
            }
        if not self.client:
            return {
                "result": "Missing InferenceClient configuration.",
                "source_documents": docs,
            }

        try:
            completion = self.client.chat.completions.create(
                model=self.repo_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            ans = completion.choices[0].message.content
        except Exception as e:
            ans = f"Connection Error: {str(e)}"

        return {
            "result": ans,
            "source_documents": docs
        }
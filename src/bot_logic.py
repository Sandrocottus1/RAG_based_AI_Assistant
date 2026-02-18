# RAGBot: retrieval + LLM orchestration for grounded policy Q&A.
import os
import re

from huggingface_hub import InferenceClient

from src.config import Cfg

class RAGBot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        # Pull the API token from env so code stays deployable without hardcoding secrets.
        self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.repo_id = Cfg.llm_model
        # Defer client creation if token is missing to keep the app usable (with warnings).
        self.client = InferenceClient(api_key=self.api_token) if self.api_token else None

    def get_chn(self):
        return self

    def invoke(self, input_dict):
        query = input_dict["query"]
        chat_history = input_dict.get("chat_history", [])

        # 1. Retrieve Docs
        # Retrieve first so the LLM response is grounded in company policy, not guesswork.
        docs = self.vector_store.similarity_search(query, k=Cfg.k_ret)
        
        # 2. Build messages for chat-completions
        context_text = "\n\n".join([d.page_content for d in docs])
        system_msg = (
            # Strict system prompt to avoid hallucination in a compliance-sensitive domain.
            "You are a helpful assistant providing clear, professional answers about company policies. "
            "Write in clear paragraphs. Only use numbered lists (1. 2. 3.) when absolutely necessary for step-by-step instructions. "
            "DO NOT use bullet points or dashes. Avoid any sub-bullets or nested formatting. "
            "Keep answers concise and grounded in the provided context. "
            "If you don't know, say 'I don't know'. Do not make up facts.\n\n"
            f"Context:\n{context_text}"
        )

        if not self.api_token:
            # Fail fast with a human-readable message so the UI can guide the admin.
            return {
                "result": "Missing HUGGINGFACEHUB_API_TOKEN in environment.",
                "source_documents": docs,
            }
        if not self.client:
            # Defensive check in case client init fails or env changes at runtime.
            return {
                "result": "Missing InferenceClient configuration.",
                "source_documents": docs,
            }

        try:
            # Keep message roles constrained to user/assistant for history.
            prior_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in chat_history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            # Low temperature keeps responses consistent for policy Q&A.
            completion = self.client.chat.completions.create(
                model=self.repo_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    *prior_msgs,
                    {"role": "user", "content": query},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            ans = completion.choices[0].message.content
            ans = self._format_answer(ans)
        except Exception as e:
            # Surface upstream errors without crashing the app.
            ans = f"Connection Error: {str(e)}"

        return {
            "result": ans,
            # Return sources so UI can show citations for trust and auditability.
            "source_documents": docs
        }

    @staticmethod
    def _format_answer(text):
        cleaned = (text or "").strip()
        if not cleaned:
            return "I don't know."
        
        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        
        # Split into lines and clean each
        lines = [line.rstrip() for line in cleaned.split("\n")]
        
        # Remove excessive blank lines
        cleaned_lines = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank:
                if not prev_blank:
                    cleaned_lines.append("")
                prev_blank = True
            else:
                cleaned_lines.append(line.strip())
                prev_blank = False
        
        # Strip leading/trailing blanks
        while cleaned_lines and cleaned_lines[0] == "":
            cleaned_lines.pop(0)
        while cleaned_lines and cleaned_lines[-1] == "":
            cleaned_lines.pop()
        
        # Check if response has intentional list formatting
        has_list = any(re.match(r"^\s*(?:-|\*|\d+[\.\)])\s+", line) for line in cleaned_lines)
        
        if has_list:
            # Clean up list items: remove extra indentation and fix numbering
            result_lines = []
            for line in cleaned_lines:
                if line:
                    # Remove leading bullets/numbers and extra spaces, keep content
                    stripped = re.sub(r"^\s*[-*•·]\s+", "", line)
                    stripped = re.sub(r"^\s*\d+[\.\)]\s+", "", stripped)
                    result_lines.append(stripped)
                else:
                    result_lines.append("")
            
            # Remove consecutive blank lines
            final_lines = []
            prev_blank = False
            for line in result_lines:
                if not line.strip():
                    if not prev_blank:
                        final_lines.append("")
                    prev_blank = True
                else:
                    final_lines.append(line)
                    prev_blank = False
            
            return "\n".join(final_lines).strip()
        
        # For non-list responses, join into paragraphs
        text_content = "\n".join(cleaned_lines)
        normalized = re.sub(r"[ \t]+", " ", text_content).strip()
        
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "I don't know."
        
        # Short replies stay as paragraph
        if len(sentences) <= 2 and len(normalized) <= 240:
            return " ".join(sentences)
        
        # Group sentences into readable paragraphs (2-3 sentences each)
        paragraphs = []
        current_para = []
        for sentence in sentences:
            current_para.append(sentence)
            if len(current_para) >= 2:
                paragraphs.append(" ".join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(" ".join(current_para))
        
        return "\n\n".join(paragraphs)
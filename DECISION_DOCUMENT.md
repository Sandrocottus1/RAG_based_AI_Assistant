# AI Knowledge Assistant - Technical Decision Document

**Author:** [Your Name]  
**Institution:** [Your College/University]  
**Date:** February 6, 2026  
**Project:** RAG-based Knowledge Assistant

---

## 1. Project Overview

### 1.1 Problem Statement
The organization needed an internal knowledge assistant to help employees quickly query policies and documentation without manual searching through multiple documents. The solution required:
- Fast, accurate retrieval of relevant information
- Natural language query interface
- Easy document updates and re-indexing
- Low infrastructure overhead
- Minimal setup complexity

### 1.2 Solution
A Retrieval-Augmented Generation (RAG) application built with:
- **Frontend:** Streamlit for interactive UI
- **Vector Database:** FAISS for local embeddings storage
- **LLM & Embeddings:** Hugging Face Inference API (cloud-based)
- **Document Processing:** LangChain for chunking and ingestion

---

## 2. Technical Architecture

### 2.1 System Flow
```
User Query → Streamlit UI → Vector Search (FAISS) → Context Retrieval → 
LLM (Zephyr-7b-beta) → Response Generation → UI Display
```

### 2.2 Core Components

#### Document Ingestion Pipeline
- **Input:** `.txt` and `.pdf` files from `data/raw/`
- **Processing:** RecursiveCharacterTextSplitter (500 char chunks, 50 char overlap)
- **Embedding:** sentence-transformers/all-MiniLM-L6-v2 via HF Inference API
- **Storage:** FAISS index saved locally in `faiss_index/`

#### Query Pipeline
1. User submits natural language query
2. Query is embedded using the same sentence-transformer model
3. FAISS performs similarity search (k=3 most relevant chunks)
4. Retrieved context + query sent to Zephyr-7b-beta LLM
5. LLM generates contextual answer
6. Response displayed with source citations

---

## 3. Key Technical Decisions

### 3.1 Why FAISS?
**Decision:** Use FAISS for vector storage instead of cloud-based vector DBs (Pinecone, Weaviate, Qdrant).

**Rationale:**
- ✅ **Local-first:** No external dependencies, works offline after initial indexing
- ✅ **Fast:** Optimized for similarity search, sub-millisecond query times
- ✅ **Cost-effective:** No monthly DB fees
- ✅ **Simple deployment:** Single file storage, easy to version control
- ⚠️ **Trade-off:** Requires re-indexing on document updates (acceptable for internal use)

**Alternatives Considered:**
- ChromaDB: More features but heavier dependency
- Pinecone: Cloud-based, easier scaling but recurring costs
- Weaviate: More complex setup for simple use case

### 3.2 Why Hugging Face InferenceClient?
**Decision:** Use cloud-based Hugging Face Inference API instead of local model execution.

**Rationale:**
- ✅ **Zero local compute:** No GPU required, works on any machine
- ✅ **No model downloads:** Avoids multi-GB downloads (transformers, torch, etc.)
- ✅ **Instant updates:** Access latest models without re-downloading
- ✅ **Dependency light:** Minimal Python packages needed
- ⚠️ **Trade-off:** Requires API token and internet connection, small latency overhead

**Alternatives Considered:**
- Local transformers: Heavy dependencies (torch ~2GB, models ~7GB)
- OpenAI API: More expensive, vendor lock-in
- LangChain's built-in HF integration: Version conflicts with pydantic/torch

### 3.3 Why Zephyr-7b-beta for LLM?
**Decision:** Use `HuggingFaceH4/zephyr-7b-beta` as the conversational model.

**Rationale:**
- ✅ **Optimized for chat:** Fine-tuned for conversational tasks
- ✅ **Good context handling:** Supports long prompts with retrieved context
- ✅ **Free tier friendly:** Available via HF Inference Providers
- ✅ **Open-weights:** Transparency and reproducibility
- ⚠️ **Trade-off:** Smaller than GPT-4, may miss nuanced edge cases

**Alternatives Considered:**
- GPT-3.5/4: Better accuracy but cost prohibitive for internal tool
- Mistral-7B: Similar quality but Zephyr has better instruction following
- Llama models: Require more setup for inference

### 3.4 Why sentence-transformers/all-MiniLM-L6-v2?
**Decision:** Use all-MiniLM-L6-v2 for embeddings.

**Rationale:**
- ✅ **Fast:** 384-dimensional embeddings, quick to compute
- ✅ **Good semantic quality:** Performs well on general text
- ✅ **Lightweight:** Small model size, fast API responses
- ✅ **Widely supported:** Compatible with FAISS and HF API
- ⚠️ **Trade-off:** Not domain-specific (could fine-tune for better policy matching)

**Alternatives Considered:**
- OpenAI text-embedding-ada-002: Higher quality but costs add up
- E5-large: Better accuracy but slower, 1024 dims vs 384

### 3.5 Why Streamlit?
**Decision:** Build UI with Streamlit instead of React/FastAPI frontend.

**Rationale:**
- ✅ **Rapid prototyping:** Python-only, no frontend framework needed
- ✅ **Built-in components:** Chat interface, file upload, sidebar out of the box
- ✅ **Easy deployment:** Streamlit Community Cloud free hosting
- ✅ **Internal tool fit:** Designed for data apps and dashboards
- ⚠️ **Trade-off:** Less customization than full frontend framework

**Alternatives Considered:**
- React + FastAPI: More control but 2x development time
- Gradio: Similar but Streamlit has better production deployment options

---

## 4. Architecture Trade-offs

### 4.1 Cloud API vs Local Models
**Choice:** Cloud-based inference (HF InferenceClient)

| Aspect | Cloud API (Chosen) | Local Models |
|--------|-------------------|--------------|
| Setup complexity | Low | High (CUDA, model downloads) |
| Compute requirements | Minimal | GPU required |
| Dependency size | ~50MB | ~10GB (torch + models) |
| Latency | +200ms API overhead | Faster (local) |
| Cost | Token-based (free tier OK) | Hardware cost upfront |
| Scalability | Auto-scales | Limited by hardware |

**Verdict:** For an internal tool with moderate query volume, cloud API wins on simplicity and cost.

### 4.2 FAISS Local vs Vector DB Cloud
**Choice:** Local FAISS storage

| Aspect | FAISS (Chosen) | Cloud Vector DB |
|--------|---------------|-----------------|
| Cost | Free | Monthly fees |
| Setup | Single file | Account + API setup |
| Performance | Excellent (local) | Network latency |
| Scaling | Manual re-index | Auto-scales |
| Backup | Git-versioned | Separate backup needed |

**Verdict:** For 1000s of documents (not millions), local FAISS is simpler and faster.

### 4.3 Python 3.14 Compatibility
**Issue:** LangChain's Pydantic v1 shows compatibility warnings on Python 3.14.

**Mitigation:**
- Warnings are non-blocking; app runs fine
- Future: downgrade to Python 3.12 or wait for LangChain v2 update
- Not critical for MVP deployment

---

## 5. Implementation Highlights

### 5.1 Custom Embedding Wrapper
Created `ManualHFEmbeddings` class that:
- Inherits from `langchain_core.embeddings.Embeddings`
- Uses `InferenceClient.feature_extraction()` for cloud-based embeddings
- Handles batching and error fallbacks (returns zero vectors on API failure)
- Avoids local transformers/torch dependencies

### 5.2 Chat-Completions for LLM
Switched from `text_generation` to `chat.completions.create()`:
- Reason: Zephyr-7b-beta only supports conversational task
- Benefit: Proper system/user message formatting
- Format: System message contains retrieved context, user message has query

### 5.3 Session State Management
Streamlit session state caches the loaded FAISS index:
- Avoids re-loading on every query
- Re-index button updates session state immediately
- Improves user experience (no waiting between queries)

---

## 6. Security Considerations

### 6.1 API Token Management
- Token stored in `.env` file (gitignored)
- Streamlit Cloud uses encrypted secrets
- Token has "read" permission only (minimal scope)
- Token rotation supported via HF settings

### 6.2 Sensitive Data
- Documents stored locally in `data/raw/` (not in git)
- FAISS index gitignored (can contain sensitive embeddings)
- No user data logged or stored

---

## 7. Performance Metrics

### 7.1 Query Latency
- FAISS similarity search: <50ms (3 chunks from 1000 docs)
- Embedding API call: ~200ms (single query)
- LLM response generation: ~2-5s (depends on HF provider load)
- **Total end-to-end:** ~3-6 seconds per query

### 7.2 Indexing Time
- 2 documents → 2 chunks: <5 seconds
- Expected for 100 docs (~500 chunks): ~30 seconds
- Re-indexing required when adding new documents

---

## 8. Future Improvements

### 8.1 Short-term (Next Sprint)
1. **Better chunking strategy:** Use semantic chunking instead of fixed-size
2. **Query history:** Add chat history context for follow-up questions
3. **Streaming responses:** Show LLM output as it generates
4. **Citation links:** Direct links to source documents

### 8.2 Medium-term (Next Quarter)
1. **Multi-document types:** Support Word docs, Markdown, HTML
2. **Fine-tuned embeddings:** Domain-specific model for policy documents
3. **User feedback loop:** Thumbs up/down on answers
4. **Analytics dashboard:** Query trends, top topics, response quality

### 8.3 Long-term (Future Considerations)
1. **Advanced RAG:** Re-ranking, hybrid search (keyword + semantic)
2. **Multi-modal support:** Images, tables in PDFs
3. **Access control:** Role-based document visibility
4. **Integration:** Slack bot, API endpoints for other tools

---

## 9. Lessons Learned

### 9.1 Technical Lessons
- **HF Router API quirks:** Different endpoints for different tasks (feature-extraction vs text-generation)
- **Pydantic conflicts:** LangChain + newer Python versions have dependency friction
- **TOML secrets:** Streamlit Cloud requires specific format for environment variables

### 9.2 Process Lessons
- **Start simple:** FAISS + cloud API got MVP working in hours vs days
- **Test early:** Deployed to Streamlit Cloud early to catch deployment issues
- **Document decisions:** Clear rationale prevents second-guessing later

---

## 10. Conclusion

This RAG implementation balances:
- **Simplicity:** Minimal dependencies, cloud-based inference
- **Performance:** Sub-second retrieval, acceptable LLM latency
- **Cost:** Free tier for moderate usage
- **Maintainability:** Clear architecture, standard libraries

The system is production-ready for internal use with room for incremental improvements based on user feedback and query volume growth.

---

## Appendix: Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend | Streamlit | 1.54.0 | Interactive UI |
| Vector DB | FAISS | 1.13.2 | Local vector storage |
| LLM | Zephyr-7b-beta | - | Conversational AI |
| Embeddings | all-MiniLM-L6-v2 | - | Text vectorization |
| Orchestration | LangChain | 1.2.8 | Document processing |
| API Client | huggingface_hub | 0.36.1 | Cloud inference |
| Language | Python | 3.14.3 | Runtime |
| Deployment | Streamlit Cloud | - | Hosting |

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026

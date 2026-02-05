# Architectural Decisions

## 1. Choice of Stack
I utilized **Streamlit** for the frontend to minimize UI boilerplate and focus on the RAG logic. 
The backend is **Python/LangChain** due to its robust connectors for vector operations.

## 2. Vector Strategy
* **Embeddings:** `all-MiniLM-L6-v2`. Chosen for its speed and low memory footprint (ideal for deployment on free tier instances).
* **Store:** `FAISS`. Selected over Pinecone/Weaviate for this specific assignment to keep the solution self-contained without external dependencies.

## 3. Chunking Logic
I implemented `RecursiveCharacterTextSplitter` with a chunk size of 500 tokens. This captures enough context for policy definitions without introducing noise from unrelated sections.
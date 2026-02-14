# Streamlit app: UI for indexing and chat-based policy Q&A.
import streamlit as st
import os
import json
from dotenv import load_dotenv

# Load environment variables (API Keys) first!
# Keep secrets out of code; env-based config matches deployment best practices.
load_dotenv()

from src.config import Cfg
from src.document_processor import DocProc
from src.vector_engine import VecEng
from src.bot_logic import RAGBot

# 1. Page Configuration
st.set_page_config(page_title=Cfg.pg_title)
st.title("AI Knowledge Assistant")

# 2. Sidebar / Admin Panel
with st.sidebar:
    st.header("Admin Panel")
    if st.button("Re-Index Knowledge Base"):
        with st.spinner("Ingesting Documents..."):
            dp = DocProc()
            frags = dp.get_frags()
            ve = VecEng()
            if frags:
                # Admin action rebuilds index to reflect newest policy files.
                idx = ve.crt_idx(frags)
                st.session_state["qa"] = RAGBot(idx).get_chn()
                st.success(f"Indexed {len(frags)} fragments successfully!")
            else:
                st.warning("No documents found. Add files to data/raw and retry.")

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []
        os.makedirs(os.path.dirname(Cfg.chat_hist_path), exist_ok=True)
        with open(Cfg.chat_hist_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        st.success("Chat history cleared.")

# 3. System Initialization (Lazy Loading)
def init_sys():
    ve = VecEng()
    # Correct: This loads the existing index from disk
    idx = ve.ld_idx() 
    if idx:
        bot = RAGBot(idx)
        return bot.get_chn()
    # Lazy-load warning so the UI stays usable even before indexing.
    st.sidebar.warning("Index not found. Use 'Re-Index Knowledge Base' to initialize.")
    return None

def load_chat_history():
    if os.path.exists(Cfg.chat_hist_path):
        try:
            with open(Cfg.chat_hist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict)]
        except Exception:
            return []
    return []

def save_chat_history(messages):
    os.makedirs(os.path.dirname(Cfg.chat_hist_path), exist_ok=True)
    with open(Cfg.chat_hist_path, "w", encoding="utf-8") as f:
        json.dump(messages, f)

if "qa" not in st.session_state:
    # Cache the chain across reruns so the app feels fast and consistent.
    st.session_state["qa"] = init_sys()

if "messages" not in st.session_state:
    st.session_state["messages"] = load_chat_history()

qa = st.session_state["qa"]
messages = st.session_state["messages"]

# 4. Chat Interface
for m in messages:
    role = m.get("role")
    content = m.get("content")
    if role in ("user", "assistant") and content:
        st.chat_message(role).markdown(content)

if q := st.chat_input("Ask about your documents..."):
    # Display user message
    st.chat_message("user").markdown(q)
    messages.append({"role": "user", "content": q})
    save_chat_history(messages)

    if qa:
        # Display AI Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Hold on a sec..._")

            # Send query to RAG pipeline
            # The chain returns both answer and sources for transparency.
            trimmed = messages[-(Cfg.hist_max_turns * 2):]
            res = qa.invoke({"query": q, "chat_history": trimmed})
            ans = res["result"]
            src = res["source_documents"]

            placeholder.markdown(ans)
            messages.append({"role": "assistant", "content": ans})
            save_chat_history(messages)

            # Show Citations
            with st.expander("View Sources"):
                for s in src:
                    source_name = s.metadata.get('source', 'Unknown File')
                    st.caption(f"Source: {source_name}")
                    # Preview snippet helps the interviewer see how retrieval worked.
                    st.text(s.page_content[:150] + "...")
    else:
        # Clear guidance to prevent confusion when index is missing.
        err_msg = "System not initialized. Please click 'Re-Index Knowledge Base' in the sidebar first."
        st.error("⚠️ " + err_msg)
        messages.append({"role": "assistant", "content": err_msg})
        save_chat_history(messages)
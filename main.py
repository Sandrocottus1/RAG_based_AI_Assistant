import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables (API Keys) first!
load_dotenv()

from src.config import Cfg
from src.document_processor import DocProc
from src.vector_engine import VecEng
from src.bot_logic import RAGBot

# 1. Page Configuration
st.set_page_config(page_title=Cfg.pg_title)
st.title("Pluang Internal Knowledge Assistant")

# 2. Sidebar / Admin Panel
with st.sidebar:
    st.header("Admin Panel")
    if st.button("Re-Index Knowledge Base"):
        with st.spinner("Ingesting Policy Documents..."):
            dp = DocProc()
            frags = dp.get_frags()
            ve = VecEng()
            if frags:
                # FIX: Use crt_idx (Create) here, not ld_idx (Load)
                idx = ve.crt_idx(frags)
                st.session_state["qa"] = RAGBot(idx).get_chn()
                st.success(f"Indexed {len(frags)} fragments successfully!")
            else:
                st.warning("No documents found. Add files to data/raw and retry.")

# 3. System Initialization (Lazy Loading)
def init_sys():
    ve = VecEng()
    # Correct: This loads the existing index from disk
    idx = ve.ld_idx() 
    if idx:
        bot = RAGBot(idx)
        return bot.get_chn()
    st.sidebar.warning("Index not found. Use 'Re-Index Knowledge Base' to initialize.")
    return None

if "qa" not in st.session_state:
    st.session_state["qa"] = init_sys()

qa = st.session_state["qa"]

# 4. Chat Interface
if q := st.chat_input("Ask about Pluang's policies..."):
    # Display user message
    st.chat_message("user").markdown(q)

    if qa:
        # Display AI Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Hold on a sec..._")

            # Send query to RAG pipeline
            res = qa.invoke({"query": q})
            ans = res["result"]
            src = res["source_documents"]

            placeholder.markdown(ans)

            # Show Citations
            with st.expander("View Sources"):
                for s in src:
                    source_name = s.metadata.get('source', 'Unknown File')
                    st.caption(f"Source: {source_name}")
                    st.text(s.page_content[:150] + "...")
    else:
        st.error("⚠️ System not initialized. Please click 'Re-Index Knowledge Base' in the sidebar first.")
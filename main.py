import streamlit as st
import os 
from src.config import Cfg
from src.document_processor import DocProc
from src.vector_engine import VecEng
from src.bot_logic import RAGBot

st.set_page_config(page_title=Cfg.pg_title)
st.title("Pluang Internal Knowledge Assistant")

if "eng" not in st.session_state:
    st.session_state.eng=None

with st.sidebar:
    st.header("Admin Panel")
    if st.button("Re-Index Knowldege Base"):
        with st.spinner("Ingesting Policy Documents..."):
            dp=DocProc()
            frags=dp.get_frags()
            ve=VecEng()
            ve.bld_idx(frags)
            st.success(f"Indexed {len(frags)} fragments.")

def init_sys():
    ve=VecEng()
    idx=ve.ld_idx()
    if idx:
        bot=RAGBot(idx)
        return bot.get_chn()
    return None
qa=init_sys()

if q:=st.chat_input("Ask about Pluang's policies..."):
    st.chat_message("user").markdown(q)

    if qa:
        res = qa({"query": q})
        ans = res["result"]
        src = res["source_documents"]
        
        with st.chat_message("assistant"):
            st.markdown(ans)
            with st.expander("View Sources"):
                for s in src:
                    st.caption(f"File: {s.metadata['source']}")
                    st.text(s.page_content[:150] + "...")
    else:
        st.error("System not initialized. Please Index data first.")



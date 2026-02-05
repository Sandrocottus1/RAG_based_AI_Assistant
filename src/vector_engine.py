from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import Cfg

import os

class VecEng:
    def __init__(self):
        self.emb=HuggingFaceEmbeddings(model_name=Cfg.mdl_nm)

        def bld_idx(self,frags):
            db=FAISS.from_documents(frags,self.emb)
            db.save_local(Cfg.v_path)
            return db
        def ld_idx(self):
            if not os.path.exists(Cfg.v_path):
                raise FileNotFoundError("Vector index not found. Build it first.")
            return FAISS.load_local(
                Cfg.v_path,
                self.emb,
                allow_dangerous_deserialization=True
            )
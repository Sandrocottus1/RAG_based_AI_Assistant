import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Cfg

class DocProc:
    def __init__(self):
        self.spl = RecursiveCharacterTextSplitter(
            chunk_size=Cfg.ch_sz,
            chunk_overlap=Cfg.ch_ol
        )
    def ld_docs(self):
        raw_d = []
        if not os.path.exists(Cfg.d_path):
            os.makedirs(Cfg.d_path)
            return raw_d

        for f in os.listdir(Cfg.d_path):
            fp = os.path.join(Cfg.d_path, f)

            if f.endswith(".txt"):
                l = TextLoader(fp, encoding="utf-8")
                raw_d.extend(l.load())
            elif f.endswith(".pdf"):
                l = PyPDFLoader(fp)
                raw_d.extend(l.load())
        return raw_d

    def get_frags(self):
        d = self.ld_docs()
        if not d:
            print("No documents found to ingest.")
            return []
        frags = self.spl.split_documents(d)
        print(f"Loaded {len(d)} docs → {len(frags)} chunks")
        return frags
      
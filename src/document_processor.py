import os
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Cfg

class DocProc:
    def __init__(self):
        self.spl=RecursiveCharacterTextSplitter(
            chunk_size=Cfg.ch_sz,
            chunk_overlap=Cfg.ch_ol
        )
    def ld_docs(self):
        raw_d=[]
        if not os.path.exists(Cfg.d_path):
            os.makedirs(Cfg.d_path)

        for f in os.listdir(Cfg.d_path):
            fp=os.path.join(Cfg.d_path,f) 

            if f.endswith(".txt"):
                l=TextLoader(fp)
                raw_d.extend(l.load())
            elif f.endswith(".pdf"):
                l=PyPDFLoader(fp)
                raw_d.extend(l.load())
        return raw_d

    def get_frags(self):
        d=self.ld_docs()
        return self.spl.split_documents(d)
      
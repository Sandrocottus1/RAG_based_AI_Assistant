# DocProc: load raw files and split into retrievable chunks.
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Cfg

class DocProc:
    def __init__(self):
        # Recursive splitter balances context size and retrieval granularity.
        self.spl = RecursiveCharacterTextSplitter(
            chunk_size=Cfg.ch_sz,
            chunk_overlap=Cfg.ch_ol
        )
    def ld_docs(self):
        raw_d = []
        if not os.path.exists(Cfg.d_path):
            # Create the folder on first run so the admin can drop files in easily.
            os.makedirs(Cfg.d_path)
            return raw_d

        keywords = getattr(Cfg, "source_filename_keywords", ())

        for f in os.listdir(Cfg.d_path):
            fp = os.path.join(Cfg.d_path, f)
            if not os.path.isfile(fp):
                continue

            # Keep the index domain-specific by only ingesting matching filenames.
            if keywords and not any(k in f.lower() for k in keywords):
                continue

            if f.endswith(".txt"):
                # TextLoader is the simplest path for internal policy docs.
                l = TextLoader(fp, encoding="utf-8")
                raw_d.extend(l.load())
            elif f.endswith(".pdf"):
                # Support PDFs because policy documents often come from HR/legal exports.
                l = PyPDFLoader(fp)
                raw_d.extend(l.load())
        return raw_d

    def get_frags(self):
        d = self.ld_docs()
        if not d:
            print("No documents found to ingest.")
            return []
        # Chunking improves retrieval accuracy by matching smaller, focused passages.
        frags = self.spl.split_documents(d)
        print(f"Loaded {len(d)} docs → {len(frags)} chunks")
        return frags
      
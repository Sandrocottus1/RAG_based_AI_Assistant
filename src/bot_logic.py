from langchain.chains import RetrievalQA
from langchain_community.llms import HiggingFaceHub
from langchain.prompts import PromptTemplate

from src.config import Cfg

class RAGBot:
    def __init__(self, v_db):
        self.db=v_db
        self.llm=HiggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature":0.1,"max_length":512}
        )
    
    def get_chn(self):
        tmpl = """
        Use the context below to answer the question. If you don't know, say "Data not found in Pluang KB".
        Context: {context}
        Question: {question}
        Answer:
        """
        p=PromptTemplate(template=tmpl, input_variables=["context","question"])


        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwars={"k":Cfg.k_ret}),
            return_source_documents=True,
            chain_type_kwargs={"prompt":p}
        )

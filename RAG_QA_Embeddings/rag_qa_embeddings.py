from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from transformers import pipeline
import faiss
import os


class rag_qa_embeddings:
    def __init__(self, params):
        env = os.environ.copy()
        env["PYTHONPATH"] = "rag_qa_database"
        self.database_path = os.path.join(os.getcwd(), "RAG_QA_Embeddings/rag_qa_database")
        self.database = self.load_database()
        self.model = self.load_model()
        self.k = int(params[0])

    def load_database(self):
        embedding_model = self.load_embedding_model()
        return FAISS.load_local(self.database_path, embedding_model, allow_dangerous_deserialization=True)

    def load_model(self):
        model = pipeline("question-answering")
        return model

    def load_embedding_model(self):
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    def get_contexts(self, question):
        return self.database.similarity_search(question, self.k)

    def get_answer(self, question, contexts):
        context = ""
        for con in contexts:
            context += con.page_content + "\n"
        answer = self.model(question=question, context=context)
        return answer['answer']

    def invoke(self, question):
        contexts = self.get_contexts(question)
        answer = self.get_answer(question, contexts)
        return answer

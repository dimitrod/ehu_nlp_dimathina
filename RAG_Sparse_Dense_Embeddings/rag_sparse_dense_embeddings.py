import joblib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from datetime import datetime
from pathlib import Path

class rag_sparse_dense_embeddings:
    def __init__(self, params):
        #setup path variable
        env = os.environ.copy()
        env["PYTHONPATH"] = "database"
        self.database_path = Path(os.getcwd())/"database"

        #load parameters
        self.k = int(params[0])
        self.model_id = params[1]

        #initialize vector base
        print(datetime.now(), ": loading vector base")
        self.vector_base = joblib.load(self.database_path/"database.pkl")
        self.vectorizer = self.initialize_vectorizer()
        self.documents = self.load_documents()

        #initialize text splitter and embedding model
        print(datetime.now(), ": loading embedding model")
        self.text_splitter = self.create_text_splitter()
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        #load reader model
        print(datetime.now(), ": loading reader model")
        self.reader_model = pipeline('question-answering', model_id=self.model_id)

    def invoke(self, question):
        contexts = self.get_contexts(question)
        answer = self.get_answer(question, contexts)
        return answer['answer']

    def get_contexts(self, question):
        print(datetime.now(), ": retrieving contexts")
        contexts = self.retrieve_contexts(question)
        print(datetime.now(), ": filtering contexts")
        paragraphs = self.filter_context(contexts, question)
        return paragraphs

    def retrieve_contexts(self, question):
        query = self.vectorizer.transform([question])
        similarity_scores = cosine_similarity(query, self.vector_base).flatten()
        top_indices = np.argsort(similarity_scores)[-self.k:][::-1][:2]
        contexts = ""
        for index in top_indices:
            contexts += self.documents[index]
        return contexts

    def filter_context(self, contexts, question):
        paragraphs = self.text_splitter.split_text(contexts)
        lib = FAISS.from_texts(paragraphs, self.embedding_model)
        top_paragraphs = lib.similarity_search(question, self.k)
        context = ""
        for paragraph in top_paragraphs:
            context += paragraph.page_content
        return context

    def get_answer(self, question, contexts):
        print(datetime.now(), ": generating answer")
        return self.reader_model(question=question, context=contexts)

    def initialize_vectorizer(self):
        vectorizer = TfidfVectorizer()
        vocabulary = joblib.load(self.database_path/"tfidf_vocabulary.pkl")
        vectorizer.fit(vocabulary)
        return vectorizer

    def load_documents(self):
        with open(self.database_path/"documents.txt", "r", encoding='utf-8') as f:
            documents = f.readlines()

        return documents

    def create_text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Maximum size of each chunk
            chunk_overlap=10,  # Overlap between consecutive chunks
            separators=["\n\n", "\n", " ", ""],  # Hierarchy of delimiters
        )
        return text_splitter
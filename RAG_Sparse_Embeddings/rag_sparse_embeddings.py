import joblib
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class rag_sparse_embeddings:
    def __init__(self, params):
        print("loading vecorizer")
        self.vectorizer = self.initialize_vectorizer()
        print("loading vector base")
        self.vector_base = joblib.load('database/document_library.pkl')
        print("loading documents")
        self.documents = self.load_documents()
        print("loading model")
        self.model = pipeline('question-answering')
        self.top_n = params[0]

    def invoke(self, question):
        print("retrieving contexts")
        contexts = self.get_contexts(question)
        print("retrieving answer")
        answer = self.get_answer(question, contexts)
        return answer

    def get_contexts(self, question):
        query = self.vectorizer.transform([question])
        similarity_scores = cosine_similarity(query, self.vector_base).flatten()
        top_indices = similarity_scores.argsort()[::-1][:self.top_n]
        contexts = ""
        for index in top_indices:
            contexts += self.documents[index]
        return contexts

    def get_answer(self, question, contexts):
        answer = self.model(question=question, context=contexts)
        return answer['answer']

    def initialize_vectorizer(self):
        vectorizer = TfidfVectorizer()
        vocabulary = joblib.load('database/tfidf_vocabulary.pkl')
        vectorizer.fit(vocabulary)
        return vectorizer

    def load_documents(self):
        with open('database/documents.txt', 'r', encoding='utf-8') as f:
            documents = f.readlines()
        return documents
import joblib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import openai

class chat_gpt_rag:
    def __init__(self, params):
        #setup path variable
        env = os.environ.copy()
        env["PYTHONPATH"] = "database"
        self.database_path = Path(os.getcwd())/"ChatGPT_RAG"/"database"

        #load parameters
        self.k = int(params[0])
        self.chunk_size = int(params[1])
        self.overlap = int(params[2])
        self.temperature = float(params[3])

        #initialize vector base
        #print(datetime.now(), ": loading vector base")
        self.vector_base = joblib.load(self.database_path/"document_library.pkl")

        self.vectorizer = self.initialize_vectorizer()
        self.documents = self.load_documents()

        #initialize text splitter and embedding model
        #print(datetime.now(), ": loading embedding model")
        self.text_splitter = self.create_text_splitter()
        self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        #load reader model
        #print(datetime.now(), ": loading reader model")
        openai.api_key = input("Enter your OpenAI API Token: ")
        self.model = openai

    def invoke(self, question):
        contexts = self.get_contexts(question)
        answer = self.get_answer(question, contexts)
        return answer

    def get_contexts(self, question):
        #print(datetime.now(), ": Retrieving documents")
        contexts = self.retrieve_contexts(question)
        #print(datetime.now(), ": Filtering contexts")
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
        #print("Number of paragraphs: ", len(paragraphs))
        lib = FAISS.from_texts(paragraphs, self.embedding_model)
        top_paragraphs = lib.similarity_search(question, self.k)
        context = ""
        for paragraph in top_paragraphs:
            context += paragraph.page_content
        return context

    def get_answer(self, question, contexts):
        #print(datetime.now(), ": Creating message")
        messages = self.create_messages(question, contexts)
        #print("Messages: ", messages)
        #print(datetime.now(), ": Generating response")
        answer = self.model.chat.completions.create(model="gpt-4o", messages=messages, temperature=self.temperature)
        return answer.choices[0].message.content

    def create_messages(self, question, contexts):
        instruction = "You are a chatbot who always responds as shortly as possible."
        question_add = "Answer in one or two words, no additional information, no punctiation. Use the following text to find the answer:"
        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": question + "\n" +question_add + "\n" + contexts},
        ]
        return messages

    def initialize_vectorizer(self):
        vocabulary = joblib.load(self.database_path/"tfidf_vocabulary.pkl")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.8,
            min_df=2,
            max_features=10000,
            ngram_range=(1, 4),
            vocabulary=vocabulary,
        )
        dummy_data = [""]
        vectorizer.fit(dummy_data)
        return vectorizer

    def load_documents(self):
        with open(self.database_path/"documents.txt", "r", encoding='utf-8') as f:
            documents = f.readlines()
        return documents

    def create_text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # Maximum size of each chunk
            chunk_overlap=self.overlap,  # Overlap between consecutive chunks
            separators=["\n\n", "\n", " ", ""],  # Hierarchy of delimiters
        )
        return text_splitter
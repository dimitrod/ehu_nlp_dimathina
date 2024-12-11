import numpy as np
from huggingface_hub import login
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch

class tiny_llama_rag:
  def __init__(self, params):
    API_KEY = input('Enter Pinecone.io token: ')
    login()

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    self.retriever = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    self.model = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16, device_map="auto")
    self.pc = Pinecone(api_key=API_KEY)
    self.vectorizer = TfidfVectorizer()
    self.k = 5
    self.chunk_size = 400
    self.max_new_tokens = 30
    self.temperature = 0.2
    self.top_k = 10
    self.text_splitter = self.create_text_splitter()

  def invoke(self, question):
    context = self.get_contexts(question)
    return self.get_answer(question, context)

  def get_contexts(self, question):
    contexts = self.retrieve_from_database(question)
    return self.filter_contexts(contexts, question)

  def retrieve_from_database(self, question):
    question_context = ""
    index = self.pc.Index('wiki-train-minilm')
    query = question
    query_encoded = self.retriever.encode([query]).tolist()
    query_return = index.query(vector=query_encoded, top_k=2, include_metadata=True)
    for x in query_return['matches']:
      question_context = question_context + str(x['metadata']['bytes'])
    return question_context

  def filter_contexts(self, contexts, question):
    texts = self.text_splitter.split_text(contexts)
    self.vectorizer.fit(texts)
    lib = self.vectorizer.transform(texts)
    query = self.vectorizer.transform([question])
    similarity_scores = cosine_similarity(query, lib).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:self.k]
    context = ""
    for index in top_indices:
      context = context + texts[index]
    return context

  def get_answer(self, question, context):
    messages = self.create_messages(question, context)
    prompt = self.create_prompt(messages)
    outputs = self.model(prompt, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=self.temperature, top_k=self.top_k, top_p=0.95)
    output = outputs[0]["generated_text"]
    index = output.find("<|assistant|>")
    answer = output[index + len("<|assistant|>") :].strip()
    return answer

  def create_messages(self, question, context):
    question_add = " Answer in one or two words, no additional information, no punctiation. Use the following text to find the answer:"
    instruction = "You are a chatbot who always responds as shortly as possible."

    messages = [
      {
        "role": "system",
        "content": instruction,
      },
      {"role": "user", "content": question + question_add + context},
    ]
    return messages

  def create_prompt(self, messages):
    prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

  def create_text_splitter(self):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=self.chunk_size,  # Maximum size of each chunk
      chunk_overlap=10,  # Overlap between consecutive chunks
      separators=["\n\n", "\n", " ", ""],  # Hierarchy of delimiters
    )
    return text_splitter

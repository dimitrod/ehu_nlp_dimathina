from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch
from transformers import pipeline

class bert_base:
  def __init__(self, params):
    API_KEY = input('Enter Pinecone.io token: ')
    login()

    model_id = "mirbostani/bert-base-uncased-finetuned-triviaqa"

    self.retriever = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    self.model = pipeline("question-answering", model = model_id)
    self.pc = Pinecone(api_key=API_KEY)
    

  def invoke(self, question):
    question_context = ""
    index = self.pc.Index('wiki-train-minilm')
    query = question
    query_encoded = self.retriever.encode([query]).tolist()
    query_return = index.query(vector=query_encoded, top_k=2, include_metadata=True)
    for x in query_return['matches']:
      question_context = question_context + str(x['metadata']['bytes'])

    answer = self.model(question = question, context = question_context)

    return answer["answer"]


from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
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

  def create_prompt(self, messages):
    prompt = self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

  def invoke(self, question):
    question_add = " Answer in one or two words, no additional information, no punctiation. Use the following text to find the answer:"
    instruction = "You are a chatbot who always responds as shortly as possible."
    question_context = ""

    index = self.pc.Index('wiki-train-minilm')
    query = question
    query_encoded = self.retriever.encode([query]).tolist()
    query_return = index.query(vector=query_encoded, top_k=2, include_metadata=True)
    for x in query_return['matches']:
      question_context = question_context + str(x['metadata']['bytes'])

    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": question + question_add + question_context},
    ]

    prompt = self.create_prompt(messages)
    outputs = self.model(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    output = outputs[0]["generated_text"]
    index = output.find("<|assistant|>")
    answer = output[index + len("<|assistant|>") :].strip()
    return answer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import torch

class mistral_instruct_dense:
  def __init__(self, params):
    API_KEY = input('Enter Pinecone.io token: ')
    login()

    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    self.retriever = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.pc = Pinecone(api_key=API_KEY)

  def invoke(self, question):
    device = "cuda:0"
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

    inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = self.model.generate(inputs, max_new_tokens=20)
    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result.replace(question + question_add + question_context + " ", "")
    result = result.replace(instruction, "")
    return result

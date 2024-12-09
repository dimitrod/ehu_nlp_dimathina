from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from google.colab import userdata



class mistral_instruct:
  def __init__(self, params):
    API_KEY = userdata.get('PINECONE_TOKEN')
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    self.retriever = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.pc = Pinecone(api_key=API_KEY)

  def invoke(self, question):
    device = "cuda:0"
    question = "When was Baron Andrew Lloyd Webber born?"
    question_add = " Answer as shortly as possible, no additional information, no punctiation. Use the following text to find the answer:"
    instruction = "You are a chatbot who always responds as shortly as possible."


    index = pc.Index('wiki-index')
    query = "When was Baron Andrew Lloyd Webber born?"
    query_encoded = model.encode([query]).tolist()
    question_context = index.query(vector=query_encoded, top_k=2, include_metadata=True)

    
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

import torch
from transformers import pipeline

class tinyllama:
  def __init__(self, params):
    self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

  def create_prompt(self, messages):
    prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

  def invoke(self, question):
    messages = [
      {
          "role": "system",
          "content": "You are a chatbot who always responds as shortly as possible.",
      },
      {"role": "user", "content": question},
    ]
    prompt = self.create_prompt(messages)
    outputs = self.pipe(prompt, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    output = outputs[0]["generated_text"]
    index = output.find("<|assistant|>")
    answer = output[index + len("<|assistant|>") :].strip()
    return answer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

class mistral_instruct:
  def __init__(self, params):
    login()
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

  def invoke(self, question):
    device = "cuda:0"
    question_add = " Answer as shortly as possible, no additional information, no punctiation."
    instruction = "You are a chatbot who always responds as shortly as possible."

    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": question + question_add},
    ]

    inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    outputs = self.model.generate(inputs, max_new_tokens=50)
    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result.replace(question + question_add + " ", "")
    result = result.replace(instruction, "")
    return result

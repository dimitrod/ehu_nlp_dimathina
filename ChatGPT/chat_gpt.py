from openai import OpenAI
import os

class chat_gpt:
  def __init__(self, params):
    OPEN_API_KEY = input("Enter your OpenAI API Token: ")
    os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
    self.model = OpenAI()

  def invoke(self, question):
    answer = self.get_answer(question)
    return answer

  def get_answer(self, question):
    messages = [
      {
          "role": "system",
          "content": "You are a chatbot who always responds as shortly as possible.",
      },
      {"role": "user", "content": question},
    ]
    answer = self.model.chat.completions.create(model="gpt-4o", messages=messages)
    return answer.choices[0].message["content"]
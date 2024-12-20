import openai

class chat_gpt_no_retriever:
  def __init__(self, params):
    openai.api_key = input("Enter your OpenAI API Token: ")
    self.model = openai
    self.temperature = float(params[0])

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
    answer = self.model.chat.completions.create(model="gpt-4o", messages=messages, temperature=self.temperature)
    return answer.choices[0].message.content
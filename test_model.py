from ChatGPT_RAG.chat_gpt_rag import chat_gpt_rag
from datetime import datetime

params = [10, 300 , 0, 0.3]

model = chat_gpt_rag(params)

question = "Where does France lie?"
print("<",datetime.now(), "> Question: ", question)

answer = model.invoke(question)

print("<",datetime.now(), "> Answer: ",  answer)

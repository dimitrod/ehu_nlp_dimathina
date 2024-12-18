from ChatGPT.chat_gpt import chat_gpt

print("Loading model...")
params = []
model = chat_gpt(params)



question = "What is the capital of france?"

print("Question: {}".format(question))

print("Generating answer")
answer = model.invoke(question)

print("Answer: {}".format(answer))

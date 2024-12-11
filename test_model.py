from TinyLlamaRAG.tiny_llama_rag import tiny_llama_rag

params = []
print("loading model")
model = tiny_llama_rag(params)


question = "What is the capital of France?"

print(question)
print("Retrieving answer")
answer = model.invoke(question)

print(answer)
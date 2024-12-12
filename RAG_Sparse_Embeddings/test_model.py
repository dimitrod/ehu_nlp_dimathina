from RAG_Sparse_Embeddings.rag_sparse_embeddings import rag_sparse_embeddings

params = [10]

model = rag_sparse_embeddings(params)

question = "What is the capital of france?"
print(question)

answer = model.invoke(question)

print(answer)
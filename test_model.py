from RAG_Sparse_Dense_Embeddings.rag_sparse_dense_embeddings import rag_sparse_dense_embeddings
from datetime import datetime

params = [10, 300 , 0, 0.5, 30]

model = rag_sparse_dense_embeddings(params)

question = "Where does France lie?"
print("<",datetime.now(), "> Question: ", question)

answer = model.invoke(question)

print("<",datetime.now(), "> Answer: ",  answer)

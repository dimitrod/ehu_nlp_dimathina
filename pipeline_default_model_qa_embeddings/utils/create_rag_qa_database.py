import sys
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from tqdm import tqdm
import faiss
import json
import argparse

def create_model(model_name):
    print("Creating model")
    model = HuggingFaceEmbeddings(model_name=model_name)
    return model

def create_index(model):
    print("Creating index")
    index = faiss.IndexFlatL2(len(model.embed_query("hello world")))
    return index

def setup_database(model_name):
    print("Setting up database")
    model = create_model(model_name)
    index = create_index(model)
    database = FAISS(
        embedding_function=model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    return database

def write_to_database(database, database_set):
    for i in tqdm(range(len(database_set)), desc="Embedding data"):
      qa_pair = json.dumps(database_set[i])
      doc = Document(page_content=qa_pair, metadata={})
      database.add_documents([doc])
    return database

def read_dataset():
    print("Reading dataset")
    with open("RAG_QA_Embeddings/rag_qa_dataset/rag_qa_dataset.json", "r", encoding="utf-8") as f:
        database_set = json.loads(f.read())
    return database_set

def save_database(database):
    database.save_local("RAG_QA_Embeddings/rag_qa_database")

def get_args():
    parser = argparse.ArgumentParser(
        description='Creation of RAG QA database')
    parser.add_argument('--embedding_model', help='model to embed the vectors')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    embedding_model = args.embedding_model
    database_set = read_dataset()
    database = setup_database(embedding_model)
    database = write_to_database(database, database_set)
    save_database(database)

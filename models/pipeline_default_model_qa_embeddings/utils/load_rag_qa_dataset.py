from datasets import load_dataset
from tqdm import tqdm
import json
import argparse

def download_dataset(split, split_size):
  evaluation_data = load_dataset('trivia_qa', name='rc.wikipedia', split=f"{split}[:{split_size}]")
  return evaluation_data

def format_to_json(data):
  file = "{\"Data\":["
  i = 0
  for i in tqdm(range(len(data)), desc="Loading dataset"):
    file += json.dumps(data[i])
    if i < len(data) - 1:
      file += ",\n"
    i += 1
  file += "]}\n"
  return json.loads(file)

def extract_qa_pairs(data):
  qa_pairs = []
  for i in tqdm(range(len(data['Data'])), desc="Extracting QA Pairs"):
    entry = data['Data'][i]
    qa_pairs.append((entry['question'], entry['answer']['aliases']))
  return qa_pairs

def write_file(file, filename):
  print("Saving file...")
  with open(filename, "w", encoding="utf-8") as f:
    f.write(file)

def get_args():
    parser = argparse.ArgumentParser(
        description='Dataset Loader for RAG QA dataset')
    parser.add_argument('--split', help='split of the wikipedia triviaqa dataset')
    parser.add_argument('--split_size', type=int, help='size of the split')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    split = args.split
    split_size = args.split_size

    data = download_dataset(split, split_size)
    data = format_to_json(data)
    data = extract_qa_pairs(data)
    data = json.dumps(data)

    path = "RAG_QA_Embeddings/rag_qa_dataset/rag_qa_dataset.json"
    write_file(data, path)
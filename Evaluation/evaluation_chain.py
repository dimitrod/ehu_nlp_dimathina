from tqdm import tqdm
import subprocess
import os
import json
import importlib
import argparse

def evaluate_triviaqa(dataset_file, prediction_file):
  script_path = "triviaqa/evaluation/triviaqa_evaluation.py"

  project_root = os.path.abspath(os.getcwd())
  triviaqa_root = os.path.join(project_root, "triviaqa")

  env = os.environ.copy()
  env["PYTHONPATH"] = triviaqa_root

  result = subprocess.run([
    "python", script_path,
    "--dataset_file", dataset_file,
    "--prediction_file", prediction_file],
    env=env,
    capture_output=True,
    text=True,
  )
  print(result.stderr)
  return result.stdout

def get_answers(directory, questions, model):
    answers = {}

    for i in tqdm(range(len(questions)), desc="Retrieving Answers"):
        question = questions[i]
        answers[question[0]] = model.invoke(question[1])
    write_data(directory + "/predictions.json", answers)

def load_questions(path, split):
    create_split(path, split)

    with open(f"{path}/evaluation_dataset.json", "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    questions = []

    for i in tqdm(range(len(data["Data"])),desc="Loading Questions"):
        question = data["Data"][i]
        questions.append([question["QuestionId"], question["Question"]])
    return questions

def create_split(path, split):
  with open(f"{path}/original_dataset.json", "r", encoding="utf-8") as f:
    data = json.loads(f.read())

  data["Data"] = data["Data"][:split]

  write_data(f"{path}/evaluation_dataset.json", data)

def load_model(path, model_name):
  print("Loading model...")
  module = importlib.import_module(path)
  model = getattr(module, model_name)
  return model()

def write_data(path, data):
  with (open(path, "w", encoding="utf-8") as f):
    f.write(json.dumps(data, ensure_ascii=False))

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluation Chain for TriviaQA {}')
    parser.add_argument('--model_directory', help='Directory of the model')
    parser.add_argument('--model_module', help='Module of the model')
    parser.add_argument('--model_class', help='Class of the model')
    parser.add_argument('--split_size', help='Size of the split')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    directory = "Evaluation/datasets"

    split_size = int(args.split_size)
    model_directory = args.model_directory
    model_module = args.model_module
    model_class = args.model_class

    model_path = f"{model_directory}.{model_module}"
    model_name = f"{model_class}"

    questions = load_questions(directory, split_size)
    model = load_model(model_path, model_name)
    print("Model initiated")
    print("Starting QA...")
    answers = get_answers(directory, questions, model)
    print("QA complete")
    print("Starting evaluation...")
    results = evaluate_triviaqa(directory + "/evaluation_dataset.json", directory + "/predictions.json")
    print("Evaluation complete")
    print("Saving results...")
    with open(f"Evaluation/{model_name}_split_size={split_size}_results.txt", "w", encoding="utf-8") as f:
      f.write(results)
    print("Test results:")
    print(results)
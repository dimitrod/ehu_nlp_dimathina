import json

class dummy_model:
    def __init__(self):
        with open("Dummy_Model/dummy_data.json", "r", encoding="utf-8") as f:
            self.data = json.loads(f.read())


    def invoke(self, question):
        if question in self.data:
            return self.data[question]
        return ""
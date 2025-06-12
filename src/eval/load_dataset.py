import json
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/MATH-500")
with open("eval_math500.json", "w", encoding="utf-8") as f:
    json.dump(dataset['test'].to_list(), f, indent=2, ensure_ascii=False)

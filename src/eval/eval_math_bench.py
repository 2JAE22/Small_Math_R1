import os
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

BSZ = 64

parser = argparse.ArgumentParser(description="Math Evaluation Benchmark")
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--file_name', type=str, required=True)
args = parser.parse_args()

MODEL_PATH = args.model_path
file_name = args.file_name

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=8192,
    gpu_memory_utilization=0.8
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer



# Load JSON file (text-only)
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)

messages = []
for x in data:
    question = x['problem']
    prompt = QUESTION_TEMPLATE.format(Question=question)
    msg = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]
    messages.append(msg)

def extract_answer(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def normalize_math(expr):
    if not isinstance(expr, str):
        return ""
    
    # 기본 정리
    expr = expr.strip()
    
    # LaTeX 명령어 정규화
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\text\s*\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\mathrm\{(.*?)\}", r"\1", expr)
    expr = re.sub(r"\\mbox\{(.*?)\}", r"\1", expr)
    
    # 분수 정규화
    expr = expr.replace("\\frac", "frac")  # 분수 명령어 통일
    
    # 특수 문자 정규화
    expr = expr.replace("\\pi", "pi").replace("π", "pi")
    expr = expr.replace("\\circ", "°").replace("°", "degrees")
    expr = expr.replace("\\sqrt", "sqrt").replace("√", "sqrt")
    expr = expr.replace("\\times", "*").replace("×", "*")
    expr = expr.replace("\\div", "/").replace("÷", "/")
    expr = expr.replace("\\cdot", "*").replace("·", "*")
    
    # 공백 및 괄호 처리
    expr = re.sub(r"\s+", "", expr)  # 공백 제거
    expr = re.sub(r"[{}]", "", expr)  # 중괄호 제거
    
    # 후행 백슬래시 제거
    expr = expr.rstrip("\\")
    
    # 수식 기호 정규화
    expr = expr.replace("\\leq", "<=").replace("\\geq", ">=")
    expr = expr.replace("\\lt", "<").replace("\\gt", ">")
    expr = expr.replace("\\neq", "!=")
    
    # 제곱, 첨자 정규화
    expr = re.sub(r"(\d+)\^\\circ", r"\1degrees", expr)
    expr = re.sub(r"(\d+)\^{\s*\\circ\s*}", r"\1degrees", expr)
    expr = re.sub(r"(\d+)\^{(.*?)}", r"\1^\2", expr)
    
    # 불필요한 공백 제거
    expr = re.sub(r"\\\s+", "", expr)
    
    return expr.lower()  # 소문자로 통일

final_output = []

for i in tqdm(range(0, len(messages), BSZ)):
    batch_messages = messages[i:i+BSZ]
    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    llm_inputs = [{"prompt": p} for p in prompts]
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    batch_output_text = [out.outputs[0].text for out in outputs]

    for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
        sample["output"] = model_output
        sample["prediction"] = extract_answer(model_output)
        pred = normalize_math(sample["prediction"])
        ans = normalize_math(sample["answer"])
        sample["correct"] = (pred == ans)
        sample["correct_NotNormal"] = sample["prediction"] == sample["answer"]
        final_output.append(sample)

# Save results
output_path = f"{os.path.splitext(file_name)[0]}_eval_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)

# Print accuracy
acc = sum(1 for x in final_output if x["correct"]) / len(final_output)
print(f"Accuracy: {acc:.2%}")
print(f"Saved to {output_path}")

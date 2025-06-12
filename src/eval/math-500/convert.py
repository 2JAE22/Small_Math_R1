import json

# 입력 파일과 출력 파일 경로를 지정하세요
input_path = "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500.json"
output_path = "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500_for_bench.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []
for item in data:
    # 실제 데이터 구조에 맞게 필드명을 수정하세요
    new_item = {
        "problem_type": item.get("problem_type", "multiple choice"),  # 없으면 기본값
        "problem": item.get("problem", item.get("question", "")),     # "question" 필드가 있으면 사용
        "options": item.get("options", []),
        "data_type": item.get("data_type", "text"),
        "path": item.get("path", ""),
        "solution": item.get("solution", item.get("answer", "")),     # "answer" 필드가 있으면 사용
    }
    converted.append(new_item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"변환 완료! {output_path} 파일을 사용하세요.")
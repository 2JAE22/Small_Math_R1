import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter

# 한글 폰트 설정 (필요시)
# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['axes.unicode_minus'] = False

# ... existing code ...

# 결과 파일 지정
OUTPUT1 = "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500_eval_RfModel.json"
OUTPUT2 = "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500_eval_SFTModel.json"
OUTPUT3 = "/home/vilab/projects/video_Reinforcement/small_math_r1/src/eval/math-500/eval_math500_eval_BaseModel.json"

# 모델 이름 (그래프에 사용)
MODEL1_NAME = "RF Model"  # 첫 번째 모델 이름
MODEL2_NAME = "SFT Model"  # 두 번째 모델 이름
MODEL3_NAME = "Base Model"  # 세 번째 모델 이름

# 결과 로드
with open(OUTPUT1, "r", encoding="utf-8") as f:
    data1 = json.load(f)["results"]

with open(OUTPUT2, "r", encoding="utf-8") as f:
    data2 = json.load(f)["results"]

with open(OUTPUT3, "r", encoding="utf-8") as f:
    data3 = json.load(f)["results"]

# 색상 설정
color1 = '#3498db'  # 파란색
color2 = '#e74c3c'  # 빨간색
color3 = '#2ecc71'  # 녹색

# 1. 전체 정확도 비교 (막대 그래프)
plt.figure(figsize=(10, 6))

# 모델1 정확도
total1 = len(data1)
correct1 = sum(1 for x in data1 if x.get("correct", False))
accuracy1 = correct1 / total1 if total1 > 0 else 0.0

# 모델2 정확도
total2 = len(data2)
correct2 = sum(1 for x in data2 if x.get("correct", False))
accuracy2 = correct2 / total2 if total2 > 0 else 0.0

# 모델3 정확도
total3 = len(data3)
correct3 = sum(1 for x in data3 if x.get("correct", False))
accuracy3 = correct3 / total3 if total3 > 0 else 0.0

# 막대 그래프 그리기
models = [MODEL1_NAME, MODEL2_NAME, MODEL3_NAME]
accuracies = [accuracy1, accuracy2, accuracy3]
counts = [f"{correct1}/{total1}", f"{correct2}/{total2}", f"{correct3}/{total3}"]

bars = plt.bar(models, accuracies, color=[color1, color2, color3])

# 그래프 설정
plt.title('Overall Accuracy Comparison by Model', fontsize=15)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)  # y축 범위 설정
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))  # 백분율로 표시

# 막대 위에 정확도와 개수 표시
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{accuracies[i]:.2%}\n({counts[i]})',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('overall_accuracy_comparison.png', dpi=300)
plt.close()

# 2. 레벨별 정확도 비교 (막대 그래프)
# 두 모델의 데이터에서 레벨 정보 추출
level_stats1 = defaultdict(lambda: {"correct": 0, "total": 0})
level_stats2 = defaultdict(lambda: {"correct": 0, "total": 0})
level_stats3 = defaultdict(lambda: {"correct": 0, "total": 0})

for x in data1:
    lvl = x.get("level", 0)
    level_stats1[lvl]["total"] += 1
    if x.get("correct", False):
        level_stats1[lvl]["correct"] += 1

for x in data2:
    lvl = x.get("level", 0)
    level_stats2[lvl]["total"] += 1
    if x.get("correct", False):
        level_stats2[lvl]["correct"] += 1

for x in data3:
    lvl = x.get("level", 0)
    level_stats3[lvl]["total"] += 1
    if x.get("correct", False):
        level_stats3[lvl]["correct"] += 1

# 모든 레벨을 병합
all_levels = sorted(set(level_stats1.keys()) | set(level_stats2.keys()) | set(level_stats3.keys()))

plt.figure(figsize=(12, 6))
x = np.arange(len(all_levels))  # 레벨 위치
width = 0.25  # 막대 너비

# 정확도 계산
acc1 = [level_stats1[lvl]["correct"]/level_stats1[lvl]["total"] 
        if level_stats1[lvl]["total"] > 0 else 0 for lvl in all_levels]
acc2 = [level_stats2[lvl]["correct"]/level_stats2[lvl]["total"] 
        if level_stats2[lvl]["total"] > 0 else 0 for lvl in all_levels]
acc3 = [level_stats3[lvl]["correct"]/level_stats3[lvl]["total"] 
        if level_stats3[lvl]["total"] > 0 else 0 for lvl in all_levels]

# 막대 그래프 그리기
plt.bar(x - width, acc1, width, label=MODEL1_NAME, color=color1)
plt.bar(x, acc2, width, label=MODEL2_NAME, color=color2)
plt.bar(x + width, acc3, width, label=MODEL3_NAME, color=color3)

# 그래프 설정
plt.title('Accuracy Comparison by Level', fontsize=15)
plt.xlabel('Problem Level', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(x, [f'Level {lvl}' for lvl in all_levels])
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

# 막대 위에 정확도 표시
for i, v in enumerate(acc1):
    plt.text(i - width, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(acc2):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(acc3):
    plt.text(i + width, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('level_accuracy_comparison.png', dpi=300)
plt.close()

# 3. 주제별 정확도 비교 (가로 막대 그래프)
# 두 모델의 데이터에서 주제 정보 추출
subject_stats1 = defaultdict(lambda: {"correct": 0, "total": 0})
subject_stats2 = defaultdict(lambda: {"correct": 0, "total": 0})
subject_stats3 = defaultdict(lambda: {"correct": 0, "total": 0})

for x in data1:
    subj = x.get("subject", "Unknown")
    subject_stats1[subj]["total"] += 1
    if x.get("correct", False):
        subject_stats1[subj]["correct"] += 1

for x in data2:
    subj = x.get("subject", "Unknown")
    subject_stats2[subj]["total"] += 1
    if x.get("correct", False):
        subject_stats2[subj]["correct"] += 1

for x in data3:
    subj = x.get("subject", "Unknown")
    subject_stats3[subj]["total"] += 1
    if x.get("correct", False):
        subject_stats3[subj]["correct"] += 1

# 모든 주제를 병합하고 정렬
all_subjects = sorted(set(subject_stats1.keys()) | set(subject_stats2.keys()) | set(subject_stats3.keys()))

# 주제가 너무 많으면 상위 10개만 선택
if len(all_subjects) > 10:
    # 총 문제 수가 많은 순으로 정렬
    subject_counts = [(subj, subject_stats1[subj]["total"] + subject_stats2[subj]["total"] + subject_stats3[subj]["total"]) 
                    for subj in all_subjects]
    subject_counts.sort(key=lambda x: x[1], reverse=True)
    all_subjects = [item[0] for item in subject_counts[:10]]

plt.figure(figsize=(12, max(6, len(all_subjects) * 0.5)))
y = np.arange(len(all_subjects))  # 주제 위치
height = 0.25  # 막대 높이

# 정확도 계산
acc1 = [subject_stats1[subj]["correct"]/subject_stats1[subj]["total"] 
        if subject_stats1[subj]["total"] > 0 else 0 for subj in all_subjects]
acc2 = [subject_stats2[subj]["correct"]/subject_stats2[subj]["total"] 
        if subject_stats2[subj]["total"] > 0 else 0 for subj in all_subjects]
acc3 = [subject_stats3[subj]["correct"]/subject_stats3[subj]["total"] 
        if subject_stats3[subj]["total"] > 0 else 0 for subj in all_subjects]

# 막대 그래프 그리기 (가로)
plt.barh(y - height, acc1, height, label=MODEL1_NAME, color=color1)
plt.barh(y, acc2, height, label=MODEL2_NAME, color=color2)
plt.barh(y + height, acc3, height, label=MODEL3_NAME, color=color3)

# 그래프 설정
plt.title('Subject Accuracy Comparison', fontsize=15)
plt.xlabel('Accuracy', fontsize=12)
plt.yticks(y, all_subjects)
plt.xlim(0, 1.0)
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))

# 막대 옆에 정확도 표시
for i, v in enumerate(acc1):
    plt.text(v + 0.02, i - height, f'{v:.2%}', va='center', fontsize=9)
for i, v in enumerate(acc2):
    plt.text(v + 0.02, i, f'{v:.2%}', va='center', fontsize=9)
for i, v in enumerate(acc3):
    plt.text(v + 0.02, i + height, f'{v:.2%}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('subject_accuracy_comparison.png', dpi=300)
plt.close()

# 4. 종합 정보 그래프 (파이 차트)
plt.figure(figsize=(15, 10))

# 첫 번째 파이 차트 (모델1 주제별 정확/오답 비율)
plt.subplot(2, 2, 1)
subject_correct = [subject_stats1[subj]["correct"] for subj in all_subjects]
subject_incorrect = [subject_stats1[subj]["total"] - subject_stats1[subj]["correct"] for subj in all_subjects]
plt.pie(subject_correct, labels=all_subjects, autopct='%1.1f%%', 
       startangle=90, colors=plt.cm.Blues(np.linspace(0.5, 0.8, len(all_subjects))))
plt.title(f'{MODEL1_NAME} Subject Correct Rate', fontsize=12)

# 두 번째 파이 차트 (모델2 주제별 정확/오답 비율)
plt.subplot(2, 2, 2)
subject_correct = [subject_stats2[subj]["correct"] for subj in all_subjects]
subject_incorrect = [subject_stats2[subj]["total"] - subject_stats2[subj]["correct"] for subj in all_subjects]
plt.pie(subject_correct, labels=all_subjects, autopct='%1.1f%%', 
       startangle=90, colors=plt.cm.Reds(np.linspace(0.5, 0.8, len(all_subjects))))
plt.title(f'{MODEL2_NAME} Subject Correct Rate', fontsize=12)

# 셋째 파이 차트 (모델3 주제별 정확/오답 비율)
plt.subplot(2, 2, 3)
subject_correct = [subject_stats3[subj]["correct"] for subj in all_subjects]
subject_incorrect = [subject_stats3[subj]["total"] - subject_stats3[subj]["correct"] for subj in all_subjects]
plt.pie(subject_correct, labels=all_subjects, autopct='%1.1f%%', 
       startangle=90, colors=plt.cm.Greens(np.linspace(0.5, 0.8, len(all_subjects))))
plt.title(f'{MODEL3_NAME} Subject Correct Rate', fontsize=12)


plt.tight_layout()
plt.savefig('detailed_comparison.png', dpi=300)
plt.close()

print("그래프가 성공적으로 생성되었습니다:")
print("1. overall_accuracy_comparison.png - 전체 정확도 비교")
print("2. level_accuracy_comparison.png - 레벨별 정확도 비교")
print("3. subject_accuracy_comparison.png - 주제별 정확도 비교")
print("4. detailed_comparison.png - 종합 정보 파이 차트")
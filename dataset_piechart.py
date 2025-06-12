import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 및 샘플 수
math_datasets = {
    "CLEVR-Math": 1000,
    "GEOS": 200,
    "Geometry3K": 2000,
    "GeoQA+": 2000,
    "UniGeo": 2000,
    "Multimath-300K": 27000,
    "Super-CLEVR": 1300
}

labels = list(math_datasets.keys())
sizes = list(math_datasets.values())
colors = plt.cm.Pastel1.colors

# 파이 차트 그리기
fig, ax = plt.subplots(figsize=(10, 8))
wedges, _ = ax.pie(
    sizes,
    labels=None,
    startangle=90,
    colors=colors
)

# 퍼센트/샘플 수 텍스트를 wedge 밖으로 배치
total = sum(sizes)
for i, w in enumerate(wedges):
    ang = (w.theta2 + w.theta1) / 2        # 해당 조각의 중앙 각도
    x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))

    txt = f'{sizes[i] / total * 100:.1f}%\n({sizes[i]})'
    ax.annotate(
        txt,
        xy=(x * 0.9,  y * 0.9),            # 화살표 시작점(파이 안쪽)
        xytext=(x * 1.15, y * 1.15),       # 텍스트 위치(파이 밖)
        ha='center', va='center',
        arrowprops=dict(arrowstyle='-', connectionstyle='arc3')
    )

# legend, 제목, 저장 부분은 유지
plt.legend(wedges, labels, title="Datasets",
           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.title('Math Dataset Distribution')
ax.set_aspect('equal')
plt.tight_layout()

# 이미지 저장
output_path_clean = "math_dataset_piechart_clean.png"
plt.savefig(output_path_clean)
plt.close()

output_path_clean

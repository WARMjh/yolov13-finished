import matplotlib.pyplot as plt
import numpy as np

# --------------------- 你的数据（8:2划分，固定不变）---------------------
defect_names = [
    '龟裂', '夹杂', '斑块',
    '麻点表面', '氧化铁皮', '划痕'
]
train = [240, 240, 240, 240, 240, 240]  # 每类训练集
val   = [60, 60, 60, 60, 60, 60]       # 每类验证集

# --------------------- 绘图设置 ---------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
x = np.arange(len(defect_names))
width = 0.35

# 创建画布
plt.figure(figsize=(10, 5), dpi=300)

# 画柱状图
plt.bar(x - width/2, train, width, label='训练集 (Train)', color='#4472C4')
plt.bar(x + width/2, val, width, label='验证集 (Val)', color='#ED7D31')

# 标签与标题
plt.xlabel('缺陷类别')
plt.ylabel('样本数量')
plt.title('NEU-DET 数据集 6 类缺陷训练集与验证集样本分布（8:2划分）')
plt.xticks(x, defect_names, rotation=15)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存高清图片（直接插入论文）
plt.savefig('NEU_DET_sample_distribution.png', dpi=300)
plt.show()
import cv2
import matplotlib.pyplot as plt

# ====================== 填入你两张图的路径 ======================
path_exp1 = r"D:\python\yolov13-Finished\runs\NEU\exp_neu_det3\val_batch0_pred.jpg"
path_exp4 = r"D:\python\yolov13-Finished\runs\NEU\exp2_cbam\val_batch0_pred.jpg"

# 读取图片
img1 = cv2.imread(path_exp1)
img2 = cv2.imread(path_exp4)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# ====================== 绘图：左右排列 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1行2列，左右排列，大小适中
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

ax1.imshow(img1)
ax1.set_title("(a) 基线模型 YOLOv13 (Exp1)", fontsize=10, weight='bold')
ax1.axis("off")

ax2.imshow(img2)
ax2.set_title("(b) 改进模型 CBAM (Exp2)", fontsize=10, weight='bold')
ax2.axis("off")

plt.tight_layout()

# 保存到你的项目文件夹
save_path = r"D:\python\yolov13-Finished\图4-3_左右对比图.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print("✅ 已生成：图4-3_左右对比图.png")
print("💾 路径：" + save_path)
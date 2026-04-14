import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# ====================== 你的4个实验CSV路径（按你项目填）======================
# 你可以先随便填，找不到我再帮你定位
exp_files = {
    "YOLOv13 (Baseline)": r"D:\python\yolov13-Finished\runs\NEU\exp_neu_det3\results.csv",
    "YOLOv13 + CBAM": r"D:\python\yolov13-Finished\runs\NEU\exp2_cbam\results.csv",
    "YOLOv13 + CBAM + WIoU": r"D:\python\yolov13-Finished\runs\NEU\exp3_cbam_wiou_v15\results.csv",
    "GCW-YOLOv13 (Ours)": r"D:\python\yolov13-Finished\runs\NEU\exp4_ours_final\results.csv",
}

# 学术论文标准配色（打印也清晰）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ====================== 绘图配置（论文级规范）======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# 创建【上下两张子图】：上图 Box Loss，下图 mAP@0.5
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

# ====================== 开始画图（只读取CSV）======================
for idx, (model_name, csv_path) in enumerate(exp_files.items()):
    # 自动补全路径（如果Thesis_Data没有，自动去runs/NEU找）
    if not os.path.exists(csv_path):
        csv_path = csv_path.replace("Thesis_Data", "runs/NEU")
        if not os.path.exists(csv_path):
            print(f"⚠️  跳过：{model_name} 找不到CSV，请检查路径")
            continue

    try:
        # 读取CSV，只取前150轮
        df = pd.read_csv(csv_path).dropna().head(150)

        # 自动识别列名（兼容所有YOLO版本）
        epoch = df.iloc[:, 0]
        val_box_loss = df.filter(like='val/box_loss').iloc[:, 0]
        map50 = df.filter(like='metrics/mAP50(B)').iloc[:, 0]

        # 平滑曲线（学术论文必备，消除波动）
        box_loss_smooth = savgol_filter(val_box_loss, window_length=7, polyorder=3)
        map50_smooth = savgol_filter(map50, window_length=7, polyorder=3)

        # 【上图】画 Box Loss
        ax1.plot(epoch, box_loss_smooth, color=colors[idx], label=model_name, linewidth=2)

        # 【下图】画 mAP@0.5
        ax2.plot(epoch, map50_smooth, color=colors[idx], label=model_name, linewidth=2)

    except Exception as e:
        print(f"❌ 处理 {model_name} 出错：{e}")

# ====================== 【上图】Box Loss 格式规范 ======================
ax1.set_title('(a) 150轮迭代边界框损失（Box Loss）收敛曲线对比', fontweight='bold')
ax1.set_xlabel('迭代轮数 Epoch')
ax1.set_ylabel('验证集边界框损失 Val Box Loss')
ax1.legend(fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_xlim(0, 150)

# ====================== 【下图】mAP@0.5 格式规范 ======================
ax2.set_title('(b) 150轮迭代平均精度（mAP@0.5）收敛曲线对比', fontweight='bold')
ax2.set_xlabel('迭代轮数 Epoch')
ax2.set_ylabel('验证集平均精度 mAP@0.5')
ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_xlim(0, 150)

# ====================== 保存高清图（直接插入论文）======================
plt.savefig('图4-1 收敛曲线对比图.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 成功！收敛曲线图已保存在项目根目录！")
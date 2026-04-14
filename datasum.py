from ultralytics import YOLO
import os

# 按你的目录树填写权重路径
model_paths = {
    "YOLOv13 (Baseline)": "./Thesis_Data/Exp1_Baseline/best.pt",
    "YOLOv13 + CBAM": "./Thesis_Data/Exp2_CBAM/best.pt",
    "YOLOv13 + CBAM + WIoU": "./Thesis_Data/Exp3_CBAM_WIoU/best.pt",
    "YOLOv13-Ours (最终改进)": "./Thesis_Data/Exp4_Ours_Final/best.pt",
}

# 遍历模型提取核心指标
print("=== 消融实验核心指标汇总 ===")
print(f"{'模型配置':<30} {'mAP@0.5':<10} {'参数量(Params/M)':<20} {'计算量(FLOPs/G)':<20}")
print("-"*80)

for model_name, weight_path in model_paths.items():
    if not os.path.exists(weight_path):
        print(f"{model_name}: 权重文件不存在，请检查路径")
        continue
    # 加载模型+验证获取mAP
    model = YOLO(weight_path)
    metrics = model.val(data="./datasets/NEU-DET/data.yaml", plots=False)
    map50 = round(metrics.box.map50, 3)
    # 提取参数量、计算量
    params = round(model.model.info()[0]/1e6, 2)  # 转换为M单位
    flops = round(model.model.info()[1]/1e9, 2)   # 转换为G单位
    # 输出结果
    print(f"{model_name:<30} {map50:<10} {params:<20} {flops:<20}")
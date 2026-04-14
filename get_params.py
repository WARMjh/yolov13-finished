from ultralytics import YOLO

# 严格对应你截图中的文件夹路径
weights_paths = [
    r"D:\python\yolov13-Finished\Thesis_Data\Exp1_Baseline\best.pt",
    r"D:\python\yolov13-Finished\Thesis_Data\Exp2_CBAM\best.pt",
    r"D:\python\yolov13-Finished\Thesis_Data\Exp3_CBAM_WIoU\best.pt",  # 若你用的是其他版本，请自行修改文件夹名
    r"D:\python\yolov13-Finished\Thesis_Data\Exp4_Ours_Final\best.pt"
]

print("开始读取各消融阶段的模型参数量...\n")

for i, path in enumerate(weights_paths):
    try:
        print(f"正在分析第 {i+1} 组模型: {path}")
        model = YOLO(path)
        # model.info() 会自动在终端打印该模型的 Parameters 和 GFLOPs
        model.info()
        print("-" * 60)
    except Exception as e:
        print(f"读取 {path} 失败，请检查文件是否存在。报错: {e}")
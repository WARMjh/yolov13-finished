from ultralytics import YOLO
import os


def run_inference():
    # 1. 锁定你千辛万苦炼出来的“仙丹”
    # 假设你解压后的 runs 文件夹就放在项目根目录。如果没放这里，你可以直接写你存放 best.pt 的绝对路径，比如 'D:/xxx/best.pt'
    # weight_path = r'D:\python\yolov13-Finished\Thesis_Data\Exp1_Baseline\best.pt'
    weight_path = r'D:\python\yolov13-Finished\Thesis_Data\Exp4_Ours_Final\best.pt'
    # 专家级防护：提前检查文件在不在，防止报错摸不着头脑
    if not os.path.exists(weight_path):
        print(f"⚠️ 警报：找不到权重文件 {weight_path}")
        print("请确认你从云端下载的 best.pt 是否解压到了正确的位置！")
        return

    print("✅ 成功加载 Ours 冠军模型！")
    model = YOLO(weight_path)

    # 2. 指定你要批改的“期末考卷”路径 (相对路径)
    test_source = r'D:\python\yolov13-Finished\datasets\NEU-DET\valid\images\scratches_3.jpg'

    # 3. 启动 CPU 本地推理
    print("🚀 开始在本地进行推理画框，请稍候（纯 CPU 跑可能需要几秒到十几秒）...")
    results = model.predict(
        source=test_source,
        save=True,  # 核心参数：必须是 True 才会把带有检测框的图片保存下来
        conf=0.45,  # 置信度阈值：只显示有 45% 以上把握的缺陷，过滤掉瞎猜的噪点
        iou=0.5,  # NMS 阈值：去除重复叠加的框
        project='runs/detect',
        name='exp4_predict'  # 预测结果保存的文件夹名称
    )

    print("\n🎉 推理大功告成！")
    print("👉 请立刻前往左侧目录：runs/detect/exp4_predict 查看你的实拍效果图！")


if __name__ == '__main__':
    run_inference()
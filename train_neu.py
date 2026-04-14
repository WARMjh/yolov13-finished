from ultralytics import YOLO

def train_model():
    # 1. 不再加载 .pt 现房，而是直接加载你刚画好的 .yaml 图纸！
    # （不用管预训练权重，框架会自动匹配尺寸相同的层级的权重，新加的 CBAM 层从零初始化）
    model = YOLO('/root/yolov13/ultralytics/cfg/models/yolov13-ours.yaml')

    results = model.train(
        data='/root/yolov13/datasets/NEU-DET/data.yaml',
        # epochs=150,  
        epochs=300,
        patience=0,         # 容错与护城河：如果连续 50 轮 mAP 不涨，自动终止，不浪费显卡寿命
        close_mosaic=15,     # 绝杀技巧：在最后 15 轮关闭强数据增强，进入纯净微调冲刺期
        imgsz=640,          
        batch=32,           
        workers=8,          
        device=0,           
        optimizer='auto',   
        amp=True,           
        project='runs/NEU', 
        name='exp5_ours_300epochs'  # 注意：实验名改为 exp3，严格区分目录！
    )

if __name__ == '__main__':
    train_model()
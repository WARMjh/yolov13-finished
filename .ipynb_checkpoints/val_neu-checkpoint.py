from ultralytics import YOLO

def validate_model():
    # 加载刚才训练出的最佳权重
    model = YOLO('/root/yolov13/runs/NEU/exp_neu_det3/weights/best.pt')
    
    # 在验证集上评估并生成所有图表
    metrics = model.val(data='/root/yolov13/datasets/NEU-DET/data.yaml')

if __name__ == '__main__':
    validate_model()
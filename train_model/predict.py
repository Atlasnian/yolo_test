from ultralytics import YOLO


model = YOLO('../train_model/runs/segment/train/weights/best.pt')
results = model.predict(source="C:/Users/15401/Desktop/yolo_test/images/11.jpg", save=True, name='predict',imgsz=[640, 640])

# 检查是否有检测结果
if len(results[0].boxes) > 0:
    first_box = results[0].boxes[0]
    print("第一个边界框信息:")
    print(f"坐标 (x1, y1, x2, y2): {first_box.xyxy[0].tolist()}")
    print(f"置信度: {first_box.conf[0].item():.4f}")
    print(f"类别ID: {int(first_box.cls[0].item())}")
    print(f"中心点和宽高 (x, y, w, h): {first_box.xywh[0].tolist()}")
else:
    print("未检测到任何目标")

from ultralytics import YOLO

if __name__ == "__main__":
    # 1. Tải mô hình YOLOv8 kích thước nano (phù hợp máy cấu hình vừa phải)
    model = YOLO("yolov8n.pt")  

    # 2. Tiến hành huấn luyện với data của bạn
    results = model.train(
        data="L:\Learning\YOLO\PlayingCards\\data.yaml",     # Đường dẫn tới file yaml vừa tạo ở Bước 2
        epochs=3,           # Số lượt huấn luyện (thử nghiệm trước với 10, 50 hoặc 100)
        imgsz=640,            # Kích thước ảnh đầu vào (chuẩn là 640)
        device=0              # Chạy bằng GPU (nếu máy không có card Nvidia, hãy xóa dòng này để chạy bằng CPU)
    )
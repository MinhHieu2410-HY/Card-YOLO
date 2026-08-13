from ultralytics import YOLO

#yolo predict model=runs/detect/train/weights/best.pt source=0 imgsz=416 device=0 show_conf=False show=True
#run real time cmd

if __name__ == "__main__":
    # 1. Load model của bạn
    model = YOLO("runs/detect/train/weights/best.pt")  

    # 2. Chạy predict với cấu hình tiết kiệm VRAM
    results = model.predict(
        source=r"L:\Learning\YOLO\PlayingCards\test\images", 
        #test 1 anh bang link
        imgsz=416,            # Giảm từ 640px xuống 416px
        save=True,
        # show=True,                      
        device=0, 
        half=True,            # giảm dung lượng VRAM tiêu thụ
        show_conf=False
    )
    
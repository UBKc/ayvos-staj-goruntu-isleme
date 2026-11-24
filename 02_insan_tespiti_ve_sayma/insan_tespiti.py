import sys
import cv2
from ultralytics import YOLO

# Video yolu komut satırından verilebilir: python dosya.py video.mp4
VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

# YOLOv8 nano modelinin başlatılması
model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(VIDEO_YOLU)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi veya kaynak sonlandı.")
        break

    # Model tahmini: Sadece insan (class 0) tespiti ve %50 güven skoru (confidence) eşiği
    results = model(frame, classes=[0], conf=0.5)

    # Tespit edilen sınır kutularının (bounding box) frame üzerine çizdirilmesi
    annotated_frame = results[0].plot()

    # İşlenmiş görüntünün ekrana yansıtılması
    cv2.imshow("YOLOv8 Insan Tespiti", annotated_frame)

    # 'q' tuşu ile çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakların serbest bırakılması
cap.release()
cv2.destroyAllWindows()
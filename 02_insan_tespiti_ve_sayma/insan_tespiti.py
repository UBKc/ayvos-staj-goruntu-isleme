import sys
import cv2
from ultralytics import YOLO

VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_YOLU)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi veya kaynak sonlandı.")
        break

    # sadece insan (class 0)
    results = model(frame, classes=[0], conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Insan Tespiti", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
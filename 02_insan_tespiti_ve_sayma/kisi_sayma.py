import sys
import cv2
from ultralytics import YOLO

VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_YOLU)

counted_ids = set()

# sayim cizgisi
line_y = 300

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi veya kaynak sonlandı.")
        break

    # sadece insan (class 0), BoT-SORT ile takip
    results = model.track(frame, classes=[0], tracker="botsort.yaml", persist=True, verbose=False)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # merkez cizgiye 10 pikselden yakinsa sayiliyor, set oldugu icin ayni kisi bir kez
            if line_y - 10 < cy < line_y + 10:
                counted_ids.add(track_id)

    cv2.putText(frame, f"Toplam Kisi: {len(counted_ids)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 + BoT-SORT Kisi Sayimi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "ortak"))
from yuz_bulaniklastirma import YuzBulaniklastirici

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "kasa1.mp4")

model = YOLO("yolov8m.pt")
yuz_bulaniklastirici = YuzBulaniklastirici()
cap = cv2.VideoCapture(VIDEO_PATH)

# sira alani, stajdaki kameraya gore cizildi
kasa_sirasi_alani = np.array([
    (697, 2),  # sol ust
    (1470, 3),  # sag ust
    (1400, 624),  # sag alt
    (675, 567)  # sol alt
], np.int32)

poly_mask = None  # %70 alan hesabi icin maske, ilk karede olusuyor

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video bitti veya okunamadı.")
        break

    if poly_mask is None:
        h, w = frame.shape[:2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [kasa_sirasi_alani], 1)

    # once yuzler bulaniklastiriliyor
    frame = yuz_bulaniklastirici.bulaniklastir(frame)

    # guven esigi 0.4
    results = model.track(frame, persist=True, classes=[0], conf=0.4, show=False)

    cv2.polylines(frame, [kasa_sirasi_alani], isClosed=True, color=(255, 0, 0), thickness=2)

    siradaki_kisi_sayisi = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # kutu goruntunun disina tasabiliyor
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw = x2 - x1
            bh = y2 - y1

            kasa_siralarina_dahil_mi = False
            oran = 0.0

            if bw > 0 and bh > 0:
                # kutunun en az %70'i alanin icindeyse sirada
                box_area = bw * bh
                intersection_area = np.sum(poly_mask[y1:y2, x1:x2])
                oran = intersection_area / box_area

                if oran >= 0.70:
                    kasa_siralarina_dahil_mi = True

                # kasa bankosu alt govdeyi kapatinca kutu kisaliyor
                # ayakta duran birinin boy/en orani genelde 2'den buyuk,
                # 1.6'dan kucukse alt govde kasanin arkasinda kalmis
                elif (bh / bw) < 1.6:
                    cx = x1 + bw // 2
                    # ayaklarin olmasi gereken yeri tahmin ediyorum (genisligin 2.5 kati asagisi)
                    tahmini_ayak_y = min(h, y1 + int(bw * 2.5))

                    iceride_mi = cv2.pointPolygonTest(kasa_sirasi_alani, (cx, tahmini_ayak_y), False)
                    if iceride_mi >= 0:
                        kasa_siralarina_dahil_mi = True

            if kasa_siralarina_dahil_mi:
                siradaki_kisi_sayisi += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # test icin oran kutunun ustunde
                cv2.putText(frame, f"%{int(oran*100)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(frame, f"Sira Bekleyenler: {siradaki_kisi_sayisi}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if siradaki_kisi_sayisi >= 2:
        cv2.putText(frame, "SIRA VAR!", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imshow("Kasa Takip", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
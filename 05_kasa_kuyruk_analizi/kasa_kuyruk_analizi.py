import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Ortak klasördeki yüz bulanıklaştırma modülünün içeri aktarılması
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "ortak"))
from yuz_bulaniklastirma import YuzBulaniklastirici

# Video yolu komut satırından verilebilir: python kasa_kuyruk_analizi.py kasa.mp4
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else str(BASE_DIR / "kasa1.mp4")

# 1. Modelleri Başlat
model = YOLO("yolov8m.pt")
yuz_bulaniklastirici = YuzBulaniklastirici()
cap = cv2.VideoCapture(VIDEO_PATH)

# Kasa önündeki sıra alanı (1920x1080 kamera görüntüsüne göre çizildi, kameraya göre değiştirilmeli)
kasa_sirasi_alani = np.array([
    (697, 2), #Sol Üst
    (1470, 3), #Sağ Üst
    (1400, 624), #Sağ Alt
    (675, 567) #Sol Alt
], np.int32)

poly_mask = None # %70 alan hesabı için kullanacağımız sanal maske

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video bitti veya okunamadı.")
        break

    # Döngünün ilk turunda video boyutlarına göre 1 kere maske oluşturulur
    if poly_mask is None:
        h, w = frame.shape[:2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [kasa_sirasi_alani], 1)

    # --- 1. GİZLİLİK: YÜZ BLURLAMA ---
    frame = yuz_bulaniklastirici.bulaniklastir(frame)

    # --- 2. İNSAN TAKİBİ (Güven skoru 0.4 eklendi) ---
    results = model.track(frame, persist=True, classes=[0], conf=0.4, show=False)

    cv2.polylines(frame, [kasa_sirasi_alani], isClosed=True, color=(255, 0, 0), thickness=2)

    siradaki_kisi_sayisi = 0

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Matris sınırlarını aşmamak için frame dışına taşan koordinatları düzelt
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Kutu genişliği ve yüksekliği
            bw = x2 - x1
            bh = y2 - y1

            kasa_siralarina_dahil_mi = False
            oran = 0.0

            if bw > 0 and bh > 0:
                # --- ÇÖZÜM 1: %70 Alan Kontrolü ---
                box_area = bw * bh
                intersection_area = np.sum(poly_mask[y1:y2, x1:x2])
                oran = intersection_area / box_area

                if oran >= 0.70:
                    kasa_siralarina_dahil_mi = True
                
                # --- ÇÖZÜM 2: Kasa Önü Kesilme Problemi (Tahmini Ayak İzdüşümü) ---
                # Ayakta duran insanın boy/en oranı genelde 2'den büyüktür.
                # Eğer oran 1.6'dan küçükse alt gövde kasa tarafından gizlenmiş demektir.
                elif (bh / bw) < 1.6:
                    cx = x1 + bw // 2
                    # Ayakların olması gereken tahmini noktayı izdüşüm ile bul (Enin 2.5 katı altı)
                    tahmini_ayak_y = min(h, y1 + int(bw * 2.5))
                    
                    iceride_mi = cv2.pointPolygonTest(kasa_sirasi_alani, (cx, tahmini_ayak_y), False)
                    if iceride_mi >= 0:
                        kasa_siralarina_dahil_mi = True

            # --- EKRANA ÇİZİM VE SAYIM ---
            if kasa_siralarina_dahil_mi:
                siradaki_kisi_sayisi += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Alanın yüzde kaçının içeride olduğunu yeşil kutunun üzerine yazdıralım (Test için iyi olur)
                cv2.putText(frame, f"%{int(oran*100)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- BİLGİ VE UYARI YAZILARI ---
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
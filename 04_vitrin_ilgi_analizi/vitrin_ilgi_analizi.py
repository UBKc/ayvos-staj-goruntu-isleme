import sys
import cv2
import numpy as np
import csv
import math
from datetime import datetime
from ultralytics import YOLO

# Video kaynağının ve YOLOv8 modelinin yüklenmesi
video_path = sys.argv[1] if len(sys.argv) > 1 else "vitrin_video.mp4" 
cap = cv2.VideoCapture(video_path)
model = YOLO("yolov8l.pt") 

# Video FPS değerinin alınması ve bekleme süresi parametrelerinin hesaplanması
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30  
WAIT_TIME_SEC = 3                            
REQUIRED_FRAMES = int(video_fps * WAIT_TIME_SEC) 

# Hafıza toleransı ve kapanma (occlusion) durumları için eşik değerlerin tanımlanması
MEMORY_TOLERANCE_FRAMES = int(video_fps * 2.0) 

# Hız filtreleme ve hareket eşiği parametrelerinin tanımlanması
STATIONARY_THRESHOLD = 10  

# Sanal vitrin ilgi alanlarının (polygon) koordinatlarının tanımlanması
zone1_pts = np.array([[270, 50], [485, 50], [485, 405], [330, 405]], np.int32)
zone1_pts = zone1_pts.reshape((-1, 1, 2))

zone2_pts = np.array([[10, 100], [250, 100], [200, 405], [10, 405]], np.int32)
zone2_pts = zone2_pts.reshape((-1, 1, 2))

# Kişi takip hafızası ve log dosyasının oluşturulması
zone_memory = {}  
log_file = "vitrin_analiz_log.csv"

with open(log_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Tarih_Saat", "Kisi_ID", "Durum", "Bolge", "Bekleme_Suresi_Sn"])

frame_count = 0

print("Analiz başlatıldı. (Hız filtreli sistem devrede...) Çıkmak için 'q' tuşuna basın.")

# Video akışının işlenmesi için ana döngünün başlatılması
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya akış kesildi.")
        break
    
    frame_count += 1

    # YOLOv8 modeli ile kare üzerindeki nesnelerin takip edilmesi
    results = model.track(frame, persist=True, tracker="botsort.yaml", classes=[0], conf=0.35, iou=0.5, verbose=False)

    # Vitrin ilgi alanlarının kare üzerine çizdirilmesi
    cv2.polylines(frame, [zone1_pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "Alan 1", (zone1_pts[0][0][0], zone1_pts[0][0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.polylines(frame, [zone2_pts], isClosed=True, color=(255, 0, 255), thickness=2)
    cv2.putText(frame, "Alan 2", (zone2_pts[0][0][0], zone2_pts[0][0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    current_ids = []

    # Kare içerisinde tespit edilen sınır kutularının kontrolü
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Sınır kutusunun ağırlık merkezinin hesaplanması
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            current_ids.append(track_id)

            # Merkez noktasının alanlar içerisinde olup olmadığının kontrolü
            in_zone1 = cv2.pointPolygonTest(zone1_pts, (cx, cy), False) >= 0
            in_zone2 = cv2.pointPolygonTest(zone2_pts, (cx, cy), False) >= 0

            # Kişinin bulunduğu aktif alanın belirlenmesi
            active_zone = None
            if in_zone1:
                active_zone = "Alan 1"
            elif in_zone2:
                active_zone = "Alan 2"

            color = (0, 255, 0) 
            label = f"ID: {track_id} (Yuruyor)" 

            if active_zone:
                # Yeni tespit edilen kişilerin hafızaya kaydedilmesi veya alan değişiminin kontrolü
                if track_id not in zone_memory or zone_memory[track_id]["zone"] != active_zone:
                    zone_memory[track_id] = {
                        "frames_looking": 0, 
                        "logged": False, 
                        "zone": active_zone,
                        "last_seen_frame": frame_count,
                        "last_cx": cx,
                        "last_cy": cy
                    }
                else:
                    # Kişinin alan içerisindeki hızının hesaplanması ve hareket kontrolü
                    mem = zone_memory[track_id]
                    frames_passed = frame_count - mem["last_seen_frame"]
                    
                    # Öklid mesafesi kullanılarak kare başına düşen piksel hızının bulunması
                    dist = math.hypot(cx - mem["last_cx"], cy - mem["last_cy"])
                    speed = dist / max(1, frames_passed) 

                    # Hız eşiği kontrolü ile kişinin inceleme süresinin artırılması
                    if speed < STATIONARY_THRESHOLD:
                        mem["frames_looking"] += 1
                        label = f"ID: {track_id} (İnceliyor)"
                    
                    # Kişinin son görülme konumu ve zaman verisinin güncellenmesi
                    mem["last_seen_frame"] = frame_count
                    mem["last_cx"] = cx
                    mem["last_cy"] = cy

                # Tamamlanan bekleme süresinin kontrol edilmesi
                frames_looking = zone_memory[track_id]["frames_looking"]
                seconds_looking = frames_looking / video_fps

                if frames_looking >= REQUIRED_FRAMES:
                    color = (0, 0, 255) 
                    label = f"Musteri: {track_id} ({seconds_looking:.1f}sn)"

                    # Loglanmamış hedeflerin CSV dosyasına yazdırılması
                    if not zone_memory[track_id]["logged"]:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(log_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([timestamp, track_id, "Vitrine Bakti", active_zone, round(seconds_looking, 2)])
                        
                        zone_memory[track_id]["logged"] = True
                else:
                    if frames_looking > 0:
                        color = (0, 165, 255) 
                        label = f"ID: {track_id} ({seconds_looking:.1f}sn)"

            else:
                # Alan dışına çıkan kişilerin hafızadan temizlenmesi
                if track_id in zone_memory:
                    del zone_memory[track_id]

            # Sınır kutusu ve merkez noktalarının kare üzerine çizdirilmesi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Ekrandan kaybolan kişilerin tolerans süresi sonunda hafızadan temizlenmesi
    stale_ids = []
    for k, v in zone_memory.items():
        if k not in current_ids:
            if (frame_count - v["last_seen_frame"]) > MEMORY_TOLERANCE_FRAMES:
                stale_ids.append(k)
            
    for k in stale_ids:
        del zone_memory[k]

    # Pencere boyutunun ölçeklendirilmesi
    disp_w = int(frame.shape[1] * 1.80)
    disp_h = int(frame.shape[0] * 1.80)
    frame = cv2.resize(frame, (disp_w, disp_h))

    # İşlenmiş görüntünün ekrana yansıtılması
    cv2.imshow("Vitrin Ilgi Analizi", frame)
    
    # Kare gecikme süresinin hesaplanması ve 'q' tuşu ile çıkış kontrolü
    delay = max(1, int(1000 / video_fps))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Kaynakların serbest bırakılması
cap.release()
cv2.destroyAllWindows()
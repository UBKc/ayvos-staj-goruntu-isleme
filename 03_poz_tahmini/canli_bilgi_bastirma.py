import sys
import cv2
import math
import time
from collections import deque
from ultralytics import YOLO

# Video yolu komut satırından verilebilir: python dosya.py video.mp4
VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "durus_gecisleri.mp4"

# Üç eklem noktası arasındaki iç açının hesaplanması
def calculate_angle(p1, p2, p3):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
    angle = abs(angle)
    return 360.0 - angle if angle > 180.0 else angle

# İki nokta arasındaki Öklid mesafesinin hesaplanması
def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

# YOLOv8 pose modelinin başlatılması
model = YOLO("yolov8n-pose.pt")

# Dosyanın bulunduğu klasör yolundan video akışının alınması
cap = cv2.VideoCapture(VIDEO_YOLU)

# Video FPS değerinin alınması ve kare gecikme süresinin hesaplanması
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = max(1, int(1000/video_fps) - 25)

# Hareket penceresi boyutu ve durum geçiş eşik değerlerinin tanımlanması
HIST_LEN = 10       
MOVE_THRESH = 0.10  
RUN_THRESH = 0.25   

# Kalça merkezi geçmişini ve durum geçmişini tutmak için kuyruk yapılarının tanımlanması
center_hist = deque(maxlen=HIST_LEN)   
state_hist = deque(maxlen=10)           

# Pozisyon canlı süre takibi değişkenlerinin tanımlanması
current_state = None
state_start_time = time.time()

# Video akışının işlenmesi için ana döngünün başlatılması
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi.")
        break

    # YOLOv8 pose modeli ile kare üzerindeki iskelet noktalarının tahmin edilmesi
    results = model(frame, verbose=False)

    # Kare içerisinde tespit edilen iskelet noktalarının kontrolü
    if results[0].keypoints is not None and results[0].keypoints.xy.nelement() > 0:
        
        # Sınır kutusu ve eklem koordinatlarının (keypoints) alınması
        keypoints_array = results[0].keypoints.xy.cpu().numpy()
        keypoints = keypoints_array[0]

        # COCO formatındaki 17 temel eklem noktasının kontrolü
        if len(keypoints) >= 17:
            
            # Sol ve sağ bacak eklem koordinatlarının atanması
            l_hip, l_knee, l_ankle = keypoints[11], keypoints[13], keypoints[15]
            r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]

            # Bacak eklem koordinatlarının geçerli veriye sahip olmasının kontrolü
            if all(l_hip) and all(l_knee) and all(l_ankle) and all(r_hip) and all(r_knee) and all(r_ankle):
                
                # Sol ve sağ diz açılarının hesaplanması ve minimum açının alınması
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                min_angle = min(l_angle, r_angle)
                
                # Yan profillerde kalça mesafesinin sıfıra yaklaşmasını önleyen alt sınırın uygulanması
                hip_width = max(calculate_distance(l_hip, r_hip), 50.0)

                # Kalça merkez noktasının hesaplanması ve geçmiş kuyruğuna eklenmesi
                center = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
                center_hist.append(center)

                # Geçmiş karelerdeki merkez yer değişiminin hesaplanması ve normalize edilmesi
                movement = 0.0
                if len(center_hist) >= 2:
                    total = sum(calculate_distance(center_hist[i], center_hist[i+1])
                                for i in range(len(center_hist)-1))
                    movement = (total / (len(center_hist)-1)) / hip_width

                # Hareket ve açı eşik değerlerine göre kişinin anlık durumunun belirlenmesi
                if len(center_hist) < HIST_LEN:
                    raw_state = "Bilinmiyor"
                elif movement > RUN_THRESH:
                    raw_state = "Kosuyor"
                elif movement > MOVE_THRESH:
                    raw_state = "Yuruyor"
                elif min_angle < 130:
                    raw_state = "Oturuyor"
                else:
                    raw_state = "Ayakta"

                # Anlık titremeleri önlemek için durum geçmişine eklenmesi ve çoğunluk oyunun alınması
                state_hist.append(raw_state)
                state = max(set(state_hist), key=state_hist.count)
                
                # Pozisyon değişiminin tespit edilmesi ve canlı süre sayacının güncellenmesi
                now = time.time()
                if current_state is None:
                    current_state = state
                    state_start_time = now
                elif state != current_state:
                    current_state = state
                    state_start_time = now

                # Mevcut pozisyonda kalınan anlık (canlı) sürenin saniye cinsinden hesaplanması
                live_duration = now - state_start_time

                # Durumlara karşılık gelen renk haritasının tanımlanması
                color_map = {
                    "Oturuyor": (0, 0, 255), 
                    "Yuruyor": (255, 165, 0),
                    "Kosuyor": (255, 0, 255),
                    "Ayakta": (0, 255, 0), 
                    "Bilinmiyor": (255, 255, 255)
                }
                color = color_map.get(state, (255, 255, 255))

                # Anlık pozisyon adının ve canlı süre bilgisinin (kronometre) kare üzerine yazdırılması
                x, y = int(l_hip[0]), int(l_hip[1] - 50)
                cv2.putText(frame, f"{state} ({live_duration:.1f}s)",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # İskelet çizimlerinin yapılması ve pencere boyutunun ölçeklendirilmesi
    annotated_frame = results[0].plot()
    disp_w = int(annotated_frame.shape[1] * 1.20)
    disp_h = int(annotated_frame.shape[0] * 1.20)
    annotated_frame = cv2.resize(annotated_frame, (disp_w, disp_h))
    
    # İşlenmiş görüntünün ekrana yansıtılması
    cv2.imshow("YOLOv8 Pose - Canli Bilgi Bastirma", annotated_frame)

    # 'q' tuşu ile çıkış kontrolü
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Kaynakların serbest bırakılması
cap.release()
cv2.destroyAllWindows()
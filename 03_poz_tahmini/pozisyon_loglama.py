import sys
import cv2
import math
import time
import csv
import json
from datetime import datetime
from collections import deque
from ultralytics import YOLO

VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "durus_gecisleri.mp4"

# p2'deki aci (kalca-diz-bilek icin diz acisi)
def calculate_angle(p1, p2, p3):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
    angle = abs(angle)
    return 360.0 - angle if angle > 180.0 else angle

def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(VIDEO_YOLU)

video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = max(1, int(1000/video_fps) - 25)

HIST_LEN = 10
MOVE_THRESH = 0.10
RUN_THRESH = 0.25

center_hist = deque(maxlen=HIST_LEN)
state_hist = deque(maxlen=10)

current_state = None
state_start_time = time.time()
log_records = []

log_file_csv = "pozisyon_loglari.csv"
log_file_json = "pozisyon_loglari.json"

with open(log_file_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Tarih_Saat", "Pozisyon", "Sure_Saniye"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi.")
        break

    results = model(frame, verbose=False)

    if results[0].keypoints is not None and results[0].keypoints.xy.nelement() > 0:

        keypoints_array = results[0].keypoints.xy.cpu().numpy()
        keypoints = keypoints_array[0]

        if len(keypoints) >= 17:

            l_hip, l_knee, l_ankle = keypoints[11], keypoints[13], keypoints[15]
            r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]

            if all(l_hip) and all(l_knee) and all(l_ankle) and all(r_hip) and all(r_knee) and all(r_ankle):

                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                min_angle = min(l_angle, r_angle)

                # yandan bakan kiside kalca genisligi sifira yaklasiyor, alt sinir 50
                hip_width = max(calculate_distance(l_hip, r_hip), 50.0)

                center = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
                center_hist.append(center)

                # son karelerdeki ortalama yer degistirme, kalca genisligine bolunuyor
                movement = 0.0
                if len(center_hist) >= 2:
                    total = sum(calculate_distance(center_hist[i], center_hist[i+1])
                                for i in range(len(center_hist)-1))
                    movement = (total / (len(center_hist)-1)) / hip_width

                # diz acisi 130'un altindaysa oturuyor
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

                # tek karelik titremeler olmasin diye son karelerin cogunluk oyu
                state_hist.append(raw_state)
                state = max(set(state_hist), key=state_hist.count)

                # durum degisince onceki durumun suresi loga yaziliyor
                now = time.time()
                if current_state is None:
                    current_state = state
                    state_start_time = now
                elif state != current_state:
                    duration = round(now - state_start_time, 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(log_file_csv, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, current_state, duration])

                    log_records.append({
                        "tarih_saat": timestamp,
                        "pozisyon": current_state,
                        "sure_saniye": duration
                    })

                    current_state = state
                    state_start_time = now

                color_map = {
                    "Oturuyor": (0, 0, 255),
                    "Yuruyor": (255, 165, 0),
                    "Kosuyor": (255, 0, 255),
                    "Ayakta": (0, 255, 0),
                    "Bilinmiyor": (255, 255, 255)
                }
                color = color_map.get(state, (255, 255, 255))

                x, y = int(l_hip[0]), int(l_hip[1] - 50)
                cv2.putText(frame, f"{state} (Aci:{int(min_angle)}, Hareket:{movement:.2f})",
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    annotated_frame = results[0].plot()
    disp_w = int(annotated_frame.shape[1] * 1.20)
    disp_h = int(annotated_frame.shape[0] * 1.20)
    annotated_frame = cv2.resize(annotated_frame, (disp_w, disp_h))

    cv2.imshow("YOLOv8 Pose - Durus Tespiti", annotated_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# video bitince son durumun suresi de yaziliyor
if current_state is not None:
    final_duration = round(time.time() - state_start_time, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, current_state, final_duration])

    log_records.append({
        "tarih_saat": timestamp,
        "pozisyon": current_state,
        "sure_saniye": final_duration
    })

with open(log_file_json, "w", encoding="utf-8") as f:
    json.dump(log_records, f, ensure_ascii=False, indent=4)

cap.release()
cv2.destroyAllWindows()
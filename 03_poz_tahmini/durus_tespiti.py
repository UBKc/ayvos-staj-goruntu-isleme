import sys
import cv2
import math
import numpy as np
from collections import deque
from ultralytics import YOLO

# kullanim: python durus_tespiti.py oturma.mp4
# stajda ayakta durma, oturma, yurume ve kosma videolariyla ayri ayri denedim
VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "kosma.mp4"

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

# pencere olcegi, buyuk videolarda 0.33 gibi verilebilir
GOSTERIM_OLCEGI = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

HIST_LEN = 10
MOVE_THRESH = 0.08
RUN_THRESH = 0.25

center_hist = deque(maxlen=HIST_LEN)
state_hist = deque(maxlen=5)

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

                movement = 0.0
                if len(center_hist) >= 2:
                    # son karelerdeki ortalama yer degistirme, kalca genisligine bolunuyor
                    total = sum(calculate_distance(center_hist[i], center_hist[i+1])
                                for i in range(len(center_hist)-1))
                    movement = (total / (len(center_hist)-1)) / hip_width

                # diz acisi 130'un altindaysa oturuyor
                if len(center_hist) < HIST_LEN:
                    state = "Bilinmiyor"
                elif movement > RUN_THRESH:
                    state = "Kosuyor"
                elif movement > MOVE_THRESH:
                    state = "Yuruyor"
                elif min_angle < 130:
                    state = "Oturuyor"
                else:
                    state = "Ayakta"

                # tek karelik titremeler olmasin diye son karelerin cogunluk oyu
                state_hist.append(state)
                state = max(set(state_hist), key=state_hist.count)

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
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    annotated_frame = results[0].plot()
    disp_w = int(annotated_frame.shape[1] * GOSTERIM_OLCEGI)
    disp_h = int(annotated_frame.shape[0] * GOSTERIM_OLCEGI)
    annotated_frame = cv2.resize(annotated_frame, (disp_w, disp_h))

    cv2.imshow("YOLOv8 Pose - Durus Tespiti", annotated_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
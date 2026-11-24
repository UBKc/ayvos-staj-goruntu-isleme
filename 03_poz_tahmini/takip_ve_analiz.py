import sys
import cv2
import math
import time
from collections import deque, defaultdict
from ultralytics import YOLO

VIDEO_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

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
delay = max(1, int(1000 / video_fps) - 25)

# model her 2 karede bir calisiyor, fps icin
FRAME_SKIP = 2
frame_count = 0

# bir anlik yanlis tespitler sayilmasin
MIN_FRAMES_TO_COUNT = 8

# bu videoya gore ayarladim
RUN_THRESH = 0.50
MOVE_THRESH = 0.15

# her kisi icin ayri gecmis
center_hist_dict = defaultdict(lambda: deque(maxlen=10))
state_hist_dict = defaultdict(lambda: deque(maxlen=10))
id_seen_count = defaultdict(int)

previous_active_ids = set()
exited_person_count = 0

# atlanan karelerde son sonuclar ciziliyor
cached_person_data = []
prev_time = time.time()
fps = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Görüntü akışı kesildi.")
        break

    frame_count += 1

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    if frame_count % FRAME_SKIP == 0:
        cached_person_data.clear()
        current_frame_ids = set()

        results = model.track(frame, persist=True, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_array = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

            for idx, person_id in enumerate(track_ids):
                current_frame_ids.add(person_id)
                id_seen_count[person_id] += 1

                if len(keypoints_array) > idx:
                    keypoints = keypoints_array[idx]
                    if len(keypoints) >= 17:
                        l_hip, l_knee, l_ankle = keypoints[11], keypoints[13], keypoints[15]
                        r_hip, r_knee, r_ankle = keypoints[12], keypoints[14], keypoints[16]

                        if all(l_hip) and all(l_knee) and all(l_ankle) and all(r_hip) and all(r_knee) and all(r_ankle):
                            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                            r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                            min_angle = min(l_angle, r_angle)

                            # yandan bakan kiside kalca genisligi sifira yaklasiyor, alt sinir 50
                            hip_width = max(calculate_distance(l_hip, r_hip), 50.0)

                            center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                            center_hist = center_hist_dict[person_id]
                            center_hist.append(center)

                            # son karelerdeki ortalama yer degistirme, kalca genisligine bolunuyor
                            movement = 0.0
                            if len(center_hist) >= 2:
                                total = sum(calculate_distance(center_hist[i], center_hist[i+1]) for i in range(len(center_hist)-1))
                                movement = (total / (len(center_hist)-1)) / hip_width

                            # diz acisi 130'un altindaysa oturuyor
                            if len(center_hist) < 5:
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
                            state_hist = state_hist_dict[person_id]
                            state_hist.append(raw_state)
                            state = max(set(state_hist), key=state_hist.count)

                            cached_person_data.append({
                                "id": person_id,
                                "state": state,
                                "pos": (int(l_hip[0]), int(l_hip[1] - 40))
                            })

        # onceki karede olup bu karede olmayanlar cikmis sayiliyor
        exited_ids = previous_active_ids - current_frame_ids
        for ex_id in exited_ids:
            if id_seen_count[ex_id] >= MIN_FRAMES_TO_COUNT:
                exited_person_count += 1

            if ex_id in id_seen_count: del id_seen_count[ex_id]
            if ex_id in center_hist_dict: del center_hist_dict[ex_id]
            if ex_id in state_hist_dict: del state_hist_dict[ex_id]

        previous_active_ids = current_frame_ids.copy()

    for pdata in cached_person_data:
        p_id = pdata["id"]
        p_state = pdata["state"]
        x, y = pdata["pos"]

        color_map = {
            "Oturuyor": (0, 0, 255),
            "Yuruyor": (255, 165, 0),
            "Kosuyor": (255, 0, 255),
            "Ayakta": (0, 255, 0),
            "Bilinmiyor": (255, 255, 255)
        }
        color = color_map.get(p_state, (255, 255, 255))

        cv2.putText(frame, f"ID:{p_id} {p_state}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.rectangle(frame, (10, 10), (380, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Anlik Kisi Sayisi: {len(cached_person_data)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Cikan Toplam Kisi: {exited_person_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} (Atlama: {FRAME_SKIP}x)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    annotated_frame = frame
    disp_w = int(annotated_frame.shape[1] * 1.80)
    disp_h = int(annotated_frame.shape[0] * 1.80)
    annotated_frame = cv2.resize(annotated_frame, (disp_w, disp_h))

    cv2.imshow("YOLOv8 - Gercek Zamanli Kisi Sayimi ve Poz Takibi", annotated_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
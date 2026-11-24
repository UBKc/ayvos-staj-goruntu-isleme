import sys
import cv2
import torch
import threading
import time
from transformers import pipeline
from PIL import Image

# 1. Model Kurulumu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Model yükleniyor, lütfen bekleyin...")
detector = pipeline(model="IDEA-Research/grounding-dino-tiny", task="zero-shot-object-detection", device=device)

# 2. Paylaşılan Değişkenler (Thread'ler arası iletişim)
shared_frame = None
shared_results = []
is_analyzing = False
is_running = True

# 3. Arka Plan İşçisi (Worker Thread)
def ai_worker():
    global shared_frame, shared_results, is_analyzing, is_running
    while is_running:
        # Arka planda işlenecek güncel bir kare varsa ve yapay zeka şu an müsaitse:
        if shared_frame is not None and not is_analyzing:
            is_analyzing = True
            
            # Çakışmayı önlemek için o anki karenin kopyasını alıyoruz
            frame_copy = shared_frame.copy()
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Ağır Transformer modelini burada çalıştırıyoruz
            try:
                results = detector(
                    pil_image,
                    candidate_labels=["cigarette", "smoking"], 
                    box_threshold=0.15,
                    text_threshold=0.15
                )
                shared_results = results
            except Exception:
                pass
                
            is_analyzing = False
        time.sleep(0.01) # İşlemciyi yormamak için çok hafif bir bekleme

# Arka plan iş parçacığını başlat
ai_thread = threading.Thread(target=ai_worker)
ai_thread.start()

# 4. Video Kurulumu
# Video yolu komut satırından verilebilir: python sigara_tespiti.py video.mp4
video_path = sys.argv[1] if len(sys.argv) > 1 else "sigara.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("HATA: Video bulunamadı!")
    is_running = False
    ai_thread.join()
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
# Akıcılık için videonun doğal bekleme süresini hesaplıyoruz (Milisaniye cinsinden)
delay = max(1, int(1000 / fps))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def on_trackbar(val):
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if abs(val - current_pos) > 5:
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)

cv2.namedWindow("Grounding DINO - Threaded")
cv2.createTrackbar("Zaman (Kare)", "Grounding DINO - Threaded", 0, total_frames - 1, on_trackbar)

print("Asenkron analiz başlatıldı. Video tamamen akıcı oynatılacak.")

# 5. Ana Video Akışı (Main Thread)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos("Zaman (Kare)", "Grounding DINO - Threaded", current_frame)
    
    # Arka plandaki zekaya güncel kareyi yolla (Video o sırada akmaya devam eder)
    shared_frame = frame

    # Ekrana arka planın bulduğu en son kutuları çiz
    for result in shared_results:
        box = result["box"]
        label = result["label"]
        score = result["score"]
        
        cv2.rectangle(frame, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 0, 255), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (box["xmin"], box["ymin"] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    current_sec = int(current_frame / fps)
    mins = current_sec // 60
    secs = current_sec % 60
    
    disp_w = int(frame.shape[1] * 0.7)
    disp_h = int(frame.shape[0] * 0.7)
    resized_frame = cv2.resize(frame, (disp_w, disp_h))
    
    cv2.putText(resized_frame, f"Sure: {mins:02d}:{secs:02d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Sol üst köşeye arka plan işleminin durumunu yazdıran durum göstergesi
    status_color = (0, 255, 0) if is_analyzing else (255, 255, 255)
    status_text = "Analiz Ediliyor..." if is_analyzing else "Hazir"
    cv2.putText(resized_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("Grounding DINO - Threaded", resized_frame)

    # Videonun FPS hızını bozmamak için 'delay' süresi kadar bekle
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Döngü kırıldığında threadleri güvenli bir şekilde kapat
is_running = False
ai_thread.join()
cap.release()
cv2.destroyAllWindows()
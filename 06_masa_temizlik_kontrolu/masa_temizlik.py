import sys
from pathlib import Path
import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent

VIDEOS_DIR = SCRIPT_DIR / "videos" if (SCRIPT_DIR / "videos").exists() else SCRIPT_DIR.parent / "videos"
MODELS_DIR = SCRIPT_DIR / "models" if (SCRIPT_DIR / "models").exists() else SCRIPT_DIR.parent / "models"
OUTPUTS_DIR = SCRIPT_DIR / "outputs" if (SCRIPT_DIR / "outputs").exists() else SCRIPT_DIR.parent / "outputs"

# butun cekirdekleri kullansin
MAX_THREADS = cv2.getNumberOfCPUs()
cv2.setNumThreads(MAX_THREADS)
torch.set_num_threads(MAX_THREADS)

sys.path.insert(0, str(SCRIPT_DIR.parent / "ortak"))
from yuz_bulaniklastirma import YuzBulaniklastirici

DEVICE = 0 if torch.cuda.is_available() else "cpu"

TABLE_LABEL = "dining table"
PERSON_LABEL = "person"
BOTTLE_LABEL = "bottle"
CUP_LABEL = "cup"
WINEGLASS_LABEL = "wine glass"

TABLE_CONF_THRESHOLD = 0.2
PERSON_CONF_THRESHOLD = 0.20
ESYA_CONF_THRESHOLD = 0.05
YOLO_IOU_THRESHOLD = 0.50
INFERENCE_IMGSZ = 1280

TABLE_SHRINK_RATIO = 0.12
MATCH_DIST_MULTIPLIER = 0.9

GRACE_PERIOD_SECONDS = 6.0
DIRTY_CONFIRM_SECONDS = 3.0

MASA_KESIF_ONAY_SANIYE = 1.5
MASA_KESIF_IOU_ESIK = 0.30

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_BOS = (50, 205, 50)
COLOR_KIRLI = (0, 0, 255)
COLOR_ID = (0, 0, 255)

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def kutu_iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def gelismis_kutu_kumeleme(kutular: list, iou_esik=0.35) -> list:
    if not kutular:
        return []
    kumeler = []
    for kutu in kutular:
        eklendi = False
        for kume in kumeler:
            if any(kutu_iou(kutu, k_kutu) > iou_esik for k_kutu in kume):
                kume.append(kutu)
                eklendi = True
                break
        if not eklendi:
            kumeler.append([kutu])
    birlesik_kutular = []
    for kume in kumeler:
        arr = np.array(kume)
        ort_box = tuple(np.mean(arr, axis=0).astype(int))
        birlesik_kutular.append(ort_box)
    return birlesik_kutular


def gelismis_masa_kesfi(cap, model: YOLO, tarama_karesi=25) -> list:
    print(f"Haritalama Başladı: Masalar taranıyor ({tarama_karesi} kare)...")
    tum_masalar = []
    for _ in range(tarama_karesi):
        ok, frame = cap.read()
        if not ok:
            break
        results = model(
            frame, conf=TABLE_CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ,
            iou=YOLO_IOU_THRESHOLD, verbose=False, device=DEVICE
        )[0]
        names = results.names
        for box in results.boxes:
            label = names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = tuple(map(int, box.xyxy[0]))
            if label == TABLE_LABEL and conf >= TABLE_CONF_THRESHOLD:
                tum_masalar.append(xyxy)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return gelismis_kutu_kumeleme(tum_masalar, iou_esik=0.30)


class MasaKesifYoneticisi:
    def __init__(self, onay_frame: int, iou_esik: float = MASA_KESIF_IOU_ESIK):
        self.onay_frame = onay_frame
        self.iou_esik = iou_esik
        self.adaylar = []

    def guncelle(self, bu_kare_masalar: list, sabit_masalar: list) -> None:
        for kutu in bu_kare_masalar:
            if any(kutu_iou(kutu, sm) > self.iou_esik for sm in sabit_masalar):
                continue

            eslesen = None
            for aday in self.adaylar:
                ort = tuple(np.mean(np.array(aday["kutular"]), axis=0).astype(int))
                if kutu_iou(kutu, ort) > self.iou_esik:
                    eslesen = aday
                    break

            if eslesen is not None:
                eslesen["kutular"].append(kutu)
                eslesen["gorulme"] += 1
            else:
                self.adaylar.append({"kutular": [kutu], "gorulme": 1})

        terfi_edenler = [a for a in self.adaylar if a["gorulme"] >= self.onay_frame]
        for aday in terfi_edenler:
            ort_box = tuple(np.mean(np.array(aday["kutular"]), axis=0).astype(int))
            sabit_masalar.append(ort_box)

        if terfi_edenler:
            self.adaylar = [a for a in self.adaylar if a["gorulme"] < self.onay_frame]


def etiket_ciz(img: np.ndarray, text: str, org: tuple, renk: tuple, scale=1.0, bg_color=(0, 0, 0)) -> None:
    font_scale = 0.55 * scale
    thickness = max(1, int(2 * scale))
    (w, h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    y = max(y, h + 10)
    cv2.rectangle(img, (x, y - h - baseline - 4), (x + w + 8, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 4, y - 4), FONT, font_scale, renk, thickness, cv2.LINE_AA)


def olcekle(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    olcek = min(max_w / w, max_h / h, 1.0)
    if olcek >= 1.0:
        return img
    return cv2.resize(img, (int(w * olcek), int(h * olcek)), interpolation=cv2.INTER_AREA)


def bbox_kucut(bbox: tuple, shrink: float = TABLE_SHRINK_RATIO) -> tuple:
    x1, y1, x2, y2 = bbox
    dx = int((x2 - x1) * shrink)
    dy = int((y2 - y1) * shrink)
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def kareyi_isle_temizlik(frame: np.ndarray, tables: list, persons: list, esyalar: list,
                         masa_durumlari: list, fps: float, kisi_masa_takip: dict) -> None:
    w = frame.shape[1]
    scale = w / 1920.0
    line_thick = max(1, int(2 * scale))
    grace_frames = max(1, int(GRACE_PERIOD_SECONDS * fps))
    dirty_confirm_frames = max(1, int(DIRTY_CONFIRM_SECONDS * fps))

    masa_merkezleri = []
    for table in tables:
        sx1, sy1, sx2, sy2 = bbox_kucut(table, TABLE_SHRINK_RATIO)
        masa_merkezleri.append(((sx1 + sx2) / 2, (sy1 + sy2) / 2, math.hypot(sx2 - sx1, sy2 - sy1)))

    bu_karede_dolu_masalar = set()
    aktif_track_idler = set()

    for (kx1, ky1, kx2, ky2), track_id in persons:
        if track_id != -1:
            aktif_track_idler.add(track_id)

        # ekranda id gosterilmiyor
        cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (255, 178, 50), line_thick)

        kisi_ref_cx = (kx1 + kx2) / 2
        kisi_ref_cy = (ky1 + ky2) / 2

        en_yakin_idx, en_yakin_d = -1, float("inf")
        for i, (mx, my, kose) in enumerate(masa_merkezleri):
            d = math.hypot(kisi_ref_cx - mx, kisi_ref_cy - my)
            if d < kose * MATCH_DIST_MULTIPLIER and d < en_yakin_d:
                en_yakin_d, en_yakin_idx = d, i

        if track_id != -1:
            if en_yakin_idx != -1:
                kisi_masa_takip[track_id] = en_yakin_idx
                bu_karede_dolu_masalar.add(en_yakin_idx)
            else:
                if track_id in kisi_masa_takip:
                    del kisi_masa_takip[track_id]
        else:
            if en_yakin_idx != -1:
                bu_karede_dolu_masalar.add(en_yakin_idx)

    kayip_idler = [tid for tid in kisi_masa_takip.keys() if tid not in aktif_track_idler]
    for tid in kayip_idler:
        del kisi_masa_takip[tid]

    for i, table in enumerate(tables):
        mx1, my1, mx2, my2 = table
        durum = masa_durumlari[i]

        if i in bu_karede_dolu_masalar:
            durum["dolu"] = True
            durum["kayip_frame"] = 0
        elif durum["dolu"]:
            durum["kayip_frame"] += 1
            if durum["kayip_frame"] > grace_frames:
                durum["dolu"] = False

        esya_sayisi = 0
        for ex1, ey1, ex2, ey2 in esyalar:
            ecx, ecy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
            if mx1 < ecx < mx2 and my1 < ecy < my2:
                esya_sayisi += 1
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 0, 0), line_thick)
                cv2.putText(frame, "bardak", (ex1, ey1 - 5), FONT, 0.5 * scale, (255, 0, 0), 1)

        if (not durum["dolu"]) and esya_sayisi > 0:
            durum["kirli_confirm_frame"] += 1
        else:
            durum["kirli_confirm_frame"] = 0

        renk = COLOR_BOS
        durum_metni = "TEMIZ / BOS"

        if durum["dolu"]:
            durum_metni = "MUSTERI VAR"
            renk = COLOR_BOS
        elif esya_sayisi > 0 and durum["kirli_confirm_frame"] >= dirty_confirm_frames:
            durum_metni = "KIRLI MASA!"
            renk = COLOR_KIRLI

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), renk, line_thick)
        etiket_ciz(frame, durum_metni, (mx1, my1 - 8), (255, 255, 255), scale=scale, bg_color=renk)


def main() -> None:
    print(f"Sistemdeki Çekirdek Sayısı ({MAX_THREADS}) optimizasyon için tahsis edildi.")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # models klasorunde yoksa ultralytics indiriyor
    model_yolu = MODELS_DIR / "yolov8x.pt"
    model = YOLO(str(model_yolu) if model_yolu.exists() else "yolov8x.pt")
    # yuz modeli models klasorunde varsa o, yoksa haar cascade
    yuz_model_yolu = MODELS_DIR / "face.pt"
    yuz_bulaniklastirici = YuzBulaniklastirici(
        model_yolu=str(yuz_model_yolu) if yuz_model_yolu.exists() else None,
        guven_esigi=0.10,
    )

    if len(sys.argv) > 1:
        videolar = [sys.argv[1]]
    else:
        videolar = ["Masa_Temizlik0.mp4", "Masa_Temizlik1.mp4", "Masa_Temizlik2.mp4"]

    for video_name in videolar:
        video_path = VIDEOS_DIR / video_name

        if not video_path.exists():
            print(f"\n[!] Video bulunamadı atlanıyor: {video_path}")
            continue

        print(f"\n--- {video_name} İşlenmeye Başladı ---")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[!] Video açılamadı: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sabit_masalar = gelismis_masa_kesfi(cap, model, tarama_karesi=25)
        print(f"Haritalama Tamamlandı: {len(sabit_masalar)} adet sabit masa.")

        masa_durumlari = [{"dolu": False, "kayip_frame": 0, "kirli_confirm_frame": 0} for _ in sabit_masalar]
        masa_kesif = MasaKesifYoneticisi(onay_frame=max(1, int(MASA_KESIF_ONAY_SANIYE * fps)))

        kisi_masa_takip = {}

        output_path = OUTPUTS_DIR / f"islenmis_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

        window_name = f"Temizlik Analizi - {video_name}"
        w, h = frame_size
        olcek = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(w * olcek), int(h * olcek))



        kare_sayaci = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            kare_sayaci += 1

            if kare_sayaci % 50 == 0:
                print(f"İşlenen Kare: {kare_sayaci} / {toplam_kare}", flush=True)

            results = model.track(
                frame, conf=0.05, imgsz=INFERENCE_IMGSZ,
                classes=[0, 39, 40, 41, 60], persist=True, tracker="bytetrack.yaml", verbose=False, device=DEVICE
            )[0]

            persons, esyalar, masa_adaylari = [], [], []
            names = results.names

            ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else None

            for i, box in enumerate(results.boxes):
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                xyxy = tuple(map(int, box.xyxy[0]))

                if label == PERSON_LABEL and conf > PERSON_CONF_THRESHOLD:
                    track_id = int(ids[i]) if ids is not None else -1
                    persons.append((xyxy, track_id))
                elif label in [BOTTLE_LABEL, CUP_LABEL, WINEGLASS_LABEL] and conf > ESYA_CONF_THRESHOLD:
                    esyalar.append(xyxy)
                elif label == TABLE_LABEL and conf >= TABLE_CONF_THRESHOLD:
                    masa_adaylari.append(xyxy)

            frame = yuz_bulaniklastirici.bulaniklastir(frame, [kutu for kutu, _ in persons])

            masa_kesif.guncelle(masa_adaylari, sabit_masalar)
            while len(masa_durumlari) < len(sabit_masalar):
                masa_durumlari.append({"dolu": False, "kayip_frame": 0, "kirli_confirm_frame": 0})

            kareyi_isle_temizlik(frame, sabit_masalar, persons, esyalar, masa_durumlari, fps, kisi_masa_takip)

            writer.write(frame)

            cv2.imshow(window_name, olcekle(frame, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        writer.release()

        cv2.destroyWindow(window_name)

        print(f"[✓] {video_name} tamamlandı, çıktı kaydedildi: {output_path}")

    cv2.destroyAllWindows()
    print("\n TÜM VİDEOLARIN İŞLEMİ BİTTİ.")

if __name__ == "__main__":
    main()
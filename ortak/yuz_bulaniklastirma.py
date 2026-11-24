"""Kamera görüntülerinde gizlilik için yüz bulanıklaştırma.

Varsayılan olarak OpenCV ile birlikte gelen Haar cascade yüz dedektörü kullanılır,
ek bir model dosyası gerekmez. YOLO formatında eğitilmiş bir yüz modeli varsa
model_yolu ile verilebilir, o zaman tespit o modelle yapılır.

Kullanım:
    from yuz_bulaniklastirma import YuzBulaniklastirici

    bulaniklastirici = YuzBulaniklastirici()
    frame = bulaniklastirici.bulaniklastir(frame)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

Kutu = Tuple[int, int, int, int]


class YuzBulaniklastirici:
    def __init__(
        self,
        model_yolu: Optional[str] = None,
        guven_esigi: float = 0.25,
        kenar_payi: int = 20,
        blur_kernel: Tuple[int, int] = (99, 99),
    ):
        self.guven_esigi = guven_esigi
        self.kenar_payi = kenar_payi
        self.blur_kernel = blur_kernel

        self.yolo_model = None
        if model_yolu:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_yolu)
        else:
            # Önden yüzler için frontal, yandan yüzler için profile cascade kullanılıyor
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.profil_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            # OpenCV 5 ile Haar dosyaları paketle gelmiyor, 4.x sürümü gerekiyor
            if self.cascade.empty() or self.profil_cascade.empty():
                raise RuntimeError(
                    "Haar cascade dosyası yüklenemedi. opencv-python 4.x kurulu olmalı "
                    "(pip install \"opencv-python<5\") ya da model_yolu ile bir YOLO yüz modeli verilmeli."
                )

    def yuzleri_bul(self, frame: np.ndarray) -> List[Kutu]:
        # Model varsa YOLO ile, yoksa Haar cascade ile yüz kutularının bulunması
        if self.yolo_model is not None:
            sonuc = self.yolo_model(frame, conf=self.guven_esigi, verbose=False)[0]
            return [tuple(int(v) for v in kutu) for kutu in sonuc.boxes.xyxy.cpu().numpy()]

        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        genislik = gri.shape[1]
        kutular = []

        for (x, y, w, h) in self.cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(x), int(y), int(x + w), int(y + h)))

        # Profile cascade sadece sola bakan yüzleri bulduğu için görüntü aynalanıp tekrar taranıyor
        for (x, y, w, h) in self.profil_cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(x), int(y), int(x + w), int(y + h)))
        aynali = cv2.flip(gri, 1)
        for (x, y, w, h) in self.profil_cascade.detectMultiScale(aynali, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(genislik - x - w), int(y), int(genislik - x), int(y + h)))

        return kutular

    def bulaniklastir(self, frame: np.ndarray, kisiler: Optional[List[Kutu]] = None) -> np.ndarray:
        # kisiler verilirse sadece bir insan kutusunun içine düşen yüzler bulanıklaştırılır,
        # bu sayede arka plandaki afiş veya ekranlardaki yanlış tespitler elenir
        h, w = frame.shape[:2]

        for x1, y1, x2, y2 in self.yuzleri_bul(frame):
            if kisiler:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if not any(px1 <= cx <= px2 and py1 <= cy <= py2 for px1, py1, px2, py2 in kisiler):
                    continue

            x1, y1 = max(x1 - self.kenar_payi, 0), max(y1 - self.kenar_payi, 0)
            x2, y2 = min(x2 + self.kenar_payi, w), min(y2 + self.kenar_payi, h)

            bolge = frame[y1:y2, x1:x2]
            if bolge.size > 0:
                frame[y1:y2, x1:x2] = cv2.blur(bolge, self.blur_kernel)

        return frame

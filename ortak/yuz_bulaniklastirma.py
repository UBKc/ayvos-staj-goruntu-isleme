# kamera goruntulerinde yuzleri bulaniklastiriyor
# varsayilan opencv'nin haar cascade'i, ek model dosyasi gerekmiyor
# yolo yuz modeli varsa model_yolu ile verilebilir, o daha iyi
#
#   bulaniklastirici = YuzBulaniklastirici()
#   frame = bulaniklastirici.bulaniklastir(frame)

import cv2


class YuzBulaniklastirici:
    def __init__(self, model_yolu=None, guven_esigi=0.25, kenar_payi=20, blur_kernel=(99, 99)):
        self.guven_esigi = guven_esigi
        self.kenar_payi = kenar_payi
        self.blur_kernel = blur_kernel

        self.yolo_model = None
        if model_yolu:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_yolu)
        else:
            # onden yuzler icin frontal, yandan icin profile
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.profil_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            # opencv 5'te haar dosyalari paketle gelmiyor
            if self.cascade.empty() or self.profil_cascade.empty():
                raise RuntimeError(
                    "Haar cascade dosyası yüklenemedi. opencv-python 4.x kurulu olmalı "
                    "(pip install \"opencv-python<5\") ya da model_yolu ile bir YOLO yüz modeli verilmeli."
                )

    def yuzleri_bul(self, frame):
        if self.yolo_model is not None:
            sonuc = self.yolo_model(frame, conf=self.guven_esigi, verbose=False)[0]
            return [tuple(int(v) for v in kutu) for kutu in sonuc.boxes.xyxy.cpu().numpy()]

        gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        genislik = gri.shape[1]
        kutular = []

        for (x, y, w, h) in self.cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(x), int(y), int(x + w), int(y + h)))

        # profile cascade sadece bir yone bakanlari buluyor, aynalayip tekrar bakiyorum
        for (x, y, w, h) in self.profil_cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(x), int(y), int(x + w), int(y + h)))
        aynali = cv2.flip(gri, 1)
        for (x, y, w, h) in self.profil_cascade.detectMultiScale(aynali, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)):
            kutular.append((int(genislik - x - w), int(y), int(genislik - x), int(y + h)))

        return kutular

    def bulaniklastir(self, frame, kisiler=None):
        # kisiler verilirse sadece insan kutusunun icindeki yuzler (afis, ekran gibi yanlis tespitler gitsin)
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

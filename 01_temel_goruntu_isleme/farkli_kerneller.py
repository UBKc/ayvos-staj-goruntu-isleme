import sys
import cv2
import numpy as np

# Resim yolu komut satırından verilebilir: python dosya.py resim.jpg
RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

# Görüntüyü gri tonlamalı okuyoruz
resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

# Resmi gri tonlamalı okuyup %50 oranında küçültüyoruz
resim_gri = cv2.resize(resim_gri, (0, 0), fx=0.5, fy=0.5)

# Küçülmüş resim üzerinden eşikleme yapıyoruz
ret, esiklenmis_resim = cv2.threshold(resim_gri, 127, 255, cv2.THRESH_BINARY)

# 1. Dikdörtgen (Kare) Çekirdek (np.ones ile yaptığımızın aynısı)
kernel_dikdortgen = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 2. Elips (Yuvarlak) Çekirdek (Oval nesneler ve kavisli kenarlar için daha iyi)
kernel_elips = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 3. Çapraz (Cross) Çekirdek (Sivri köşeleri ve ince çizgileri korumak için)
kernel_capraz = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# Bu farklı çekirdekleri sadece Dilate (Genişletme) üzerinde test edip farkı görüyoruz
dilate_dikdortgen = cv2.dilate(esiklenmis_resim, kernel_dikdortgen, iterations=1)
dilate_elips = cv2.dilate(esiklenmis_resim, kernel_elips, iterations=1)
dilate_capraz = cv2.dilate(esiklenmis_resim, kernel_capraz, iterations=1)

# Sonuçları ekranda gösterme
cv2.imshow("0 - Threshold (Esikleme)", esiklenmis_resim)
cv2.imshow("1 - Dilate (Dikdortgen / Kare)", dilate_dikdortgen)
cv2.imshow("2 - Dilate (Elips / Yuvarlak)", dilate_elips)
cv2.imshow("3 - Dilate (Capraz / Hac)", dilate_capraz)

# Pencerelerin kapanmaması için bekleme komutları
cv2.waitKey(0)
cv2.destroyAllWindows()
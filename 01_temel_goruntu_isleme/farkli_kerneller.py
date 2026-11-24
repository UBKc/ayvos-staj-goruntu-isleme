import sys
import cv2
import numpy as np

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

resim_gri = cv2.resize(resim_gri, (0, 0), fx=0.5, fy=0.5)

ret, esiklenmis_resim = cv2.threshold(resim_gri, 127, 255, cv2.THRESH_BINARY)

# kare (np.ones ile ayni)
kernel_dikdortgen = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# elips, kavisli kenarlarda daha iyi
kernel_elips = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# capraz, ince cizgileri koruyor
kernel_capraz = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# farki gormek icin hepsini dilate ile deniyorum
dilate_dikdortgen = cv2.dilate(esiklenmis_resim, kernel_dikdortgen, iterations=1)
dilate_elips = cv2.dilate(esiklenmis_resim, kernel_elips, iterations=1)
dilate_capraz = cv2.dilate(esiklenmis_resim, kernel_capraz, iterations=1)

cv2.imshow("0 - Threshold (Esikleme)", esiklenmis_resim)
cv2.imshow("1 - Dilate (Dikdortgen / Kare)", dilate_dikdortgen)
cv2.imshow("2 - Dilate (Elips / Yuvarlak)", dilate_elips)
cv2.imshow("3 - Dilate (Capraz / Hac)", dilate_capraz)

cv2.waitKey(0)
cv2.destroyAllWindows()
import sys
import cv2
import numpy as np

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

resim_gri = cv2.resize(resim_gri, (0, 0), fx=0.5, fy=0.5)

ret, esiklenmis_resim = cv2.threshold(resim_gri, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

# erode: beyaz alanlar inceliyor
asindirilmis_resim = cv2.erode(esiklenmis_resim, kernel, iterations=1)

# dilate: beyaz alanlar kalinlasiyor
genisletilmis_resim = cv2.dilate(esiklenmis_resim, kernel, iterations=1)

# opening: once erode sonra dilate, disaridaki gurultu gidiyor
acma_resim = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_OPEN, kernel)

# closing: once dilate sonra erode, icerideki delikler kapaniyor
kapatma_resim = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_CLOSE, kernel)

# gradient: dilate - erode, kenarlari veriyor
gradyan_resim = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_GRADIENT, kernel)

ust_sapka_resim = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_TOPHAT, kernel)

kara_sapka_resim = cv2.morphologyEx(esiklenmis_resim, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("1 - Threshold (Esikleme)", esiklenmis_resim)
cv2.imshow("2 - Erode (Asindirma)", asindirilmis_resim)
cv2.imshow("3 - Dilate (Genisletme)", genisletilmis_resim)
cv2.imshow("4 - Opening (Acma)", acma_resim)
cv2.imshow("5 - Closing (Kapatma)", kapatma_resim)
cv2.imshow("6 - Morphological Gradient (Gradyan)", gradyan_resim)
cv2.imshow("7 - Top Hat (Ust Sapka)", ust_sapka_resim)
cv2.imshow("8 - Black Hat (Kara Sapka)", kara_sapka_resim)

cv2.waitKey(0)
cv2.destroyAllWindows()
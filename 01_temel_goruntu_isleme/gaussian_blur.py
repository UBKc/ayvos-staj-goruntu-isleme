import sys
import cv2

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim = cv2.imread(RESIM_YOLU)

yeni_boyut = (500, 500)
kucuk_resim = cv2.resize(resim, yeni_boyut)

# kernel boyutu tek sayi olmali
bulanik_resim = cv2.GaussianBlur(kucuk_resim, (15, 15), 0)

cv2.imshow("Orijinal Resim", kucuk_resim)
cv2.imshow("Gaussian Blur Uygulanmis Resim", bulanik_resim)

cv2.waitKey(0)
cv2.destroyAllWindows()
import sys
import cv2
import numpy as np

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)
resim_gri = cv2.resize(resim_gri, (0, 0), fx=0.5, fy=0.5)

# once blur, gurultu kenar gibi cikmasin
resim_blur = cv2.GaussianBlur(resim_gri, (5, 5), 0)

# sobel x: dikey kenarlar
sobel_x = cv2.Sobel(resim_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)

# sobel y: yatay kenarlar
sobel_y = cv2.Sobel(resim_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel_bilesik = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

laplacian = cv2.Laplacian(resim_blur, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

canny_standart = cv2.Canny(resim_blur, 100, 200)

# esikler medyana gore
medyan = np.median(resim_blur)
alt_esik = int(max(0, (1.0 - 0.33) * medyan))
ust_esik = int(min(255, (1.0 + 0.33) * medyan))
canny_otomatik = cv2.Canny(resim_blur, alt_esik, ust_esik)

cv2.imshow("0 - Orijinal Blur", resim_blur)
cv2.imshow("1 - Sobel X (Dikey)", sobel_x)
cv2.imshow("2 - Sobel Y (Yatay)", sobel_y)
cv2.imshow("3 - Sobel Bilesik", sobel_bilesik)
cv2.imshow("4 - Laplacian", laplacian)
cv2.imshow("5 - Standart Canny", canny_standart)
cv2.imshow("6 - Otomatik Canny", canny_otomatik)

cv2.waitKey(0)
cv2.destroyAllWindows()
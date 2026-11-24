import sys
import cv2

# Resim yolu komut satırından verilebilir: python dosya.py resim.jpg
RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

# Resmi Gri Tonlamalı Olarak Okutma
resim_gri = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

# Orijinal gri resmi ekranda gösterme
cv2.imshow("Orijinal Gri Resim:", resim_gri)

# Basit Eşikleme (Binary Threshold) işlemi
ret, esiklenmis_resim = cv2.threshold(resim_gri, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Basit Esikleme (Binary - 127)", esiklenmis_resim)

# Otsu Eşikleme (Otsu's Thresholding)
ret_otsu, otsu_resim = cv2.threshold(resim_gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu Algoritmasının Hesapladığı En İdeal Eşik Değeri: {ret_otsu}")
cv2.imshow("Otsu Esikleme (Otomatik)", otsu_resim)

# Uyarlamalı Eşikleme (Adaptive Thresholding)
adaptive_resim = cv2.adaptiveThreshold(
    resim_gri, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,2,
)
cv2.imshow("Uyarlamalı Esikleme (Adaptive)", adaptive_resim)

# Pencerelerin kapanmaması için bekleme komutları
cv2.waitKey(0)
cv2.destroyAllWindows()
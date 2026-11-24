import sys
import numpy as np
import math
import cv2
import time

RESIM_YOLU = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

def gaussian_kernel_olustur(boyut, sigma):
    merkez = boyut // 2

    kernel = np.zeros((boyut, boyut))

    for x in range(boyut):
        for y in range(boyut):

            x_uzaklik = x - merkez
            y_uzaklik = y - merkez

            # gaussian formulu
            pay = -(x_uzaklik**2 + y_uzaklik**2) / (2 * (sigma**2))
            payda = 2 * math.pi * (sigma**2)

            kernel[x, y] = np.exp(pay) / payda

    # toplam 1 olsun diye normalize
    kernel = kernel / np.sum(kernel)

    return kernel

def konvolusyon_uygula(resim, kernel):
    resim_yukseklik, resim_genislik = resim.shape
    kernel_boyut = kernel.shape[0]

    pad_miktari = kernel_boyut // 2

    # kenarlarda tasmasin diye reflect padding
    cerceveli_resim = np.pad(resim, pad_miktari, mode='reflect')

    yeni_resim = np.zeros((resim_yukseklik, resim_genislik))

    for y in range(resim_yukseklik):
        for x in range(resim_genislik):

            kesit = cerceveli_resim[y : y + kernel_boyut, x : x + kernel_boyut]

            yeni_piksel = np.sum(kesit * kernel)

            yeni_resim[y, x] = yeni_piksel

    return yeni_resim

# once 5x5, sigma 1 ile kontrol
benim_filtrem = gaussian_kernel_olustur(5, 1.0)

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

print("Oluşturulan Normalleştirilmiş Kernel:")
print(benim_filtrem)
print("Yeni Matrisin Toplamı:", np.sum(benim_filtrem))


# opencv ile hiz karsilastirmasi

# fark net gorunsun diye 15x15, sigma 5
benim_filtrem = gaussian_kernel_olustur(15, 5.0)

orijinal_resim = cv2.imread(RESIM_YOLU, cv2.IMREAD_GRAYSCALE)

print("\n1. Kendi Algoritmamız Çalışıyor...")
print("(Lütfen işlem bitene kadar bekleyiniz...)")

baslangic_benim = time.time()
bulanik_resim_benim = konvolusyon_uygula(orijinal_resim, benim_filtrem)
bitis_benim = time.time()

bulanik_resim_benim = np.uint8(bulanik_resim_benim)
sure_benim = bitis_benim - baslangic_benim
print(f"-> Kendi kodumuzun çalışma süresi: {sure_benim:.4f} saniye")

print("\n2. OpenCV'nin Hazır Fonksiyonu Çalışıyor...")

baslangic_cv2 = time.time()
bulanik_resim_cv2 = cv2.GaussianBlur(orijinal_resim, (15, 15), 5.0)
bitis_cv2 = time.time()

sure_cv2 = bitis_cv2 - baslangic_cv2
print(f"-> OpenCV'nin çalışma süresi: {sure_cv2:.4f} saniye")

hiz_farki = sure_benim / sure_cv2 if sure_cv2 > 0 else 0
print(f"\nSONUÇ: OpenCV bizim Python döngülerimizden yaklaşık {hiz_farki:.0f} kat daha hızlı çalıştı!")

cv2.imshow("1 - Orijinal Resim", orijinal_resim)
cv2.imshow("2 - Kendi Yazdigimiz (15x15 Yogun Blur)", bulanik_resim_benim)
cv2.imshow("3 - OpenCV (15x15 Yogun Blur)", bulanik_resim_cv2)

print("\nİşlem tamamlandı! Pencereleri kapatmak için herhangi bir tuşa basabilirsin.")
cv2.waitKey(0)
cv2.destroyAllWindows()
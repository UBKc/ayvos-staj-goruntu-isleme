# Görev 19: Görüntü Bölütleme (Segmentation) Teorisi ve Çalışma Mantığı

Görüntü bölütleme (segmentation), nesneleri sadece sınır kutusu (bounding box) içine almak yerine piksellerini teker teker ayrıştırarak nesnenin gerçek konturlarını ve sınırlarını tespit eden yöntemdir.

# 1. Semantic vs. Instance Segmentation
* **Semantic Segmentation:** Görseldeki her pikseli sınıflara (insan, araba, yol) ayırır ancak aynı sınıftaki nesneleri ayırmaz. Ekranda kaç insan olursa olsun hepsini tek bir maske katmanı olarak etiketler.
* **Instance Segmentation:** Nesne tespiti ile anlamsal bölütlemeyi birleştirir. Aynı sınıfta olsalar bile her nesneye farklı ID atar (örneğin İnsan_1, İnsan_2) ve piksellerini bağımsız şekilde ayırır.

# 2. Mask Çıktısı (Mask Output) Mantığı
* **Ikili (Binary) Matris:** Görselle aynı boyutta (H x W) oluşturulan, nesne pikselleri için 1 (True), arka plan için 0 (False) değerini tutan matristir.
* **Poligon Koordinatları:** YOLOv8-seg gibi hızlı modellerde bellek yükünü azaltmak için piksel matrisi yerine nesne sınır noktalarının koordinatları [(x1, y1), (x2, y2)] cinsinden tutulur.
* **Görselleştirme (Overlay):** Çıkarılan maskenin orijinal görüntü üzerine saydam renk harmanı (alpha-blending) olarak uygulanıp ekrana basılmasıdır.

# 3. Görsel Senaryo Örneği ile Farklar
* **Örnek Senaryo:** Yolda yan yana duran 3 kırmızı araba olduğunu düşünelim.
  * **Bounding Box (Detection):** 3 ayrı kırmızı kutu çizer.
  * **Semantic Segmentation:** 3 arabayı tek bir kırmızı alan olarak kaplar, arabaları birbirinden ayırmaz.
  * **Instance Segmentation:** 1. Arabayı kırmızı, 2. Arabayı mavi, 3. Arabayı yeşil maske ile kaplayarak her nesneyi bağımsız piksel grubu olarak ayırır.
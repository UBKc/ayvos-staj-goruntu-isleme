# Görev 20: OpenCLIP Teorisi ve Çalışma Mantığı

OpenCLIP, OpenAI tarafından geliştirilen CLIP mimarisinin açık kaynaklı bir uygulamasıdır. Metin ve görüntü verilerini ortak bir vektör uzayında (embedding space) buluşturarak multimodal (çok modlu) öğrenme sağlar.

# 1. Kontrastif Öğrenme (Contrastive Learning)
* **İkili Kodlayıcı Yapısı:** Model bünyesinde hem Görsel Kodlayıcı (Image Encoder - ViT/ResNet) hem de Metin Kodlayıcı (Text Encoder - Transformer) barındırır.
* **Eşleştirme Mantığı:** Milyonlarca (Görsel, Metin) çifti modele beslenir. Model, doğru eşleşen çiftlerin vektör benzerliğini (kosinüs benzerliği) maksimuma çıkarırken, yanlış eşleşenleri birbirinden uzaklaştırır.

# 2. Ortak Vektör Uzayı (Shared Embedding Space)
* **Boyut İndirgeme:** Görsel ve metin verileri birbirinden bağımsız olsa da en sonunda aynı boyuttaki (örneğin 512 boyutlu) bir vektör uzayına izdüşürülür.
* **Anlamsal Yakınlık:** "Siyah bir kedi" metninin oluşturduğu vektör ile siyah kedi görselinin oluşturduğu vektör bu uzayda birbirine çok yakın koordinatlara düşer.

# 3. Sıfır-Eğitim (Zero-Shot) Sınıflandırma
* **Etiket Bağımsızlığı:** Model hiç eğitilmediği yeni bir veri setinde sınıflandırma yapabilir.
* **Sınıf Şablonu (Prompting):** Hedef sınıflar "Bir [Sınıf] fotoğrafı" şablonuna sokularak metin vektörlerine dönüştürülür. Gelen görselin vektörü hangi metin vektörüne en yakınsa görsel o sınıfa ait kabul edilir.
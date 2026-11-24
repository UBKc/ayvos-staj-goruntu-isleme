# Görev 21: DINOv2 ve Self-Supervised Vision Transformer Teorisi

Meta tarafından geliştirilen DINOv2, insan etiketlemesine (supervised labels) ihtiyaç duymadan, kendi kendine öğrenme (Self-Supervised Learning) yöntemiyle eğitilmiş gelişmiş bir Vision Transformer (ViT) mimarisidir.

# 1. Self-Supervised (Kendi Kendine Öğrenme) Mantığı
* **Etiket İhtiyacının Kaldırılması:** ImageNet gibi binlerce insan tarafından etiketlenmiş veri setleri yerine, milyonlarca ham görseli etiket olmadan kendi içinde anlamlandırır.
* **Öğrenci-Öğretmen (Student-Teacher) Mimarisi:** Aynı görselin farklı kırpılmış (crop) ve bozulmuş halleri iki ayrı ağa (Student ve Teacher) verilir. Student ağı, Teacher ağının çıktısını tahmin etmeye çalışarak resmin görsel özelliklerini kendi kendine öğrenir.

# 2. Klasik Supervised Modellerden Farkları
* **Klasik Modeller:** Sadece eğitildiği sınıflara (örneğin kedi/köpek) odaklanır; resmin arka planını veya detay piksellerini göz ardı eder.
* **DINOv2:** Resmi nesne bazlı değil, piksel ve doku düzeyinde anlar (Dense Representation). Nesnelerin sınırlarını, derinliğini ve anlamsal parçalarını etiket olmadan çıkartabilir.

# 3. Kullanım Alanları ve Avantajları
* **Fine-Tuning Gerektirmez:** Modelden alınan özelleştirilmiş vektörler (features) direkt olarak derinlik tahmini (Depth Estimation), segmentasyon veya görsel benzerlik aramada yeniden eğitilmeden kullanılabilir.
* **Güçlü Görsel Temsil:** Değişen ışık, açı veya arka plan koşullarında dahi aynı nesne parçalarını son derece tutarlı bir şekilde eşleştirebilir.
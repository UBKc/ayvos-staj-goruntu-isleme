# Pose Estimation: Top-Down vs Bottom-Up Mimari Farkları

Pose Estimation (İnsan Duruş Tahmini) modelleri, insanların eklem noktalarını (keypoints) tespit ederken iki farklı ana mimari yaklaşım kullanır. 

# 1. Top-Down (Yukarıdan Aşağıya) Yaklaşım
Bu yaklaşım, işlemi iki aşamada gerçekleştirir:
* **Önce İnsanı Bul:** Görüntüdeki her bir insanı bir nesne tespit algoritması (örn. standart YOLO bbox) ile tespit eder ve kırpar (crop).
* **Sonra Eklemleri Bul:** Kırpılan her bir insan görüntüsünün içindeki eklem noktalarını (omuz, dirsek, diz vb.) hesaplar.

**Avantajları:** Doğruluğu (accuracy) çok yüksektir çünkü model her insana odaklanacak vakti bulur.
**Dezavantajları:** Kalabalık ortamlarda çok yavaştır. Ekranda 50 kişi varsa, model 50 kere ayrı ayrı eklem hesaplaması yapmak zorunda kalır.

# 2. Bottom-Up (Aşağıdan Yukarıya) Yaklaşım
Bu yaklaşım, bütünden ziyade doğrudan noktalara odaklanır:
* **Önce Tüm Eklemleri Bul:** Görüntüdeki insanların kime ait olduğuna bakmaksızın *bütün* dirsekleri, dizleri ve omuzları aynı anda tespit eder.
* **Sonra İnsanları Oluştur:** Bulunan bu karmaşık noktaları belirli algoritmalarla (Part Affinity Fields vb.) birleştirerek "bu sağ kol şu gövdeye ait" şeklinde iskeletleri çıkarır.

**Avantajları:** Kalabalık ortamlarda çok hızlıdır (Gerçek zamanlı çalışmaya uygundur). Ekranda kaç kişi olursa olsun işlem süresi çok az değişir. YOLOv8-pose yapısı temel olarak bu pratikliğe odaklanır.
**Dezavantajları:** Doğruluk oranı Top-Down mimarilere göre bir miktar daha düşüktür; kalabalıkta eklemler birbirine karışabilir.
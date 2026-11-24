# Ayvos Stajı - Görüntü İşleme ve Mağaza Video Analizi

2026 yazında Ayvos'ta yaptığım bilgisayarlı görü stajının kodları. Temel görüntü işleme ile başlayıp YOLO ile insan tespiti, takip, poz tahmini ve mağaza/kafe kameraları için analiz modüllerine kadar ilerledim.

Kodlar staj sırasında yazdığım halleri, GitHub'a koymadan önce sadece şunları değiştirdim: dosya ve video yolları komut satırından verilebilir hale geldi, aynı kodun dört kopyası olan duruş tespiti tek dosyada toplandı, bir arkadaşımın yüz bulanıklaştırma modülü yerine kendi yazdığım `ortak/yuz_bulaniklastirma.py` kullanıldı.

## Klasörler

| Klasör | İçerik |
|---|---|
| `01_temel_goruntu_isleme/` | Gaussian blur (OpenCV ve kendi yazdığım konvolüsyon ile hız karşılaştırması), eşikleme (binary, Otsu, adaptive), morfoloji, farklı kernel şekilleri, Sobel/Laplacian/Canny, SIFT/ORB ve özellik eşleştirme |
| `02_insan_tespiti_ve_sayma/` | YOLOv8 ile insan tespiti, BoT-SORT ile takip edip çizgiden geçen kişileri sayma |
| `03_poz_tahmini/` | YOLOv8-pose ile ayakta / oturuyor / yürüyor / koşuyor tespiti, pozisyonda kalma süresi, CSV ve JSON loglama, çoklu kişide ID bazlı poz takibi |
| `04_vitrin_ilgi_analizi/` | vitrin önündeki iki bölgede 3 saniyeden uzun duran kişileri "vitrine baktı" olarak loglama |
| `05_kasa_kuyruk_analizi/` | kasa önündeki sıra alanında bekleyen kişi sayısı, 2 ve üstünde uyarı |
| `06_masa_temizlik_kontrolu/` | kafede masaları bulup müşteri var / boş / kirli (bardak, şişe kalmış) diye işaretleme |
| `07_sigara_tespiti/` | Grounding DINO ile sıfır-eğitim (zero-shot) sigara tespiti, model arka planda ayrı thread'de çalışıyor |
| `ortak/` | yüz bulanıklaştırma modülü |
| `notlar/` | staj görevlerinde hazırladığım teori notları: CNN, görüntü işleme görevleri, YOLO, pose estimation mimarileri, segmentasyon, OpenCLIP, DINOv2 |

## Kurulum

```
pip install -r requirements.txt
```

- YOLO modelleri (`yolov8n.pt`, `yolov8n-pose.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) ilk çalıştırmada Ultralytics tarafından otomatik indiriliyor.
- `opencv-python` 4.x olmalı. OpenCV 5 ile Haar cascade dosyaları paketle gelmiyor, yüz bulanıklaştırma çalışmaz.
- Masa temizlik kontrolü GPU varsa onu, yoksa işlemciyi kullanıyor. `yolov8x` ve 1280 giriş boyutu işlemcide yavaş.

## Çalıştırma

Her dosya tek başına çalışıyor, görüntü ya da video yolu komut satırından veriliyor. Çıkmak için `q`.

```
python 01_temel_goruntu_isleme/kenar_tespiti.py resim.jpg
python 02_insan_tespiti_ve_sayma/kisi_sayma.py video.mp4
python 03_poz_tahmini/durus_tespiti.py oturma.mp4 0.6      # ikinci değer pencere ölçeği
python 03_poz_tahmini/pozisyon_loglama.py video.mp4
python 04_vitrin_ilgi_analizi/vitrin_ilgi_analizi.py vitrin.mp4
python 05_kasa_kuyruk_analizi/kasa_kuyruk_analizi.py kasa.mp4
python 06_masa_temizlik_kontrolu/masa_temizlik.py video.mp4   # video, videos/ klasöründe olmalı
python 07_sigara_tespiti/sigara_tespiti.py video.mp4
```

Vitrin bölgeleri ve kasa sıra alanı staj sırasında kullandığım kamera görüntülerine göre çizildi. Başka bir kamerada koordinatların değiştirilmesi gerekiyor.

## Nasıl çalışıyor

**Duruş tespiti:** pose modelinden kalça, diz ve ayak bileği noktaları alınıyor. Diz açısı 130 derecenin altındaysa oturuyor sayılıyor. Kalça merkezinin son 10 karedeki ortalama yer değiştirmesi kalça genişliğine bölünüp normalize ediliyor, bu değere göre yürüyor/koşuyor ayrımı yapılıyor. Anlık titremeleri önlemek için son karelerin çoğunluk oyu alınıyor. Yandan bakan kişide kalça genişliği sıfıra yaklaştığı için alt sınır 50 piksel.

**Vitrin ilgi analizi:** kişinin kutu merkezi bölge poligonunun içindeyse ve kare başına hızı 10 pikselin altındaysa "inceliyor" sayılıyor. 3 saniyeyi geçince loga yazılıyor. Kişi birkaç kare kaybolursa 2 saniye boyunca hafızada tutuluyor, böylece başka biri önünden geçince sayaç sıfırlanmıyor.

**Kasa kuyruğu:** kişinin kutusunun en az %70'i sıra alanının içindeyse sırada sayılıyor. Kasa bankosu alt gövdeyi kapattığında kutu kısalıyor (boy/en oranı 1.6'nın altına düşüyor). Bu durumda ayakların olması gereken nokta kutu genişliğinden tahmin edilip alanın içinde mi diye bakılıyor.

**Masa temizlik kontrolü:** ilk 25 karede masalar taranıp IoU ile kümelenerek sabit masa listesi çıkarılıyor. Sonradan görünen masalar 1.5 saniye boyunca tutarlı görülürse listeye ekleniyor. Masaya yakın bir kişi varsa "müşteri var". Kişi kalktıktan sonra 6 saniye bekleniyor, masada hâlâ bardak/şişe varsa ve bu 3 saniye boyunca sürüyorsa "kirli masa".

## Gizlilik

Mağaza ve kafe görüntülerinde yüzler bulanıklaştırılıyor. Staj sırasında YOLO ile eğitilmiş bir yüz modeli kullanıldı, o model bana ait olmadığı için repoya eklenmedi. `ortak/yuz_bulaniklastirma.py` varsayılan olarak OpenCV'nin Haar cascade dedektörünü kullanıyor (önden ve yandan). Bu yöntem dönük ya da küçük yüzleri kaçırabiliyor, gerçek kullanımda bir yüz modeli verilmeli:

```python
YuzBulaniklastirici(model_yolu="face.pt")
```

Masa temizlik modülünde `models/face.pt` varsa otomatik olarak o kullanılıyor.

Staj videoları, model ağırlıkları ve log dosyaları repoya eklenmedi.

## Örnek log çıktıları

Pozisyon loglama (`pozisyon_loglari.csv`):

```
Tarih_Saat,Pozisyon,Sure_Saniye
2026-08-14 11:23:51,Ayakta,2.62
2026-08-14 11:23:52,Yuruyor,1.55
2026-08-14 11:23:55,Kosuyor,1.97
2026-08-14 11:23:59,Yuruyor,3.75
2026-08-14 11:24:07,Ayakta,7.83
```

Vitrin ilgi analizi (`vitrin_analiz_log.csv`):

```
Tarih_Saat,Kisi_ID,Durum,Bolge,Bekleme_Suresi_Sn
2026-08-21 10:30:10,7,Vitrine Bakti,Alan 1,3.0
2026-08-21 10:30:20,24,Vitrine Bakti,Alan 2,3.0
```

## Teknolojiler

Python, OpenCV, Ultralytics YOLOv8 (detect, pose, track), BoT-SORT / ByteTrack, Hugging Face Transformers (Grounding DINO), NumPy

## English summary

Code from my 2026 computer vision internship at Ayvos. It starts with classic image processing (my own Gaussian convolution vs OpenCV, thresholding, morphology, Canny, SIFT/ORB) and moves on to retail and cafe camera analytics with YOLOv8:

- people counting with tracking
- pose-based standing / sitting / walking / running detection with duration logging
- shop-window dwell-time analysis
- checkout queue counting with an occlusion fix
- table cleanliness monitoring
- zero-shot cigarette detection with Grounding DINO

Faces are blurred for privacy. Videos and model weights are not included.

## Lisans

AGPL-3.0 (kodlar Ultralytics YOLO kullandığı için)

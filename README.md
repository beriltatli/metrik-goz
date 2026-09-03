# metrik-goz

**Tek fotoğraftan gerçek ölçü — ve dürüst bir hata payı.**

Bir görüntü işleme sistemi "41,2 cm" diyorsa bu tek başına bilgi değil.
"41,2 ± 0,9 cm, %95 güven" bilgi. Bu paket ikincisini üretiyor, ve ürettiği
güven aralığının gerçekten tuttuğunu ölçerek gösteriyor.

```
Mesafe:   412.3 ± 8.7 mm (%95: 395.1–429.4)
```

---

## Neden

Kameradan karar üreten her sistemde aynı adım var: pikselden gerçek birime
geçmek. Buzdolabındaki domatesin kaç gram olduğunu bilmeden hangi tarifin ne
kadar malzeme kurtaracağını hesaplayamazsın; enkazdaki geçidin kaç santim
olduğunu bilmeden 48 cm'lik bir aracın oradan geçip geçemeyeceğine karar
veremezsin.

İkinci örnekte hatanın maliyeti asimetrik: geçilebilir yolu kaçırmak zaman
kaybettirir, **geçilemez yolu geçilebilir sanmak aracı içeride bırakır.** Bu
yüzden nokta tahmini yetmez — aralığın alt ucuna göre karar vermen gerekir. Bu
paket o alt ucu güvenilir biçimde üretmek için var.

---

## Ne yapıyor

Sahnede boyutu bilinen bir referans (ArUco işareti, kredi kartı, A4 kâğıt)
olduğunda, o referansla **aynı düzlem üzerindeki** her şeyi milimetre cinsinden
ölçer:

| Ölçüm | Ne veriyor |
|---|---|
| `mesafe` | iki nokta arası, mm |
| `uzunluk` | kırık çizginin toplam boyu, mm |
| `alan` | çokgen alanı, mm² |
| `en_dar_gecit` | serbest alanın en dar yeri + "şu ayak izi geçer mi" kararı |

Her ölçüm bir `Olcum` nesnesi döner: değer, standart sapma, güven aralığı ve
hangi yöntemle hesaplandığı.

---

## Kullanım

```python
from metrik_goz import Homografi, olcum, belirsizlik, referans
import cv2

goruntu = cv2.imread("tezgah.jpg")
dunya, resim, kimlik = referans.aruco_bul(goruntu, kenar_mm=100.0)

h = Homografi.kur(dunya, resim)
noktalar = [(412, 690), (905, 712)]          # ölçmek istediğin iki nokta

sonuc = belirsizlik.monte_carlo(
    dunya, resim,
    lambda hh, nn: olcum.mesafe(hh, nn[0], nn[1]),
    noktalar, sigma_px=0.4,
)
print(sonuc)          # 412.3 ± 8.7 mm (%95: 395.1–429.4)
```

Komut satırından:

```bash
metrik-goz mesafe tezgah.jpg --aruco 100 --nokta 412,690 --nokta 905,712
metrik-goz gecit  enkaz.jpg  --aruco 200 --maske serbest.png --ayak-izi 480
metrik-goz dogrula --cikti dogrulama/
```

Geçit komutu kararı aralığın **alt ucuna** göre verir — yukarıdaki asimetri
yüzünden.

---

## Web paneli

```bash
pip install -e ".[web]"
metrik-goz panel            # http://127.0.0.1:8000
```

Görseli tarayıcıya sürükle, referansı işaretle, ölç. Sol tarafta görüntü ve
tıkladığın noktalar, sağ tarafta ölçüm, hata payı ve uyarılar.

| Adım | Panelde |
|---|---|
| Görsel | sürükle-bırak, `⌘V` ile yapıştır ya da dosya seç |
| Referans | ArUco'yu otomatik bul, ya da kredi kartı / A4 / kare-dikdörtgen köşelerini elle tıkla |
| Ölçüm | mesafe, kırık çizgi uzunluğu, çokgen alanı, serbest alanın en dar geçidi |
| Çalıştır | sonuç, güven aralığı, geçit için GEÇER/GEÇMEZ kararı ve genişlik profili |

Noktaları sürükleyerek düzeltebilir, tekerlekle yakınlaşabilirsin; imlecin
yanındaki büyüteç köşeyi piksel piksel oturtmak için. Tıklama gürültüsü ölçüm
belirsizliğine giren gerçek bir terim (`sigma_px`), o yüzden büyüteç süs değil.

**Panel hiçbir şey ölçmüyor.** Gördüğün her sayı `metrik_goz.olcum` ve
`metrik_goz.belirsizlik` içindeki, testleri geçen aynı fonksiyonlardan geliyor;
tarayıcı yalnız piksel koordinatı topluyor. Aynı hesabın iki yerde iki kez
yazılması, bir gün ikisinin ayrışması demektir —
`testler/test_web.py` panelin döndürdüğü sayının kütüphanenin döndürdüğüyle
birebir aynı olduğunu kontrol ediyor.

Bir incelik: yüklenen görüntü sunucuda bir kez çözülüp **EXIF'siz** olarak
yeniden kodlanıyor ve tarayıcıya o kopya gidiyor. Telefon fotoğrafları döndürme
bayrağı taşıyor; tarayıcı onu uygular, sunucu uygulamazsa senin tıkladığın
(x, y) ile ölçülen (x, y) farklı yerler olur ve ölçüm sessizce yanlış çıkar.

### Elinde fotoğraf yoksa

```bash
metrik-goz ornek --sahne tezgah --cikti ornekler/
```

Panelin sağ üstündeki **örnek** düğmeleri doğru cevabı bilinen sentetik sahneler
açıyor: kamerayı, referansı ve ölçülecek mesafeyi biz koyduğumuz için panel
ölçümün yanına gerçek değeri de yazabiliyor. Güven aralığının tutup tutmadığını
tek bakışta görmenin başka yolu yok — gerçek fotoğrafta doğru cevap yoktur.

| Örnek | Ne var | Doğru cevap |
|---|---|---|
| `tezgah` | 100 mm ArUco, iki hedef, kenarları çizili dikdörtgen | 410,0 mm · 336,0 cm² |
| `gecit` | 200 mm ArUco, ortada daralan serbest koridor | 520,0 mm |

---

## Nasıl çalışıyor

Üç adım, ve her adımın kendi tuzağı var.

**1 · Homografi kurulumu.** Referansın dört köşesi, dünya düzlemi ile görüntü
arasındaki projektif dönüşümü belirliyor. Önce Hartley normalizasyonlu DLT ile
kapalı formda bir başlangıç çözümü, sonra elle yazılmış Levenberg–Marquardt ile
iyileştirme.

DLT tek başına neden yetmiyor: DLT *cebirsel* hatayı minimize eder, oysa bizim
küçültmek istediğimiz *geometrik* yeniden izdüşüm hatası. Gürültü altında bu
ikisi aynı çözümü vermiyor.

LM çözücüsü de bu depoda (`lm.py`), çünkü paketin bütün belirsizlik iddiası
çözücünün ürettiği kovaryansa dayanıyor — nereden geldiğini bilmediğimiz bir
kovaryansa "±3 cm" demeye hakkımız yok. Analitik Jacobian elle türetildi ve
sayısal türevle karşılaştırılarak doğrulanıyor (`testler/test_lm.py`).

**2 · Ölçüm.** Projektif dönüşüm doğruyu doğruya taşıdığı için çokgen
köşelerini dünya düzlemine taşıyıp orada ölçmek yeterli. Tek istisna
`en_dar_gecit`: orada serbest alanın sınırı eğri olabildiğinden dünya
düzleminde, ilerleme eksenine dik kesitlerle tarama yapılıyor. Her kesitte
kesintisiz serbest parçaların **en uzunu** alınıyor — bir adayla ikiye bölünmüş
koridorda toplam genişlik yanıltıcı olurdu.

**3 · Belirsizlik.** Hata iki ayrı yerden geliyor ve ikisini birden saymayan
her sistem yalancı biçimde dar aralık üretir:

- referans köşelerinin piksel gürültüsü → homografi yanlış kurulur
- ölçtüğün noktaların piksel gürültüsü → doğru homografide yanlış yer

İki yöntem de ikisini birden sayıyor. **Monte Carlo** her iki gürültüyü de
örnekleyip homografiyi yeniden kurar; varsayımsızdır, referans doğrudur.
**Analitik** yol birinci mertebeden yayılım yapar; hızlıdır ve Monte Carlo'ya
karşı doğrulanır.

---

## Doğrulama

Gerçek fotoğrafta doğru cevap yoktur — olsa ölçmeye gerek kalmazdı. Bu yüzden
doğrulama sentetik sahnelerle yapılıyor: kamerayı, referansı ve ölçülecek
mesafeyi biz koyuyoruz, sisteme yalnız gürültülü pikselleri veriyoruz.

Aşağıdaki sayıların hepsi `metrik-goz dogrula` komutuyla üretiliyor;
README'ye elle yazılmış tek bir sayı yok.

### Güven aralığı gerçekten tutuyor mu

Asıl soru bu. Sistem "%95" diyorsa, çok sayıda bağımsız ölçümde gerçek değer o
aralığın içinde %95 oranında düşmeli.

![kapsama](dogrulama/kapsama.png)

Sekiz koşulun ortalaması **%94,3**. Nominal %95'ten bir puanlık eksiklik gerçek
ve açıklanabilir: parametrik bootstrap, dağılımı gerçek köşeler etrafında değil
*gözlenen* (yani zaten gürültülü) köşeler etrafında merkezliyor. Aralığın
genişliği doğru, merkezi biraz kayıyor. Bunu gizlemek yerine yazıyoruz — %95
deyip %80 tutturmaktan iyidir.

Dikkat çeken şey şu: **doğruluk bozulduğu koşullarda bile kapsama tutuyor.**
Referansın 4 katı uzağa ölçüm yapıldığında ortanca hata %3,4'e çıkıyor ama
aralık hâlâ %94,7 kapsıyor — sistem kötüleştiğini biliyor ve söylüyor, sessizce
yanılmıyor. Kapsamanın en çok düştüğü yer 3 metre mesafe (%92,1); orada
referansın görüntüdeki piksel boyutu küçüldüğü için birinci mertebe
varsayımları en çok zorlanıyor.

### Hata nereye kadar %3'ün altında

![hata](dogrulama/hata_uzaklik.png)

| Koşul | Kapsama | Ortanca hata | p90 hata |
|---|---|---|---|
| 0,6 m mesafe | %95,7 | %0,59 | %1,27 |
| 1,2 m mesafe | %95,0 | %1,31 | %2,76 |
| 2,0 m mesafe | %95,0 | %1,83 | %4,57 |
| 3,0 m mesafe | %92,1 | %2,82 | %7,24 |
| Tepeden bakış (0°) | %95,0 | %1,00 | %2,53 |
| 40° bakış | %93,6 | %1,08 | %2,63 |
| 55° bakış (çok yatık) | %93,6 | %1,15 | %3,79 |
| Referansın 4 katı uzakta | %94,7 | %3,41 | %10,51 |

**İlan edilen çalışma bölgesi:** 100 mm referans, 0,5 px köşe gürültüsü,
3 metreye kadar mesafe, referans boyutunun 2 katına kadar uzaklık. Bu bölgede
ortanca bağıl hata %3'ün altında.

Bakış açısının hatayı neredeyse hiç etkilememesi ilk bakışta şaşırtıcı; sebebi
homografinin perspektifi zaten tam olarak modellemesi. Belirleyici olan bakış
açısı değil, **referansın görüntüdeki piksel boyutu** ve ölçülen yerin ondan ne
kadar uzakta olduğu.

### Hızlı yol yavaş yolla aynı sonucu veriyor mu

![analitik](dogrulama/analitik_vs_mc.png)

Analitik yayılımın standart sapması, Monte Carlo'nunkine göre ortanca **1,00**
oranında. Yani bu çalışma bölgesinde birinci mertebeden yaklaşım geçerli ve
400 kat daha hızlı yol güvenle kullanılabilir.

---

## Sınırlar

Bunları README'nin sonuna değil ortasına yazmak lazım, çünkü sistemin sessizce
yanıldığı yerler bunlar:

- **Düzlem varsayımı.** Ölçülen her şey referansla aynı düzlemde olmalı.
  Tezgâha koyduğun kartla tezgâhtaki domatesi ölçebilirsin; raftaki kutuyu
  ölçemezsin. `Homografi.duzlem_disi_uyarisi` referanstan ne kadar
  uzaklaştığını raporluyor; 2'nin üstünde ölçüme güvenme.
- **Lens bozulması.** Geniş açılı telefon kameralarında kenar bölgeler için
  önce düzeltme gerekiyor. Şu an paket bunu yapmıyor, kamera iç parametreleri
  desteği bir sonraki adım.
- **Köşe gürültüsü tahmini.** `sigma_px` varsayılanları deneyimsel (ArUco 0,4;
  elle tıklama 1,5). Kendi kurulumunda ölçmek daha doğru sonuç verir.
- **Geçit ölçümünde maske kalitesi.** `en_dar_gecit`, verilen serbest alan
  maskesinin doğru olduğunu varsayar. Maskenin kendi hatası bu pakette
  modellenmiyor — o, bölütleme katmanının işi.

---

## Kurulum ve testler

```bash
pip install -e ".[gelistirme]"
pytest testler/ -q          # 39 test, ~8 sn
metrik-goz dogrula          # grafikleri ve sonuclar.json'u üretir (~1 dk)
metrik-goz panel            # web panelini aç
```

Çekirdek matematik yalnız NumPy'a bağlı — güven aralığının kritik değerleri bile
standart kütüphaneden geliyor, scipy gerekmiyor. OpenCV yalnız ArUco tespiti ve
görüntü okuma için, Flask yalnız panel için gerekiyor; kendi köşelerini verirsen
çekirdek ikisi olmadan da çalışır.

| Ekstra | Ne getiriyor |
|---|---|
| `.[goruntu]` | OpenCV — ArUco tespiti, görüntü okuma |
| `.[web]` | Flask + OpenCV — `metrik-goz panel` |
| `.[grafik]` | matplotlib — `metrik-goz dogrula` grafikleri |
| `.[gelistirme]` | hepsi + pytest |

---

## Bu depo neyin parçası

`metrik-goz`, aynı görüntü işleme çekirdeğini iki ayrı dünyada koşturan bir
serinin ilk halkası: bir yanda buzdolabında çürümeye giden gıdayı kurtarmak,
öbür yanda enkazda geçilebilir güzergâhı bulmak. İkisi de aynı problemi
çözüyor — düzensiz bir yığına bakıp tek tek nesnelerin gerçek ölçüsünü
çıkarmak ve bir kısıt altında karar vermek. Değişen tek şey kısıtın ne olduğu.

Bu paket o zincirin "gerçek ölçü" halkası.

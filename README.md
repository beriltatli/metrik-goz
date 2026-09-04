<div align="center">
<img width="327" height="334" alt="IMG_4886 3" src="https://github.com/user-attachments/assets/1f5f1636-43f0-4b58-956d-539f2a85974e" />


# metrik-goz

**Tek fotoğraftan gerçek ölçü — ve dürüst bir hata payı.**

```
Mesafe:   412.3 ± 8.7 mm (%95: 395.1–429.4)
```

`saf NumPy çekirdek` · `elle yazılmış Levenberg–Marquardt` · `56 test` · `sentetik doğrulama`

</div>

---

Bir görüntü işleme sistemi "41,2 cm" diyorsa bu tek başına bilgi değil.
"41,2 ± 0,9 cm, %95 güven" bilgi. Bu paket ikincisini üretiyor, ve ürettiği
güven aralığının gerçekten tuttuğunu ölçerek gösteriyor.

```bash
pip install -e ".[web]"
metrik-goz panel          # http://127.0.0.1:8000 — elinde fotoğraf yoksa örnek sahneler hazır
```

<sub>

[Neden](#neden) · [Ne yapıyor](#ne-yapıyor) · [Kullanım](#kullanım) ·
[Web paneli](#web-paneli) · [HTTP API](#http-api) · [Nasıl çalışıyor](#nasıl-çalışıyor) ·
[Doğrulama](#doğrulama) · [Sınırlar](#sınırlar) · [Kurulum](#kurulum-ve-testler)

</sub>

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

Sahnede boyutu bilinen bir şey olduğunda, onunla **aynı düzlem üzerindeki** her
şeyi milimetre cinsinden ölçer:

| Ölçüm | Ne veriyor |
|---|---|
| `kutu` | dört köşesi işaretlenen nesnenin eni, boyu, alanı |
| `mesafe` | iki nokta arası, mm |
| `uzunluk` | kırık çizginin toplam boyu, mm |
| `alan` | çokgen alanı, mm² |
| `en_dar_gecit` | serbest alanın en dar yeri + "şu ayak izi geçer mi" kararı |

Her ölçüm bir `Olcum` nesnesi döner: değer, standart sapma, güven aralığı ve
hangi yöntemle hesaplandığı.

Referans iki aileden biri olabilir ve fark, ölçümün ne kadarına
güvenebileceğini belirliyor:

| Referans ailesi | Ne veriyorsun | Ne kazanıyorsun |
|---|---|---|
| **benzerlik** (`Homografi.olcekten`) | tek bir bilinen **uzunluk** — madeni paranın çapı, kartın uzun kenarı — ve iki ucu | ölçek. Perspektif **düzeltilmiyor**: fotoğraf tepeden çekildiyse doğru, eğik çekildiyse sistematik olarak yanlış |
| **projektif** (`Homografi.kur`) | dört nokta — ArUco işareti ya da dikdörtgen bir nesnenin köşeleri | ölçek **ve** perspektif düzeltmesi |

Ucuz olan yol her zaman geçerli değil; hangisinin ne zaman tutmadığı
[Doğrulama](#doğrulama) bölümünde ölçülmüş durumda.

Kutudan çıkan referans tablosu (`metrik_goz.referans`):

| Aile | Hazır seçenekler |
|---|---|
| Tek uzunluk | 1 TL · 50/25/10/5/1 kuruş · 2 € · 1 € · 50 sent · ABD 25 cent · kredi kartının uzun ya da kısa kenarı |
| Dikdörtgen | `kredi_karti` (ISO ID-1) · `a4` · `a5` · `cd` · `post_it` |
| ArUco | `DICT_4X4_50` (kenar uzunluğunu sen verirsin), köşeler alt piksel doğrulukta |

---

## Kullanım

```python
from metrik_goz import Homografi, olcum, belirsizlik, referans
import cv2

goruntu = cv2.imread("masa.jpg")
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

Elinde ArUco yoksa, yanına koyduğun bir madeni paranın çapı da yeter — o zaman
homografiyi referansın kendi modeli kuruyor:

```python
kur = lambda uc: Homografi.olcekten(uc[0], uc[1], 26.15)   # 1 TL çapı

sonuc = belirsizlik.monte_carlo(
    None, para_uclari_px,
    lambda hh, nn: olcum.kutu(hh, nn).en_mm,
    telefon_koseleri_px, sigma_px=1.5, kur_fn=kur,
)
```

Komut satırından:

```bash
metrik-goz kutu   masa.jpg  --olcek-ad 1_tl --uc 812,455 --uc 888,455 \
                            --kose-nesne ... (nesnenin 4 köşesi)
metrik-goz mesafe masa.jpg  --aruco 100 --nokta 412,690 --nokta 905,712
metrik-goz alan   masa.jpg  --nesne a4 --kose ... (referansın 4 köşesi) \
                            --nokta ... (çokgenin en az 3 noktası)
metrik-goz gecit  enkaz.jpg --aruco 200 --maske serbest.png --ayak-izi 480
metrik-goz ornek  --sahne hepsi --cikti ornekler/
metrik-goz dogrula --cikti dogrulama/
metrik-goz panel  --port 8000
```

Her komut referansı aynı bayraklarla alıyor: `--aruco KENAR_MM`,
`--olcek-ad AD --uc x,y ×2`, `--olcek UZUNLUK_MM --uc x,y ×2` ya da
`--nesne AD --kose x,y ×4`. Ortak `--mc` Monte Carlo örnek sayısını verir.

Geçit komutu kararı aralığın **alt ucuna** göre verir — yukarıdaki asimetri
yüzünden.

---

## Web paneli

```bash
pip install -e ".[web]"
metrik-goz panel            # http://127.0.0.1:8000
```

Görseli tarayıcıya sürükle, referansı işaretle, ölç. Solda akış ve örnek
sahneler, ortada tuval, sağda adımlar ve uyarılar; üstteki dört kart her zaman
son ölçümün gerçek çıktısını gösteriyor.

| Kart | Ne yazıyor |
|---|---|
| **ÖLÇÜ** | değerin kendisi (en / boy / alan) |
| **HATA PAYI** | standart sapma |
| **GÜVEN ARALIĞI** | alt–üst uç; kararı bu aralığın ucu verir |
| **AKTİF REFERANS** | hangi referans, hangi model, σ ve yeniden izdüşüm RMS'i |

Akış üç adım:

| Adım | Panelde |
|---|---|
| **1 · Fotoğraf** | sürükle-bırak, `⌘V` ile yapıştır ya da dosya seç |
| **2 · Referans** | **Uzunluk**: listeden bir para/kart seç (ya da mm'yi elle yaz), iki ucunu tıkla · **Dikdörtgen**: kredi kartı / A4 / A5 / CD / post-it ya da kendi ölçün, dört köşesini tıkla · **ArUco**: kenarı yaz, otomatik bulunsun |
| **3 · Nesne** | ölçmek istediğin şeyin üstüne dört köşeli bir kutu çiz |
| **Sonuç** | en, boy ve alan; her biri güven aralığıyla, en tehlikelisi başta olmak üzere uyarılarla |

**Gelişmiş** bölümünde üç şey senin: tıklama gürültüsü (`sigma_px`, elle
işaretlemede tipik 1–2 px), güven seviyesi (%68 / %90 / %95 / %99) ve Monte
Carlo örnek sayısı. Üçü de doğrudan hesaba giriyor — süs değil.

Panel tek bir akış sürüyor — "şu nesne kaç santim". `mesafe`, `uzunluk`, `alan`
ve `en_dar_gecit` ölçümleri kütüphanede, CLI'da ve `/api/olc` uç noktasında
duruyor; panelde yok, çünkü tek akışlı bir arayüz yanlış tıklamayla yanlış
ölçüm yapılan bir arayüzden iyi.

Noktaları sürükleyerek düzeltebilir, tekerlekle yakınlaşabilir, `sığdır` ile
geri dönebilirsin; imlecin yanındaki büyüteç köşeyi piksel piksel oturtmak için.
Tıklama gürültüsü ölçüm belirsizliğine giren gerçek bir terim (`sigma_px`),
o yüzden büyüteç süs değil.

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

Kenar çubuğundaki **Örnek sahneler** düğmeleri doğru cevabı bilinen sentetik
sahneler açıyor: kamerayı, referansı ve ölçülecek mesafeyi biz koyduğumuz için
panel ölçümün yanına gerçek değeri de yazabiliyor — ve gerçek değer aralığın
içine düşerse **✓**, düşmezse **✗** koyuyor. Güven aralığının tutup tutmadığını
tek bakışta görmenin başka yolu yok; gerçek fotoğrafta doğru cevap yoktur.
Sahne açılırken referansın uçları ve nesnenin köşeleri de yerleştiriliyor, yani
tek tıkla ölçülmüş bir sonuç görüyorsun; noktaları oynatıp ne değiştiğine
bakabilirsin.

Aynı sahneler diske de yazılabiliyor (görüntü + doğru cevabı taşıyan JSON):

```bash
metrik-goz ornek --sahne hepsi --cikti ornekler/
```

| Örnek | Ne var | Doğru cevap | Panelde |
|---|---|---|---|
| `duz` | masada telefon, yanında 1 TL; neredeyse tam tepeden çekim | 146,7 × 71,5 mm · 104,9 cm² | ✓ |
| `egik` | aynı sahne, 26° eğik çekim | aynı | ✓ |
| `gecit` | 200 mm ArUco, ortada daralan serbest koridor | 520,0 mm | CLI/API |

`duz` ile `egik` aynı doğru cevaba sahip ve aralarındaki tek fark kamera açısı —
ikisini arka arkaya çalıştırmak, tek uzunluktan kurulan ölçeğin nerede tuttuğunu
nerede tutmadığını bir bakışta gösteriyor. `egik` sahnede sistem 146,7 mm yerine
112 mm ölçüyor **ve bunu yüksek seviyeli bir uyarıyla söylüyor**; sessizce
yanılmıyor.

---

## HTTP API

Panel kendi sunucusuyla yalnız bu uç noktalar üzerinden konuşuyor; aynı uçlar
dışarıdan da kullanılabilir (`metrik-goz panel`, varsayılan `127.0.0.1:8000`).

| Uç nokta | Ne yapıyor |
|---|---|
| `POST /api/gorsel` | görsel yükler (çok parçalı `dosya`), EXIF'siz kopyasını üretir, `gorsel_id` döner |
| `POST /api/ornek` | `{"ad": "duz"}` — sentetik sahne üretir; doğru cevabı ve ipucu noktalarını da döner |
| `POST /api/aruco` | görselde ArUco arar, köşeleri ve tipik `sigma_px`'i döner |
| `POST /api/olc` | asıl ölçüm: referans + noktalar → değer, std, güven aralığı, uyarı listesi |
| `GET /api/durum` | sürüm, OpenCV sürümü, yükleme sınırı |
| `GET /gorsel/<kimlik>` | yüklenen görselin sunucudaki normalize kopyası |

Uyarılar `/api/olc` yanıtında `seviye` (`yuksek` / `orta` / `bilgi`) ile geliyor
ve en tehlikelisi başta sıralanıyor; eşikler modele göre değişiyor, çünkü
benzerlik ile projektif modelin zaafları farklı yerlerde.

---

## Nasıl çalışıyor

Üç adım, ve her adımın kendi tuzağı var.

**1 · Homografi kurulumu.** Referansın dört köşesi, dünya düzlemi ile görüntü
arasındaki projektif dönüşümü belirliyor. Önce Hartley normalizasyonlu DLT ile
kapalı formda bir başlangıç çözümü, sonra elle yazılmış Levenberg–Marquardt ile
iyileştirme.

Elde dört nokta yoksa — cebinden çıkardığın paranın yalnız çapını biliyorsun —
projektif dönüşüm kurulamaz: iki nokta ve bir uzunluk üç sayı taşır, projektif
dönüşümün ise sekiz serbestliği vardır. Eksik bilgiyi uydurmak yerine daha dar
bir model kuruluyor (`Homografi.olcekten`): ölçek + döndürme + öteleme, yani
perspektif **düzeltilmiyor**. Bu, ucuz bir kısayol değil; ilan edilmiş ve
ölçülmüş bir sınır — nerede tuttuğu aşağıda.

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

### Tek uzunluktan ölçek nerede tutuyor, nerede tutmuyor

Yukarıdaki her şey dört noktalı projektif referans için. Kullanıcının elinde
çoğu zaman o yok, bir madeni para var — ve o modelin zaafı bambaşka bir yerde.

![benzerlik](dogrulama/benzerlik.png)

| Koşul | Sistematik yanlılık | Kapsama |
|---|---|---|
| Tam tepeden çekim, referans nerede olursa olsun | %0,000 | %95,1 |
| 30° eğik çekim | %16,9 | %27,1 |

Okunacak üç şey var:

**Yanlılığı üreten şey eğim, uzaklık değil.** Tam tepeden çekimde ölçek düzlemin
her yerinde aynı olduğu için para nerede durursa dursun yanlılık sıfır — projektif
modelde risk olan "referanstan uzaklık" burada tek başına zararsız. Bakış
yatıklaştıkça yanlılık büyüyor ve uzaklık onu çarpan olarak büyütüyor.

**Bu yanlılık hata payının İÇİNDE DEĞİL.** Monte Carlo tıklama gürültüsünü
sayıyor, modelin kendi kusurunu sayamaz — o yüzden eğim büyürken kapsama
%95,1'ten %27,1'e çöküyor. Aralığın genişliği doğru, merkezi kayıyor.
Bir sistemin sessizce yanılabildiği yer tam olarak burası, o yüzden ortaya
yazıyoruz.

**Kullanıcı eğimi bilmiyor ama sistem görebiliyor.** Ölçülen nesnenin karşılıklı
kenarları eğik çekimde ayrışıyor; `Kutu.dikdortgenlik` bu ayrışmayı ölçüyor ve
eğimin gözlenebilir vekili oluyor. %6 eşiği, %5'ten büyük yanlılığın
**%78'ini** yakalıyor; %16 yanlış alarm karşılığında. Panelin ve CLI'ın
"fotoğraf eğik çekilmiş, hata payı bunu kapsamıyor" uyarısı bu sayıya dayanıyor —
uyarı metni bir tahmin değil, ölçülmüş bir yakalama oranı. Panel aynı ölçüyü
%2 eşiğinde bir de "orta" seviyede kullanıyor: hafif perspektifi, kararı
bozmadan önce söylemek için.

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
  Masaya koyduğun kartla masadaki domatesi ölçebilirsin; raftaki kutuyu
  ölçemezsin. `Homografi.duzlem_disi_uyarisi` referanstan ne kadar
  uzaklaştığını raporluyor; 2'nin üstünde ölçüme güvenme.
- **Tek uzunluktan ölçekte perspektif.** Bir paranın çapıyla kurulan benzerlik
  modeli perspektifi düzeltmiyor ve bıraktığı yanlılık hata payının içinde
  değil — 30° eğik çekimde %16,9 yanlılık, %27,1 kapsama. Sistem bunu
  `Kutu.dikdortgenlik` üzerinden yakalayıp uyarıyor (ciddi yanlılığın %78'i),
  ama yakalayamadığı %22 var: fotoğrafı nesnenin tam üstünden çek, ya da
  referans olarak dikdörtgen bir şey kullanıp dört köşesini işaretle.
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
pytest testler/ -q          # 56 test, ~15 sn
metrik-goz dogrula          # grafikleri ve sonuclar.json'u üretir (~1 dk)
metrik-goz panel            # web panelini aç
```

Çekirdek matematik yalnız NumPy'a bağlı — güven aralığının kritik değerleri bile
standart kütüphaneden geliyor, scipy gerekmiyor. OpenCV yalnız ArUco tespiti ve
görüntü okuma için, Flask yalnız panel için gerekiyor; kendi köşelerini verirsen
çekirdek ikisi olmadan da çalışır. Python 3.10+.

| Ekstra | Ne getiriyor |
|---|---|
| `.[goruntu]` | OpenCV — ArUco tespiti, görüntü okuma |
| `.[web]` | Flask + OpenCV — `metrik-goz panel` |
| `.[grafik]` | matplotlib — `metrik-goz dogrula` grafikleri |
| `.[gelistirme]` | hepsi + pytest |

### Depo düzeni

```
metrik_goz/
  homografi.py     DLT + LM ile homografi, benzerlik modeli, düzlem dışı uyarısı
  lm.py            elle yazılmış Levenberg–Marquardt (analitik Jacobian)
  olcum.py         mesafe, uzunluk, alan, kutu, en_dar_gecit
  belirsizlik.py   Monte Carlo ve analitik yayılım, Olcum tipi
  referans.py      ArUco tespiti, bilinen uzunluk/nesne tabloları
  sentetik.py      doğru cevabı bilinen sahne üreteci
  ornek.py         panelin ve CLI'ın örnek sahneleri
  dogrulama.py     kapsama/hata/benzerlik taramaları ve grafikleri
  cli.py           komut satırı
  web/             Flask sunucusu + panel (sunucu hesabı, tarayıcı yalnız piksel)
testler/           56 test: geometri, LM, kapsama, web-kütüphane eşitliği
dogrulama/         `metrik-goz dogrula` çıktısı: grafikler + sonuclar.json
ornekler/          `metrik-goz ornek` çıktısı: görüntü + doğru cevap JSON'u
metrik-goz-atolye/ aynı çekirdeği sıfırdan yazmak için adım adım atölye
                   (kasıtlı olarak yarım: testler hazır, kod sana ait)
```

---

## Bu depo neyin parçası

`metrik-goz`, aynı görüntü işleme çekirdeğini iki ayrı dünyada koşturan bir
serinin ilk halkası: bir yanda buzdolabında çürümeye giden gıdayı kurtarmak,
öbür yanda enkazda geçilebilir güzergâhı bulmak. İkisi de aynı problemi
çözüyor — düzensiz bir yığına bakıp tek tek nesnelerin gerçek ölçüsünü
çıkarmak ve bir kısıt altında karar vermek. Değişen tek şey kısıtın ne olduğu.

Bu paket o zincirin "gerçek ölçü" halkası.

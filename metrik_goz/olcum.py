"""
Ölçüm katmanı: homografi kurulduktan sonra piksel üzerinde yapılan işlemler,
milimetre cinsinden cevap verir.

Projektif dönüşüm doğruyu doğruya taşıdığı için çokgen köşelerini dünyaya
taşıyıp orada ölçmek doğru sonucu verir; kenarların içini ayrıca örneklemeye
gerek yok. Tek istisna `en_dar_gecit`: orada serbest alanın kendisi eğri
sınırlı olabildiği için dünya düzleminde tarama yapıyoruz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homografi import Homografi


@dataclass
class Gecit:
    """En dar geçidin sonucu."""

    genislik_mm: float
    konum_mm: np.ndarray          # geçidin orta noktası, dünya koordinatı
    eksen: np.ndarray             # ilerleme yönü (birim vektör)
    profil_mm: np.ndarray         # her istasyondaki serbest genişlik
    istasyonlar_mm: np.ndarray    # istasyonların eksen üzerindeki konumu
    kenar_payi_mm: float = 0.0    # iki uçtan atlanan mesafe (aşağıda anlatılıyor)

    def gecer_mi(self, ayak_izi_mm: float, pay_mm: float = 0.0) -> bool:
        """Verilen ayak izine sahip araç bu geçitten geçer mi."""
        return self.genislik_mm >= ayak_izi_mm + pay_mm


@dataclass
class Kutu:
    """Dört köşesi işaretlenmiş bir nesnenin düzlem üzerindeki ölçüleri."""

    en_mm: float                  # 1. ve 3. kenarın ortalaması (çizim sırasına göre)
    boy_mm: float                 # 2. ve 4. kenarın ortalaması
    alan_mm2: float
    kenarlar_mm: np.ndarray       # dört kenar, köşe sırasıyla
    koseler_mm: np.ndarray
    dikdortgenlik: float          # karşılıklı kenar uyumsuzluğu; 0 = kusursuz

    @property
    def kosegen_mm(self) -> float:
        k = self.koseler_mm
        return float((np.hypot(*(k[2] - k[0])) + np.hypot(*(k[3] - k[1]))) / 2.0)


# ----------------------------------------------------------------- temel ölçüler
def mesafe(homografi: Homografi, p1_px, p2_px) -> float:
    """İki piksel arasındaki gerçek mesafe (mm)."""
    d = homografi.dunyaya(np.array([p1_px, p2_px], dtype=float))
    return float(np.hypot(*(d[1] - d[0])))


def uzunluk(homografi: Homografi, kirik_cizgi_px) -> float:
    """Kırık çizginin toplam uzunluğu (mm)."""
    d = homografi.dunyaya(np.asarray(kirik_cizgi_px, dtype=float))
    return float(np.sum(np.hypot(*(np.diff(d, axis=0).T))))


def kutu(homografi: Homografi, dort_kose_px) -> Kutu:
    """
    Dört köşesi verilen nesnenin en, boy ve alanı (mm, mm²).

    Karşılıklı kenarların ORTALAMASINI alıyoruz, birini değil: kullanıcının
    köşeleri birkaç piksel şaşması kaçınılmaz, ortalama bu şaşmayı yarıya
    indiriyor.

    `dikdortgenlik` bu ortalamanın gizlediği şeyi geri veriyor: karşılıklı iki
    kenar birbirinden ne kadar farklı ölçülüyor. Sıfırdan uzaklaşması iki şeyden
    birine işaret eder — ya nesne referansla aynı düzlemde değil, ya da
    perspektif düzeltilmemiş (benzerlik modelinde eğik bakış nesneyi yamuk
    gösterir). İkisi de ölçümü bozar, ikisi de sessiz kalırsa fark edilmez.
    """
    d = homografi.dunyaya(np.asarray(dort_kose_px, dtype=float))
    if len(d) != 4:
        raise ValueError(f"Kutu için tam 4 köşe gerekiyor, {len(d)} verildi.")

    kenarlar = np.array([float(np.hypot(*(d[(i + 1) % 4] - d[i]))) for i in range(4)])
    en = float((kenarlar[0] + kenarlar[2]) / 2.0)
    boy = float((kenarlar[1] + kenarlar[3]) / 2.0)

    x, y = d[:, 0], d[:, 1]
    alan_mm2 = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    sapma = max(abs(kenarlar[0] - kenarlar[2]) / max(en, 1e-9),
                abs(kenarlar[1] - kenarlar[3]) / max(boy, 1e-9))
    return Kutu(en_mm=en, boy_mm=boy, alan_mm2=alan_mm2, kenarlar_mm=kenarlar,
                koseler_mm=d, dikdortgenlik=float(sapma))


def alan(homografi: Homografi, cokgen_px) -> float:
    """Çokgenin alanı (mm^2). Ayakkabı bağı (shoelace) formülü."""
    d = homografi.dunyaya(np.asarray(cokgen_px, dtype=float))
    x, y = d[:, 0], d[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ----------------------------------------------------------------- en dar geçit
def en_dar_gecit(
    homografi: Homografi,
    serbest_maske: np.ndarray,
    *,
    eksen=None,
    adim_mm: float = 20.0,
    ornek_mm: float = 5.0,
    kenar_payi_mm: float | None = None,
    en_fazla_ornek: int = 4_000_000,
) -> Gecit:
    """
    Serbest alanın en dar noktasını dünya düzleminde tarayarak bulur.

    `serbest_maske`: görüntü boyutunda bool dizi, True = geçilebilir zemin.
    `eksen`: ilerleme yönü (dünya koordinatında birim vektör). Verilmezse
             serbest alanın ana ekseni (PCA) kullanılır.
    `adim_mm`: kaç milimetrede bir kesit alınacağı.
    `ornek_mm`: kesit üzerinde örnekleme sıklığı — ölçüm çözünürlüğü budur.
    `kenar_payi_mm`: iki uçtan atlanacak mesafe. None ise koridor boyunun %5'i.

    Kenar payı neden var ve neden varsayılanı sıfır değil: serbest alan sivri
    biten bir çokgense (elle çizilen maskeler genelde öyle) en uçtaki kesit
    neredeyse sıfır genişlikte çıkar ve minimum oraya kilitlenir. O sıfır
    geçidin darlığı değil, maskenin bittiği yerdir. Uçları atmak bunu önlüyor;
    ne kadar attığımızı `kenar_payi_mm` alanında raporluyoruz.

    Yöntem: eksene DİK kesitler alınır, her kesitte kesintisiz serbest
    parçaların en uzunu o istasyonun genişliğidir (bir adada bölünmüş
    koridorda toplam genişlik yanıltıcı olurdu). Tüm istasyonların minimumu
    geçidin en dar yeridir.

    AEON tarafında bu fonksiyon "İKA buradan geçer mi" sorusunu, mutfak
    tarafında "bu rafa kaç kutu sığar" sorusunu aynı kodla cevaplıyor.
    """
    if adim_mm <= 0 or ornek_mm <= 0:
        raise ValueError("adim_mm ve ornek_mm pozitif olmalı.")
    if serbest_maske.dtype != bool:
        serbest_maske = serbest_maske.astype(bool)
    if serbest_maske.ndim != 2:
        raise ValueError("Serbest maske iki boyutlu olmalı.")
    yuk, gen = serbest_maske.shape

    ys, xs = np.nonzero(_sinir_pikselleri(serbest_maske))
    if len(xs) < 10:
        raise ValueError("Serbest maske neredeyse boş; ölçülecek geçit yok.")

    # Yalnız maskenin SINIRINI dünyaya taşıyoruz, içini değil. Tarama aralığını
    # belirleyen uç değerler tanım gereği sınırda; iç pikseller aynı sayıyı
    # yeniden üretmekten başka bir şey yapmıyor. Tipik bir koridorda bu, yarım
    # milyon noktalık dönüşümü birkaç bine indiriyor — Monte Carlo bu işi
    # yüzlerce kez tekrarladığı için ölçümün tamamını kat kat hızlandırıyor.
    dunya = homografi.dunyaya(np.column_stack([xs, ys]).astype(float))

    # Eksen: verilmemişse serbest alanın ana ekseni (sınır noktalarının PCA'sı;
    # uzunlamasına yönü alan tabanlı PCA ile aynı verir, uzama yönü baskındır).
    if eksen is None:
        merkezli = dunya - dunya.mean(axis=0)
        _, _, Vt = np.linalg.svd(merkezli, full_matrices=False)
        eksen = Vt[0]
    eksen = np.asarray(eksen, dtype=float).reshape(2)
    norm = np.linalg.norm(eksen)
    if norm < 1e-12:
        raise ValueError("Eksen sıfır vektör olamaz.")
    eksen = eksen / norm
    dik = np.array([-eksen[1], eksen[0]])

    merkez = dunya.mean(axis=0)
    t = (dunya - merkez) @ eksen        # eksen boyunca konum
    s = (dunya - merkez) @ dik          # eksene dik konum

    t0, t1 = float(t.min()), float(t.max())
    s_yari = float(np.abs(s).max()) + 2 * ornek_mm

    # Uçlarda maske kırpması yanlış darlık üretir; iki uçtan pay bırak.
    if kenar_payi_mm is None:
        kenar_payi_mm = 0.05 * (t1 - t0)
    t0 += kenar_payi_mm
    t1 -= kenar_payi_mm
    if t1 <= t0:
        raise ValueError("Kenar payı serbest alanın tamamını yiyor.")

    istasyonlar = np.arange(t0, t1 + 1e-9, adim_mm)
    ofsetler = np.arange(-s_yari, s_yari + 1e-9, ornek_mm)
    if istasyonlar.size * ofsetler.size > en_fazla_ornek:
        raise ValueError(
            f"Tarama ızgarası çok büyük ({istasyonlar.size}×{ofsetler.size}). "
            f"adim_mm ya da ornek_mm değerini büyüt.")

    # Bütün kesitleri tek dönüşümde piksel uzayına taşı: istasyon başına ayrı
    # homografi çağrısı yapmak ölçümün en pahalı kısmıydı, Monte Carlo bunu
    # yüzlerce kez tekrarlıyor.
    izler = merkez + istasyonlar[:, None, None] * eksen + ofsetler[None, :, None] * dik
    kesit_px = homografi.resme(izler.reshape(-1, 2))
    u = np.rint(kesit_px[:, 0]).astype(np.int64)
    v = np.rint(kesit_px[:, 1]).astype(np.int64)
    gecerli = (u >= 0) & (u < gen) & (v >= 0) & (v < yuk)
    serbest = np.zeros(u.shape, dtype=bool)
    serbest[gecerli] = serbest_maske[v[gecerli], u[gecerli]]
    serbest = serbest.reshape(len(istasyonlar), len(ofsetler))

    genislikler = np.zeros(len(istasyonlar))
    orta_noktalar = np.zeros((len(istasyonlar), 2))
    for i, ti in enumerate(istasyonlar):
        bas, boy = _en_uzun_kesintisiz(serbest[i])
        genislikler[i] = boy * ornek_mm
        orta = (ofsetler[bas] + (boy - 1) * ornek_mm / 2.0) if boy > 0 else 0.0
        orta_noktalar[i] = merkez + ti * eksen + orta * dik

    en_dar = int(np.argmin(genislikler))
    return Gecit(
        genislik_mm=float(genislikler[en_dar]),
        konum_mm=orta_noktalar[en_dar],
        eksen=eksen,
        profil_mm=genislikler,
        istasyonlar_mm=istasyonlar - istasyonlar[0],
        kenar_payi_mm=float(kenar_payi_mm),
    )


def _sinir_pikselleri(maske: np.ndarray) -> np.ndarray:
    """
    Maskenin sınır pikselleri: dört komşusunun hepsi serbest OLMAYAN serbest
    pikseller. Görüntü kenarına dayanan pikseller de sınır sayılır.
    """
    if maske.shape[0] < 3 or maske.shape[1] < 3:
        return maske
    ic = np.zeros_like(maske)
    ic[1:-1, 1:-1] = (maske[1:-1, 1:-1] & maske[:-2, 1:-1] & maske[2:, 1:-1]
                      & maske[1:-1, :-2] & maske[1:-1, 2:])
    sinir = maske & ~ic
    return sinir if sinir.sum() >= 10 else maske


def _en_uzun_kesintisiz(bayraklar: np.ndarray) -> tuple[int, int]:
    """En uzun ardışık True dizisinin (başlangıç indeksi, uzunluk) değeri."""
    if not bayraklar.any():
        return 0, 0
    # Kenarlara sıfır ekleyip geçişleri bul
    d = np.diff(np.concatenate([[0], bayraklar.view(np.int8), [0]]))
    baslar = np.nonzero(d == 1)[0]
    bitisler = np.nonzero(d == -1)[0]
    boylar = bitisler - baslar
    k = int(np.argmax(boylar))
    return int(baslar[k]), int(boylar[k])

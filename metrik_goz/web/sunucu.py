"""
Flask sunucusu: panelin arka ucu.

Tasarım kararı — sunucu ölçüm yapmıyor, ölçümü çağırıyor. Bütün matematik
`metrik_goz.olcum` ve `metrik_goz.belirsizlik` içinde ve testlerle korunuyor.
Buradaki kod üç şeyden sorumlu:

  1) Yüklenen görüntüyü tarayıcıdakiyle BİREBİR aynı piksel ızgarasında tutmak.
     Bu göründüğünden önemli: telefon fotoğrafları EXIF döndürme bayrağı
     taşıyor, tarayıcı onu uyguluyor. Sunucu uygulamazsa kullanıcının tıkladığı
     (x, y) ile bizim dizideki (x, y) farklı yerler olur ve ölçüm sessizce
     yanlış çıkar. Çözüm: görüntüyü bir kez çözüp EXIF'siz olarak yeniden
     kodluyoruz ve tarayıcıya o kopyayı gönderiyoruz.
  2) Girdiyi doğrulamak — eksik nokta, saçma sigma, kayıp görüntü.
  3) Sonucu, uyarılarıyla birlikte, çizilebilir biçimde döndürmek.
"""

from __future__ import annotations

import io
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

from .. import __version__, belirsizlik, olcum, ornek, referans
from ..homografi import Homografi

# Yükleme sınırları
EN_BUYUK_YUKLEME = 32 * 1024 * 1024        # 32 MB
EN_FAZLA_GORSEL = 8                        # bellekteki görüntü sayısı (LRU)
EN_FAZLA_NOKTA = 400                       # çokgen başına
MC_ARALIK = (50, 4000)
GECIT_MC_TAVAN = 120                       # geçit örneği pahalı; MC'yi burada kes

OLCUM_TURLERI = {
    "kutu": dict(en_az=4, en_cok=4, birim="mm"),
    "mesafe": dict(en_az=2, en_cok=2, birim="mm"),
    "uzunluk": dict(en_az=2, en_cok=EN_FAZLA_NOKTA, birim="mm"),
    "alan": dict(en_az=3, en_cok=EN_FAZLA_NOKTA, birim="cm²"),
    "gecit": dict(en_az=3, en_cok=EN_FAZLA_NOKTA, birim="mm"),
}


class IstekHatasi(Exception):
    """Kullanıcı girdisinden kaynaklanan, 400 ile dönülecek hata."""

    def __init__(self, mesaj: str, kod: int = 400):
        super().__init__(mesaj)
        self.mesaj = mesaj
        self.kod = kod


# ------------------------------------------------------------------ görüntü deposu
@dataclass
class Gorsel:
    kimlik: str
    dizi: np.ndarray                 # BGR, sunucunun ölçtüğü ızgara
    veri: bytes                      # tarayıcıya giden kodlanmış kopya (EXIF'siz)
    mime: str
    ad: str
    eklendi: float = field(default_factory=time.time)

    @property
    def genislik(self) -> int:
        return int(self.dizi.shape[1])

    @property
    def yukseklik(self) -> int:
        return int(self.dizi.shape[0])

    def ozet(self) -> dict:
        return dict(kimlik=self.kimlik, ad=self.ad,
                    genislik=self.genislik, yukseklik=self.yukseklik,
                    url=f"/gorsel/{self.kimlik}")


class Depo:
    """
    Görüntüleri bellekte tutan, en eskiyi düşüren küçük bir depo.

    Diske yazmıyoruz: panel tek kullanıcılık bir araç, yüklenen fotoğrafın
    sunucuda iz bırakmaması hem daha temiz hem daha güvenli. Sınır bellekte
    kaç görüntü tutulacağı; aşılınca en eski düşer.
    """

    def __init__(self, kapasite: int = EN_FAZLA_GORSEL):
        self._kapasite = kapasite
        self._ogeler: OrderedDict[str, Gorsel] = OrderedDict()
        self._kilit = threading.Lock()

    def ekle(self, gorsel: Gorsel) -> Gorsel:
        with self._kilit:
            self._ogeler[gorsel.kimlik] = gorsel
            self._ogeler.move_to_end(gorsel.kimlik)
            while len(self._ogeler) > self._kapasite:
                self._ogeler.popitem(last=False)
        return gorsel

    def al(self, kimlik: str) -> Gorsel:
        with self._kilit:
            gorsel = self._ogeler.get(kimlik)
            if gorsel is None:
                raise IstekHatasi(
                    "Görüntü sunucuda yok — sekmeyi yenile ve yeniden yükle. "
                    "(Panel yalnız son birkaç görüntüyü bellekte tutuyor.)", 404)
            self._ogeler.move_to_end(kimlik)
            return gorsel


# ------------------------------------------------------------------ yardımcılar
def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as hata:                                # pragma: no cover
        raise IstekHatasi(
            "Bu işlem için OpenCV gerekiyor: pip install 'metrik-goz[web]'", 500) from hata


def _gorseli_coz(veri: bytes, ad: str) -> Gorsel:
    """Baytları çözer ve tarayıcıya gidecek EXIF'siz kopyayı hazırlar."""
    cv2 = _cv2()
    dizi = cv2.imdecode(np.frombuffer(veri, np.uint8), cv2.IMREAD_COLOR)
    if dizi is None:
        raise IstekHatasi("Görüntü okunamadı. JPEG, PNG, WEBP ya da BMP olmalı.")
    if min(dizi.shape[:2]) < 32:
        raise IstekHatasi("Görüntü ölçüm için fazla küçük (en küçük kenar 32 piksel).")

    # imdecode EXIF yönelimini uyguladı; yeniden kodlarken EXIF taşınmıyor, yani
    # tarayıcının göreceği ızgara bizimkiyle birebir aynı.
    basarili, tampon = cv2.imencode(".jpg", dizi, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not basarili:                                           # pragma: no cover
        raise IstekHatasi("Görüntü yeniden kodlanamadı.")
    return Gorsel(kimlik=uuid.uuid4().hex, dizi=dizi, veri=tampon.tobytes(),
                  mime="image/jpeg", ad=ad or "yuklenen")


def _sayi(govde: dict, anahtar: str, varsayilan=None, *,
          en_az=None, en_cok=None, zorunlu=False) -> float | None:
    ham = govde.get(anahtar, varsayilan)
    if ham is None:
        if zorunlu:
            raise IstekHatasi(f"'{anahtar}' alanı gerekli.")
        return None
    try:
        deger = float(ham)
    except (TypeError, ValueError):
        raise IstekHatasi(f"'{anahtar}' sayı olmalı, '{ham}' verildi.") from None
    if not np.isfinite(deger):
        raise IstekHatasi(f"'{anahtar}' sonlu bir sayı olmalı.")
    if en_az is not None and deger < en_az:
        raise IstekHatasi(f"'{anahtar}' en az {en_az:g} olmalı.")
    if en_cok is not None and deger > en_cok:
        raise IstekHatasi(f"'{anahtar}' en çok {en_cok:g} olabilir.")
    return deger


def _noktalar(ham, ad: str, en_az: int, en_cok: int, boyut: tuple[int, int]) -> np.ndarray:
    if not isinstance(ham, (list, tuple)):
        raise IstekHatasi(f"'{ad}' nokta listesi olmalı.")
    if len(ham) < en_az:
        raise IstekHatasi(f"{ad} için en az {en_az} nokta gerekiyor, {len(ham)} verildi.")
    if len(ham) > en_cok:
        raise IstekHatasi(f"{ad} için en çok {en_cok} nokta verilebilir.")
    try:
        p = np.asarray(ham, dtype=float).reshape(len(ham), 2)
    except (ValueError, TypeError):
        raise IstekHatasi(f"'{ad}' [[x, y], ...] biçiminde olmalı.") from None
    if not np.all(np.isfinite(p)):
        raise IstekHatasi(f"'{ad}' içinde sonlu olmayan koordinat var.")

    # Çerçeve dışı nokta geometrik olarak geçersiz DEĞİL — homografi düzlemin
    # tamamında tanımlı ve kadraja sığmayan bir koridorun kenarını işaretlemek
    # olağan. Reddetmiyoruz; ne kadar dışarıda olduğunu uyarı olarak veriyoruz.
    # Sınır yalnız taşmaya karşı: düzlem çok uzakta sayısal olarak anlamsızlaşır.
    yuk, gen = boyut
    if np.abs(p[:, 0]).max() > 50 * gen or np.abs(p[:, 1]).max() > 50 * yuk:
        raise IstekHatasi(f"'{ad}' görüntüden inanılmayacak kadar uzakta.")
    return p


@dataclass
class RefModeli:
    """
    Referansın hangi modeli kurduğu.

    İki aile var ve farkları ölçümün ne kadarına güvenebileceğini belirliyor:

    benzerlik  — tek bir bilinen UZUNLUK (paranın çapı). Ölçek biliniyor,
                 perspektif DÜZELTİLMİYOR. İki gözlem noktası var; kalan iki
                 köşe onlardan türetiliyor, bu yüzden homografiyi `kur_fn`
                 kuruyor: hangi sayıların gerçekten gözlem olduğunu model bilir.
    projektif  — dört nokta (dikdörtgen köşeleri ya da ArUco). Perspektif
                 düzeltiliyor.
    """

    tur: str
    aile: str
    resim: np.ndarray                       # gözlenen noktalar
    sigma: float
    etiket: str
    dunya: np.ndarray | None = None
    kur_fn: object | None = None

    def kur(self) -> Homografi:
        if self.kur_fn is not None:
            return self.kur_fn(self.resim)
        return Homografi.kur(self.dunya, self.resim)


def _referansi_coz(govde: dict, gorsel: Gorsel) -> RefModeli:
    ref = govde.get("referans") or {}
    tur = ref.get("tur", "olcek")
    boyut = (gorsel.yukseklik, gorsel.genislik)

    if tur == "olcek":
        # En basit yol: nesnenin yanındaki paranın iki ucunu tıkla, çapını yaz.
        uzunluk = ref.get("uzunluk_mm")
        ad = ref.get("ad")
        if uzunluk is None and ad:
            if ad not in referans.BILINEN_UZUNLUKLAR:
                raise IstekHatasi(f"Bilinmeyen referans '{ad}'.")
            uzunluk = referans.BILINEN_UZUNLUKLAR[ad][0]
        uzunluk = _sayi({"uzunluk_mm": uzunluk}, "uzunluk_mm", zorunlu=True,
                        en_az=0.1, en_cok=1_000_000.0)
        noktalar = _noktalar(ref.get("noktalar"), "referans uçları", 2, 2, boyut)
        etiket = (referans.BILINEN_UZUNLUKLAR[ad][1] if ad in referans.BILINEN_UZUNLUKLAR
                  else f"{uzunluk:g} mm bilinen uzunluk")
        return RefModeli(
            tur=tur, aile="benzerlik", resim=noktalar,
            sigma=referans.TIPIK_SIGMA_PX["elle"],
            etiket=f"{etiket} · {uzunluk:g} mm",
            kur_fn=lambda g, u=uzunluk: Homografi.olcekten(g[0], g[1], u),
        )

    if tur == "aruco":
        kenar = _sayi(ref, "kenar_mm", zorunlu=True, en_az=1.0, en_cok=100_000.0)
        sozluk = str(ref.get("sozluk", "DICT_4X4_50"))
        koseler = ref.get("koseler")
        if koseler:
            # Panel işareti bir kez bulup köşeleri saklıyor; tekrar aramaya gerek yok.
            resim = _noktalar(koseler, "referans köşeleri", 4, 4,
                              (gorsel.yukseklik, gorsel.genislik))
            etiket = ref.get("etiket") or f"ArUco {kenar:g} mm"
        else:
            _, resim, kimlik = _aruco_bul(gorsel, kenar, sozluk)
            etiket = f"ArUco #{kimlik} · {kenar:g} mm"
        return RefModeli(tur=tur, aile="projektif", dunya=referans.kare_dunya(kenar),
                         resim=resim, sigma=referans.TIPIK_SIGMA_PX["aruco"],
                         etiket=etiket)

    if tur == "kare":
        kenar = _sayi(ref, "kenar_mm", zorunlu=True, en_az=1.0, en_cok=100_000.0)
        dunya = referans.kare_dunya(kenar)
        etiket = f"Elle kare · {kenar:g} mm"
    elif tur == "dikdortgen":
        gen_mm = _sayi(ref, "genislik_mm", zorunlu=True, en_az=1.0, en_cok=100_000.0)
        yuk_mm = _sayi(ref, "yukseklik_mm", zorunlu=True, en_az=1.0, en_cok=100_000.0)
        dunya = referans.dikdortgen_dunya(gen_mm, yuk_mm)
        etiket = f"Elle dikdörtgen · {gen_mm:g}×{yuk_mm:g} mm"
    elif tur == "nesne":
        ad = str(ref.get("nesne", ""))
        try:
            dunya = referans.bilinen_nesne(ad)
        except KeyError as hata:
            raise IstekHatasi(str(hata)) from None
        g, y = referans.BILINEN_NESNELER[ad]
        etiket = f"{ad} · {g:g}×{y:g} mm"
    else:
        raise IstekHatasi(f"Bilinmeyen referans türü '{tur}'.")

    resim = _noktalar(ref.get("koseler"), "referans köşeleri", 4, 4, boyut)
    return RefModeli(tur=tur, aile="projektif", dunya=dunya, resim=resim,
                     sigma=referans.TIPIK_SIGMA_PX["elle"], etiket=etiket)


def _aruco_bul(gorsel: Gorsel, kenar_mm: float, sozluk: str):
    _cv2()
    try:
        return referans.aruco_bul(gorsel.dizi, kenar_mm, sozluk=sozluk)
    except AttributeError:
        raise IstekHatasi(f"Bilinmeyen ArUco sözlüğü '{sozluk}'.") from None
    except ValueError as hata:
        raise IstekHatasi(str(hata)) from None


def _maske_kur(noktalar: np.ndarray, boyut: tuple[int, int]) -> np.ndarray:
    """Çizilen çokgeni serbest alan maskesine çevirir."""
    cv2 = _cv2()
    yuk, gen = boyut
    maske = np.zeros((yuk, gen), np.uint8)
    kirpik = np.clip(np.rint(noktalar), -1e6, 1e6).astype(np.int32)
    cv2.fillPoly(maske, [kirpik], 255)
    if maske.sum() == 0:
        raise IstekHatasi("Çizilen çokgen boş bir alan kaplıyor.")
    return maske > 127


def _uyarilar(h: Homografi, noktalar: np.ndarray, ref: RefModeli,
              en_kotu_bagil: float, boyut: tuple[int, int],
              sigma_px: float, kutu_sonucu=None) -> list[dict]:
    """
    Ölçümün nerede sessizce yanılabileceğini söyleyen liste, en tehlikelisi başta.
    Kullanıcı listenin tamamını okumayacak; ilk satırı okuyacak.

    Uyarıların eşikleri modele göre değişiyor, çünkü iki modelin zaafları farklı:

      benzerlik — tek zaafı perspektif. Nesnenin referanstan UZAK olması tek
                  başına hata üretmez; ölçek her yerde aynıdır. Uzaklık ancak
                  perspektif varsa hatayı büyütür, o yüzden ikisini tek uyarıda
                  birleştiriyoruz. `dogrulama.benzerlik_taramasi` bunu ölçüyor:
                  tam tepeden çekimde para nerede olursa olsun hata sıfır.
      projektif — homografi referans köşelerine oturtuluyor; ondan uzaklaştıkça
                  parametre hatası büyüyerek taşınıyor. Orada uzaklığın kendisi
                  bir risk.
    """
    uyari: list[dict] = []
    yuk, gen = boyut
    en_uzak = max(float(h.duzlem_disi_uyarisi(n)) for n in noktalar)
    ref_px = float(np.hypot(*(ref.resim[1] - ref.resim[0]))) if len(ref.resim) >= 2 else 0.0

    # 1) Perspektif — benzerlik modelinin tek gerçek zaafı ve en pahalı hata.
    if ref.aile == "benzerlik":
        sapma = kutu_sonucu.dikdortgenlik if kutu_sonucu is not None else None
        uzakta = en_uzak > 1.5
        if sapma is not None and sapma > 0.06:
            uyari.append(dict(seviye="yuksek", metin=(
                f"Karşılıklı kenarlar %{_tr(sapma * 100)} farklı ölçülüyor — fotoğraf "
                f"belirgin biçimde eğik çekilmiş. Tek uzunluktan kurulan ölçek "
                f"perspektifi DÜZELTMEZ; bu eğimde hata %10'u geçebilir ve aşağıdaki "
                f"hata payı bunu kapsamıyor. Kamerayı nesnenin tam üstünden düz tut, "
                f"ya da referans olarak dikdörtgen bir şey (kart, A4) kullanıp dört "
                f"köşesini işaretle — o zaman perspektifi düzeltebilirim." +
                (" Referansı nesnenin üstüne koymak da hatayı yarıya indirir."
                 if uzakta else ""))))
        elif sapma is not None and sapma > 0.02:
            uyari.append(dict(seviye="orta", metin=(
                f"Karşılıklı kenarlar %{_tr(sapma * 100)} farklı ölçülüyor: hafif bir "
                f"perspektif var ve hata payı bunu kapsamıyor." +
                (" Referans nesneden uzakta; üstüne koymak hatayı yaklaşık yarıya "
                 "indirir." if uzakta else " Kamerayı biraz daha dik tutmak düzeltir."))))
        else:
            uyari.append(dict(seviye="bilgi", metin=(
                "Tek uzunluktan ölçek kuruldu: perspektif düzeltilmiyor. Nesnenin "
                "karşılıklı kenarları eşit ölçüldüğü için fotoğraf yeterince dik "
                "görünüyor — eğik çekimde bu sayı büyür ve uyarı çıkar."
                if kutu_sonucu is not None else
                "Tek uzunluktan ölçek kuruldu: perspektif düzeltilmiyor. Fotoğraf "
                "nesnenin tam üstünden çekildiyse sonuç doğru.")))
    else:
        if en_uzak > 2.0:
            uyari.append(dict(seviye="yuksek", metin=(
                f"Ölçüm, referansın {_tr(en_uzak)} kutu boyu dışında. Bu bölgede hata "
                f"hızla büyüyor — referansı ölçtüğün şeye yaklaştır.")))
        elif en_uzak > 1.0:
            uyari.append(dict(seviye="orta", metin=(
                f"Ölçüm, referansın {_tr(en_uzak)} kutu boyu dışında; yaklaştırmak "
                f"hatayı düşürür.")))

    # 2) Referansın görüntüdeki boyu — belirsizliğin çoğu genelde buradan gelir.
    if 0 < ref_px < 150 and ref.aile == "benzerlik":
        uyari.append(dict(seviye="orta", metin=(
            f"Referans görüntüde yalnız {ref_px:.0f} piksel uzunluğunda. Ölçek "
            f"hatası kabaca σ√2 / {ref_px:.0f} px, yani hata payının çoğu buradan "
            f"geliyor. Daha büyük bir referans (kart, A4) ya da daha yakın çekim "
            f"aralığı belirgin biçimde daraltır.")))

    # 3) Kadraj ve uyum
    disarida = int(np.sum((noktalar[:, 0] < -2) | (noktalar[:, 0] > gen + 2) |
                          (noktalar[:, 1] < -2) | (noktalar[:, 1] > yuk + 2)))
    if disarida:
        uyari.append(dict(seviye="orta",
                          metin=f"{disarida} nokta kadrajın dışında."))
    if h.rms_px > 1.5:
        uyari.append(dict(seviye="yuksek", metin=(
            f"Yeniden izdüşüm hatası {_tr(h.rms_px, 2)} px — referans köşeleri iyi "
            f"oturmamış olabilir.")))
    if not h.yakinsadi:
        uyari.append(dict(seviye="yuksek",
                          metin="Homografi çözücüsü yakınsamadı; sonuca güvenme."))
    if en_kotu_bagil > 0.05:
        uyari.append(dict(seviye="orta", metin=(
            f"Bağıl belirsizlik %{_tr(en_kotu_bagil * 100)} — ilan edilen çalışma "
            f"bölgesinin (%3) dışındasın.")))

    # 4) Her zaman geçerli olan sınırlar
    uyari.append(dict(seviye="bilgi", metin=(
        "ArUco köşeleri alt piksel doğrulukta okundu." if ref.tur == "aruco" else
        f"Noktalar elle işaretlendi; tıklama gürültüsü {_tr(sigma_px, 2)} px "
        f"varsayıldı. Büyüteçle köşeye tam oturmak bu sayıyı gerçekten düşürür.")))
    uyari.append(dict(seviye="bilgi", metin=(
        "Düzlem varsayımı: ölçülen şey referansla aynı düzlemde (aynı yükseklikte) "
        "olmalı. Masadaki parayla masadaki nesneyi ölçebilirsin, raftakini "
        "ölçemezsin.")))
    return uyari


def _tr(deger: float, basamak: int = 1) -> str:
    """Türkçe ondalık ayracı. Panelin geri kalanı virgül kullanıyor; uyarı
    metinlerinin nokta kullanması aynı ekranda iki farklı sayı biçimi demek."""
    return f"{deger:.{basamak}f}".replace(".", ",")


def _olcum_sozlugu(o: belirsizlik.Olcum, ad: str = "") -> dict:
    return dict(ad=ad, deger=o.deger, std=o.std, alt=o.alt, ust=o.ust, guven=o.guven,
                yontem=o.yontem, birim=o.birim, bagil_hata=o.bagil_hata, metin=str(o))


# ------------------------------------------------------------------ ölçüm akışı
def _olc(govde: dict, depo: Depo) -> dict:
    gorsel = depo.al(str(govde.get("gorsel_id", "")))
    boyut = (gorsel.yukseklik, gorsel.genislik)

    ref = _referansi_coz(govde, gorsel)
    sigma_px = _sayi(govde, "sigma_px", ref.sigma, en_az=0.01, en_cok=20.0)
    guven = _sayi(govde, "guven", 0.95, en_az=0.5, en_cok=0.999)
    yontem = str(govde.get("yontem", "monte_carlo"))
    if yontem not in ("monte_carlo", "analitik"):
        raise IstekHatasi(f"Bilinmeyen yöntem '{yontem}'.")

    olcum_istek = govde.get("olcum") or {}
    tur = olcum_istek.get("tur", "kutu")
    if tur not in OLCUM_TURLERI:
        raise IstekHatasi(f"Bilinmeyen ölçüm türü '{tur}'.")
    kural = OLCUM_TURLERI[tur]
    noktalar = _noktalar(olcum_istek.get("noktalar"), "ölçüm noktaları",
                         kural["en_az"], kural["en_cok"], boyut)

    try:
        h = ref.kur()
    except (ValueError, np.linalg.LinAlgError) as hata:
        raise IstekHatasi(f"Referans kurulamadı: {hata}") from None

    notlar: list[dict] = []
    if yontem == "analitik" and ref.kur_fn is not None:
        # Analitik yayılım sekiz projektif parametrenin kovaryansına dayanıyor;
        # benzerlik modelinde o parametrelerin dördü serbest bile değil.
        yontem = "monte_carlo"
        notlar.append(dict(seviye="bilgi", metin=(
            "Analitik yayılım yalnız dört noktalı projektif referansta tanımlı; "
            "bu ölçüm Monte Carlo ile yapıldı.")))

    ek: dict = {}
    kutu_sonucu = None
    if tur == "gecit":
        fnler, gecit = _gecit_hazirla(h, noktalar, olcum_istek, boyut, ek)
        mc_noktalar = None
    elif tur == "kutu":
        kutu_sonucu = olcum.kutu(h, noktalar)
        fnler = [("en", lambda hh, nn: olcum.kutu(hh, nn).en_mm, "mm"),
                 ("boy", lambda hh, nn: olcum.kutu(hh, nn).boy_mm, "mm"),
                 ("alan", lambda hh, nn: olcum.kutu(hh, nn).alan_mm2 / 100.0, "cm²")]
        mc_noktalar, gecit = noktalar, None
    else:
        fnler = {
            "mesafe": [("mesafe", lambda hh, nn: olcum.mesafe(hh, nn[0], nn[1]), "mm")],
            "uzunluk": [("uzunluk", lambda hh, nn: olcum.uzunluk(hh, nn), "mm")],
            "alan": [("alan", lambda hh, nn: olcum.alan(hh, nn) / 100.0, "cm²")],
        }[tur]
        mc_noktalar, gecit = noktalar, None

    mc_tavan = GECIT_MC_TAVAN if tur == "gecit" else MC_ARALIK[1]
    mc_n = int(min(_sayi(govde, "mc_n", 400, en_az=MC_ARALIK[0], en_cok=MC_ARALIK[1]),
                   mc_tavan))

    baslangic = time.perf_counter()
    olculer = []
    for ad, fn, birim in fnler:
        try:
            if yontem == "analitik":
                sonuc = belirsizlik.analitik(h, ref.dunya, ref.resim, fn, mc_noktalar,
                                             sigma_px=sigma_px, guven=guven, birim=birim)
            else:
                # Aynı tohum: üç ölçü de AYNI bozulma kümesinden geliyor, yani
                # "en" ve "boy" birbiriyle tutarlı bir dünyada ölçülüyor.
                sonuc = belirsizlik.monte_carlo(ref.dunya, ref.resim, fn, mc_noktalar,
                                                sigma_px=sigma_px, n=mc_n, guven=guven,
                                                birim=birim, kur_fn=ref.kur_fn)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as hata:
            raise IstekHatasi(f"Ölçüm yapılamadı: {hata}") from None
        olculer.append(_olcum_sozlugu(sonuc, ad))
    sure_ms = (time.perf_counter() - baslangic) * 1000.0

    # Alanın bağıl hatası uzunluğunkinin kabaca iki katı; eşiği o belirlerse
    # uyarı her ölçümde tetiklenir. Uzunluk ölçülerine bakıyoruz.
    uzunluklar = [o["bagil_hata"] for o in olculer if o["birim"] == "mm"]
    en_kotu = max(uzunluklar or [o["bagil_hata"] for o in olculer])
    yanit = dict(
        tur=tur,
        olculer=olculer,
        olcum=olculer[0],                       # birincil ölçü
        referans=dict(tur=ref.tur, aile=ref.aile, etiket=ref.etiket,
                      sigma_px=sigma_px, noktalar=ref.resim.tolist(),
                      piksel_boyu=(float(np.hypot(*(ref.resim[1] - ref.resim[0])))
                                   if len(ref.resim) >= 2 else None)),
        homografi=dict(rms_px=h.rms_px, model=h.model, yakinsadi=bool(h.yakinsadi),
                       olcek_mm_px=float(h.olcek_mm_px(noktalar.mean(axis=0)))),
        uyarilar=notlar + _uyarilar(h, noktalar, ref, en_kotu, boyut, sigma_px,
                                    kutu_sonucu),
        sure_ms=sure_ms,
        **ek,
    )

    if kutu_sonucu is not None:
        yanit["kutu"] = dict(
            kenarlar_mm=kutu_sonucu.kenarlar_mm.tolist(),
            dikdortgenlik=kutu_sonucu.dikdortgenlik,
            kosegen_mm=kutu_sonucu.kosegen_mm,
        )
    if tur in ("mesafe", "uzunluk"):
        d = h.dunyaya(noktalar)
        yanit["parcalar"] = [float(np.hypot(*(d[i + 1] - d[i])))
                             for i in range(len(d) - 1)]
    if gecit is not None:
        yanit["gecit"] = _gecit_ozeti(h, gecit, olcum_istek, olculer[0])
    return yanit


def _gecit_hazirla(h: Homografi, noktalar, istek, boyut, ek):
    maske = _maske_kur(noktalar, boyut)
    adim = _sayi(istek, "adim_mm", 20.0, en_az=1.0, en_cok=1000.0)
    ornek_mm = _sayi(istek, "ornek_mm", 5.0, en_az=0.5, en_cok=200.0)
    kenar_payi = _sayi(istek, "kenar_payi_mm", None, en_az=0.0, en_cok=100_000.0)
    try:
        gecit = olcum.en_dar_gecit(h, maske, adim_mm=adim, ornek_mm=ornek_mm,
                                   kenar_payi_mm=kenar_payi)
    except ValueError as hata:
        raise IstekHatasi(str(hata)) from None

    # Eksen Monte Carlo boyunca sabit kalmalı: her örnekte yeniden PCA yapmak
    # ölçülen şeyi değiştirir, belirsizliği değil.
    def fn(hh, _):
        return olcum.en_dar_gecit(hh, maske, eksen=gecit.eksen, adim_mm=adim,
                                  ornek_mm=ornek_mm,
                                  kenar_payi_mm=gecit.kenar_payi_mm).genislik_mm

    ek["maske_alani_px"] = int(maske.sum())
    return [("gecit", fn, "mm")], gecit


def _gecit_ozeti(h: Homografi, gecit: olcum.Gecit, istek: dict, sonuc: dict) -> dict:
    dik = np.array([-gecit.eksen[1], gecit.eksen[0]])
    uclar_mm = np.array([gecit.konum_mm - gecit.genislik_mm / 2.0 * dik,
                         gecit.konum_mm + gecit.genislik_mm / 2.0 * dik])
    ozet = dict(
        genislik_mm=gecit.genislik_mm,
        kenar_payi_mm=gecit.kenar_payi_mm,
        eksen=gecit.eksen.tolist(),
        istasyonlar_mm=gecit.istasyonlar_mm.tolist(),
        profil_mm=gecit.profil_mm.tolist(),
        en_dar_indeks=int(np.argmin(gecit.profil_mm)),
        cizgi_px=h.resme(uclar_mm).tolist(),
    )
    ayak_izi = _sayi(istek, "ayak_izi_mm", None, en_az=0.0, en_cok=100_000.0)
    if ayak_izi is not None:
        pay = _sayi(istek, "pay_mm", 0.0, en_az=0.0, en_cok=100_000.0)
        gerekli = ayak_izi + pay
        # Karar aralığın ALT ucuna göre: geçemeyeceği yola girmek, geçebileceği
        # yolu kaçırmaktan pahalı. Asimetrik maliyet, asimetrik eşik.
        ozet["karar"] = dict(
            ayak_izi_mm=ayak_izi, pay_mm=pay, gerekli_mm=gerekli,
            gecer=bool(sonuc["alt"] >= gerekli),
            nokta_tahmini_gecer=bool(sonuc["deger"] >= gerekli),
            marj_mm=float(sonuc["alt"] - gerekli),
        )
    return ozet


# ------------------------------------------------------------------ uygulama
def uygulama_kur(*, depo: Depo | None = None) -> Flask:
    uygulama = Flask(__name__)
    uygulama.config["MAX_CONTENT_LENGTH"] = EN_BUYUK_YUKLEME
    # Flask 3'te eski JSON_SORT_KEYS ayarı okunmuyor; anahtarları alfabetik
    # sıralamak yanıttaki anlamlı sırayı (örneğin "önce mesafe, sonra alan")
    # sessizce bozuyordu.
    uygulama.json.sort_keys = False
    kutu = depo if depo is not None else Depo()

    @uygulama.errorhandler(IstekHatasi)
    def _istek_hatasi(hata: IstekHatasi):
        return jsonify(hata=hata.mesaj), hata.kod

    @uygulama.errorhandler(413)
    def _cok_buyuk(_):
        return jsonify(hata=f"Dosya çok büyük (sınır "
                            f"{EN_BUYUK_YUKLEME // (1024 * 1024)} MB)."), 413

    @uygulama.get("/")
    def panel():
        return render_template(
            "panel.html",
            surum=__version__,
            uzunluklar=[{"ad": ad, "mm": mm, "aciklama": aciklama}
                        for ad, (mm, aciklama) in referans.BILINEN_UZUNLUKLAR.items()],
            nesneler={ad: list(b) for ad, b in referans.BILINEN_NESNELER.items()},
            sigmalar=referans.TIPIK_SIGMA_PX,
            # Panel tek akış sürüyor (nesnenin ölçüsü); geçit sahnesi o
            # akışa uymadığı için düğmelerde yok — CLI ve API'de duruyor.
            ornekler=list(ornek.PANEL_SAHNELERI),
        )

    @uygulama.get("/gorsel/<kimlik>")
    def gorsel_ver(kimlik: str):
        g = kutu.al(kimlik)
        return send_file(io.BytesIO(g.veri), mimetype=g.mime,
                         download_name=g.ad, max_age=3600)

    @uygulama.post("/api/gorsel")
    def gorsel_yukle():
        dosya = request.files.get("dosya")
        if dosya is None or not dosya.filename:
            raise IstekHatasi("Dosya alanı boş.")
        veri = dosya.read()
        if not veri:
            raise IstekHatasi("Dosya boş.")
        return jsonify(kutu.ekle(_gorseli_coz(veri, dosya.filename)).ozet())

    @uygulama.post("/api/ornek")
    def ornek_uret():
        cv2 = _cv2()
        ad = str((request.get_json(silent=True) or {}).get("ad", "duz"))
        try:
            sahne = ornek.ornek_sahne(ad)
        except KeyError as hata:
            raise IstekHatasi(str(hata)) from None
        basarili, tampon = cv2.imencode(".png", sahne["goruntu"])
        if not basarili:                                       # pragma: no cover
            raise IstekHatasi("Örnek sahne kodlanamadı.")
        g = kutu.ekle(Gorsel(kimlik=uuid.uuid4().hex, dizi=sahne["goruntu"],
                             veri=tampon.tobytes(), mime="image/png",
                             ad=sahne["ad"]))
        return jsonify({**g.ozet(),
                        "aciklama": sahne["aciklama"],
                        "varsayilan_olcum": sahne["varsayilan_olcum"],
                        "referans": sahne["referans"],
                        "gercek": sahne["gercek"],
                        "ipucu": sahne["ipucu"],
                        "ayak_izi_mm": sahne.get("ayak_izi_mm")})

    @uygulama.post("/api/aruco")
    def aruco_ara():
        govde = request.get_json(silent=True) or {}
        g = kutu.al(str(govde.get("gorsel_id", "")))
        kenar = _sayi(govde, "kenar_mm", zorunlu=True, en_az=1.0, en_cok=100_000.0)
        sozluk = str(govde.get("sozluk", "DICT_4X4_50"))
        _, resim, kimlik = _aruco_bul(g, kenar, sozluk)
        return jsonify(koseler=resim.tolist(), isaret_id=kimlik,
                       etiket=f"ArUco #{kimlik} · {kenar:g} mm",
                       sigma_px=referans.TIPIK_SIGMA_PX["aruco"])

    @uygulama.post("/api/olc")
    def olc():
        return jsonify(_olc(request.get_json(silent=True) or {}, kutu))

    @uygulama.get("/api/durum")
    def durum():
        try:
            import cv2
            cv_surum = cv2.__version__
        except ImportError:                                    # pragma: no cover
            cv_surum = None
        return jsonify(surum=__version__, opencv=cv_surum,
                       en_buyuk_yukleme_mb=EN_BUYUK_YUKLEME // (1024 * 1024))

    uygulama.depo = kutu          # testlerin ve gömülü kullanımın erişimi için
    return uygulama


def calistir(host: str = "127.0.0.1", port: int = 8000, hata_ayikla: bool = False) -> None:
    """Geliştirme sunucusunu başlatır."""
    uygulama = uygulama_kur()
    print(f"metrik-goz paneli:  http://{host}:{port}")
    uygulama.run(host=host, port=port, debug=hata_ayikla, threaded=True)

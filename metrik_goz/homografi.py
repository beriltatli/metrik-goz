"""
Düzlem homografisi: görüntü pikselleri ile dünya düzlemi (mm) arasındaki eşleme.

Varsayım — ve bu paketin en önemli sınırı: ölçülecek her şey referans nesneyle
AYNI DÜZLEM üzerinde olmalı. Tezgâhın üstündeki domatesi tezgâha koyduğun
kartla ölçebilirsin; rafta duran kutuyu ölçemezsin. Bu varsayım kırıldığında
hata sessizce büyür, bu yüzden `Homografi.duzlem_disi_uyarisi` ile ne kadar
uzağa ekstrapolasyon yaptığını raporluyoruz.

Kurulum iki adımlı:
  1) DLT — Hartley normalizasyonuyla, kapalı formda başlangıç çözümü.
  2) LM  — geometrik yeniden izdüşüm hatasını minimize eden iyileştirme.

Neden iki adım: DLT cebirsel hatayı minimize eder, bu da gürültü altında
geometrik olarak en iyi çözüm DEĞİLDİR. Ölçüm hatası iddiası için geometrik
hatayı minimize etmemiz gerekiyor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lm import coz


# ------------------------------------------------------------------ yardımcı
def _normalizasyon_matrisi(noktalar: np.ndarray) -> np.ndarray:
    """
    Hartley normalizasyonu: ağırlık merkezini orijine taşı, ortalama uzaklığı
    sqrt(2) yap. DLT'nin sayısal koşullanması buna bağlı; atlanırsa piksel
    değerlerinin karesi (10^6 mertebesi) tasarım matrisini bozuyor.
    """
    merkez = noktalar.mean(axis=0)
    kaydirilmis = noktalar - merkez
    ort_uzaklik = np.sqrt((kaydirilmis ** 2).sum(axis=1)).mean()
    if ort_uzaklik < 1e-12:
        olcek = 1.0
    else:
        olcek = np.sqrt(2.0) / ort_uzaklik
    return np.array([
        [olcek, 0.0, -olcek * merkez[0]],
        [0.0, olcek, -olcek * merkez[1]],
        [0.0, 0.0, 1.0],
    ])


# w'nin (projektif ölçek) altına inmesine izin verdiğimiz taban.
_W_TABAN = 1e-12


def _homojen(noktalar: np.ndarray) -> np.ndarray:
    return np.hstack([noktalar, np.ones((len(noktalar), 1))])


def _uygula(H: np.ndarray, noktalar: np.ndarray) -> np.ndarray:
    """
    H ile projektif dönüşüm; (N,2) -> (N,2).

    w (üçüncü bileşen) sıfıra yaklaşırsa nokta ufuk çizgisindedir; orada
    gerçek bir cevap yok. Bölmeyi patlatmamak için w'yi işaretini KORUYARAK
    tabana oturtuyoruz — işaret kaybolursa nokta düzlemin yanlış yarısına
    düşer ve hata sessizce yayılır.
    """
    noktalar = np.atleast_2d(np.asarray(noktalar, dtype=float))
    hn = _homojen(noktalar) @ H.T
    w = hn[:, 2:3]
    kucuk = np.abs(w) < _W_TABAN
    if kucuk.any():
        w = np.where(kucuk, np.where(w < 0.0, -_W_TABAN, _W_TABAN), w)
    return hn[:, :2] / w


# ------------------------------------------------------------------ DLT
def dlt(dunya: np.ndarray, resim: np.ndarray) -> np.ndarray:
    """
    Doğrudan doğrusal dönüşüm: dünya (mm) -> resim (px) homografisi.
    En az 4 nokta gerekir, üçü doğrusal olmamalı.
    """
    dunya = np.asarray(dunya, dtype=float)
    resim = np.asarray(resim, dtype=float)
    if len(dunya) < 4:
        raise ValueError("Homografi için en az 4 nokta gerekiyor.")
    if len(dunya) != len(resim):
        raise ValueError("Dünya ve resim nokta sayıları eşleşmiyor.")

    T_d = _normalizasyon_matrisi(dunya)
    T_r = _normalizasyon_matrisi(resim)
    d = _uygula(T_d, dunya)
    r = _uygula(T_r, resim)

    A = []
    for (X, Y), (u, v) in zip(d, r):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_r) @ H_norm @ T_d
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Dejenere homografi: h33 sıfıra çok yakın.")
    return H / H[2, 2]


# ------------------------------------------------------------------ LM iyileştirme
def _artiklar(p: np.ndarray, dunya: np.ndarray, resim: np.ndarray) -> np.ndarray:
    """Yeniden izdüşüm hatası, piksel cinsinden, düzleştirilmiş."""
    H = np.append(p, 1.0).reshape(3, 3)
    return (_uygula(H, dunya) - resim).ravel()


def _jacobian(p: np.ndarray, dunya: np.ndarray, resim: np.ndarray) -> np.ndarray:
    """
    Artıkların 8 parametreye göre analitik türevi.

    u = a/w,  a = h11*X + h12*Y + h13
    v = b/w,  b = h21*X + h22*Y + h23
              w = h31*X + h32*Y + 1
    """
    h11, h12, h13, h21, h22, h23, h31, h32 = p
    X = dunya[:, 0]
    Y = dunya[:, 1]
    a = h11 * X + h12 * Y + h13
    b = h21 * X + h22 * Y + h23
    w = h31 * X + h32 * Y + 1.0
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)

    n = len(X)
    J = np.zeros((2 * n, 8))
    # du satırları (çift indisler)
    J[0::2, 0] = X / w
    J[0::2, 1] = Y / w
    J[0::2, 2] = 1.0 / w
    J[0::2, 6] = -a * X / w ** 2
    J[0::2, 7] = -a * Y / w ** 2
    # dv satırları (tek indisler)
    J[1::2, 3] = X / w
    J[1::2, 4] = Y / w
    J[1::2, 5] = 1.0 / w
    J[1::2, 6] = -b * X / w ** 2
    J[1::2, 7] = -b * Y / w ** 2
    return J


# ------------------------------------------------------------------ ana sınıf
@dataclass
class Homografi:
    """
    Dünya düzlemi (mm) ile görüntü (px) arasındaki eşleme ve kalitesi.

    H          : dünya -> resim
    H_ters     : resim -> dünya (ölçüm bu yönü kullanır)
    rms_px     : yeniden izdüşüm hatasının RMS'i, piksel
    kovaryans  : 8 homografi parametresinin kovaryansı (LM'den)
    referans_kutu : referansın dünya koordinatlarındaki sınırları — ekstrapolasyon
                    uyarısı bunun üstünden hesaplanır
    """

    H: np.ndarray
    H_ters: np.ndarray
    rms_px: float
    kovaryans: np.ndarray | None
    referans_kutu: tuple[float, float, float, float]
    yakinsadi: bool
    model: str = "projektif"      # "projektif" | "benzerlik" — aşağıda anlatılıyor

    # -------------------------------------------------------------- kurucu
    @classmethod
    def kur(cls, dunya_mm, resim_px, *, iyilestir: bool = True,
            kovaryans: bool = True) -> "Homografi":
        """
        `kovaryans=False`: LM'nin çözüm sonrası kovaryans kestirimini atlar.
        Monte Carlo bunu binlerce kez kurar ve orada kovaryansa bakmayız —
        atlamak örnek başına bir Jacobian ve bir matris tersi kazandırıyor.
        """
        dunya = np.asarray(dunya_mm, dtype=float)
        resim = np.asarray(resim_px, dtype=float)

        H = dlt(dunya, resim)
        rms = float(np.sqrt(np.mean((_uygula(H, dunya) - resim) ** 2)))
        kov = None
        yakinsadi = True

        # 4 nokta ile 8 parametre: serbestlik derecesi yok, LM'nin iyileştirecek
        # bir şeyi de yok — DLT çözümü zaten tam uyuyor.
        if iyilestir and len(dunya) >= 5:
            p0 = (H / H[2, 2]).ravel()[:8]
            sonuc = coz(
                lambda p: _artiklar(p, dunya, resim),
                p0,
                lambda p: _jacobian(p, dunya, resim),
                kovaryans_hesapla=kovaryans,
            )
            H = np.append(sonuc.p, 1.0).reshape(3, 3)
            rms = sonuc.rms
            kov = sonuc.kovaryans
            yakinsadi = sonuc.yakinsadi

        try:
            H_ters = np.linalg.inv(H)
        except np.linalg.LinAlgError as hata:
            raise ValueError("Homografi tersinir değil: referans köşeleri "
                             "doğrusal ya da çakışık olabilir.") from hata

        return cls(
            H=H,
            H_ters=H_ters,
            rms_px=rms,
            kovaryans=kov,
            referans_kutu=(
                float(dunya[:, 0].min()), float(dunya[:, 1].min()),
                float(dunya[:, 0].max()), float(dunya[:, 1].max()),
            ),
            yakinsadi=yakinsadi,
        )

    # -------------------------------------------------------------- benzerlik
    @classmethod
    def olcekten(cls, p1_px, p2_px, uzunluk_mm: float) -> "Homografi":
        """
        Tek bir bilinen uzunluktan (madeni paranın çapı, kartın uzun kenarı)
        BENZERLİK homografisi: ölçek + döndürme + öteleme.

        Neden ayrı bir kurucu ve neden "projektif" değil: iki nokta ve bir
        uzunluk üç sayı taşıyor, projektif dönüşümün ise sekiz serbestliği var.
        Eksik bilgiyi uydurmak yerine daha dar bir model kuruyoruz — perspektif
        DÜZELTİLMİYOR, yalnız ölçek biliniyor.

        Bu modelin geçerli olduğu yer: kamera düzleme dik bakıyor ve ölçtüğün
        şey referansla aynı derinlikte. Eğik bakışta hata sessizce büyür; bu
        yüzden `model` alanı "benzerlik" olarak işaretleniyor ve ölçüm katmanı
        (dikdörtgenlik sapması üzerinden) eğikliği yakalayıp uyarabiliyor.

        Perspektifi gerçekten düzeltmek için dört nokta gerekiyor: dikdörtgen
        bir referansın (kart, A4) köşeleri ya da bir ArUco işareti — `kur`.
        """
        p1 = np.asarray(p1_px, dtype=float).reshape(2)
        p2 = np.asarray(p2_px, dtype=float).reshape(2)
        if uzunluk_mm <= 0:
            raise ValueError("Referans uzunluğu pozitif olmalı.")
        v = p2 - p1
        boy_px = float(np.hypot(*v))
        if boy_px < 1e-6:
            raise ValueError("Referansın iki ucu çakışık; aralarında mesafe olmalı.")

        yon = v / boy_px
        dik = np.array([-yon[1], yon[0]])
        orta = (p1 + p2) / 2.0
        yari = boy_px / 2.0

        # Sentetik kare: köşeleri, gözlenen iki noktayı karşılıklı kenarların
        # orta noktası yapacak biçimde yerleştiriliyor. Böylece dört nokta
        # eşlemesi tam olarak bir benzerlik dönüşümü veriyor ve `referans_kutu`
        # gerçekten referansın çevresine oturuyor.
        resim = np.array([
            orta - yari * yon - yari * dik,
            orta + yari * yon - yari * dik,
            orta + yari * yon + yari * dik,
            orta - yari * yon + yari * dik,
        ])
        yari_mm = uzunluk_mm / 2.0
        dunya = np.array([
            [-yari_mm, -yari_mm], [yari_mm, -yari_mm],
            [yari_mm, yari_mm], [-yari_mm, yari_mm],
        ])
        h = cls.kur(dunya, resim, iyilestir=False)
        h.model = "benzerlik"
        return h

    # -------------------------------------------------------------- dönüşüm
    def dunyaya(self, resim_noktalari) -> np.ndarray:
        """Piksel -> mm (düzlem üzerinde)."""
        return _uygula(self.H_ters, resim_noktalari)

    def resme(self, dunya_noktalari) -> np.ndarray:
        """mm -> piksel."""
        return _uygula(self.H, dunya_noktalari)

    # -------------------------------------------------------------- kalite
    def olcek_mm_px(self, resim_noktasi) -> float:
        """
        Verilen piksel civarında yerel ölçek (mm / piksel).

        Homografi projektif olduğu için ölçek görüntü boyunca sabit değil —
        uzaktaki piksel daha çok milimetreye karşılık gelir. Bu fonksiyon
        yerel Jacobian'ın tekil değerlerinin geometrik ortalamasını döner.
        """
        nokta = np.asarray(resim_noktasi, dtype=float).reshape(2)
        h = 0.5
        merkez = self.dunyaya(nokta)[0]
        dx = self.dunyaya(nokta + [h, 0])[0] - merkez
        dy = self.dunyaya(nokta + [0, h])[0] - merkez
        J = np.column_stack([dx / h, dy / h])
        return float(np.sqrt(abs(np.linalg.det(J))))

    def duzlem_disi_uyarisi(self, resim_noktasi) -> float:
        """
        Ölçülen noktanın referans kutusunun kaç katı dışına düştüğü.

        0 ise nokta referansın içinde; 1 ise kutu genişliği kadar dışında.
        Deneyimsel eşik: 2'nin üstünde ölçüme güvenme, referansı ölçtüğün
        şeye yaklaştır.
        """
        d = self.dunyaya(resim_noktasi)[0]
        x0, y0, x1, y1 = self.referans_kutu
        gen = max(x1 - x0, 1e-9)
        yuk = max(y1 - y0, 1e-9)
        dx = max(x0 - d[0], d[0] - x1, 0.0) / gen
        dy = max(y0 - d[1], d[1] - y1, 0.0) / yuk
        return float(np.hypot(dx, dy))

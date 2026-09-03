"""
Referans nesne tespiti — ölçeği sahneye sokan şey.

Tek fotoğraftan mutlak ölçü çıkarmanın matematiksel olarak tek yolu, sahnede
boyutu bilinen bir şeyin olması. Kamera ne kadar iyi olursa olsun, referanssız
görüntüden "bu 40 cm" bilgisi çıkarılamaz — ölçek serbest kalır.

İki yol destekleniyor:
  ArUco işareti  — en güvenilir; köşe konumu alt piksel doğrulukta, kimliği
                   belli, otomatik bulunur. Yazdırıp mutfağa/sahaya koyman
                   yeterli.
  Bilinen nesne  — kredi kartı, A4 kâğıt gibi standart boyutlu bir şeyin dört
                   köşesini elle vermek. Elde işaret yokken kurtarır ama köşe
                   okuması daha gürültülüdür (~1,5 px).
"""

from __future__ import annotations

import numpy as np

# Standart boyutlu, her yerde bulunan referanslar (mm)
BILINEN_NESNELER: dict[str, tuple[float, float]] = {
    "kredi_karti": (85.60, 53.98),   # ISO/IEC 7810 ID-1
    "a4": (297.0, 210.0),
    "a5": (210.0, 148.0),
    "cd": (120.0, 120.0),
    "post_it": (76.0, 76.0),
}

# Tek bir bilinen UZUNLUK taşıyan, herkesin cebinde/çekmecesinde olan şeyler.
# Madeni para en pratiği: yuvarlak olduğu için hangi yönde ölçtüğün fark etmez,
# fotoğrafta hangi açıyla dursa çapı çaptır.
#
# ad -> (uzunluk_mm, insan tarafından okunur açıklama)
BILINEN_UZUNLUKLAR: dict[str, tuple[float, str]] = {
    # Türk lirası madeni paraları, çap (TCMB 2009 serisi)
    "1_tl":            (26.15, "1 TL — çap"),
    "50_kurus":        (23.85, "50 kuruş — çap"),
    "25_kurus":        (20.50, "25 kuruş — çap"),
    "10_kurus":        (18.50, "10 kuruş — çap"),
    "5_kurus":         (17.50, "5 kuruş — çap"),
    "1_kurus":         (16.50, "1 kuruş — çap"),
    # Yaygın diğer paralar
    "2_euro":          (25.75, "2 euro — çap"),
    "1_euro":          (23.25, "1 euro — çap"),
    "50_cent":         (24.25, "50 euro sent — çap"),
    "us_quarter":      (24.26, "ABD 25 cent — çap"),
    # Standart nesneler
    "kredi_karti_uzun": (85.60, "kredi kartı — uzun kenar"),
    "kredi_karti_kisa": (53.98, "kredi kartı — kısa kenar"),
    "a4_uzun":         (297.0, "A4 kâğıt — uzun kenar"),
    "a4_kisa":         (210.0, "A4 kâğıt — kısa kenar"),
    "cd":              (120.0, "CD/DVD — çap"),
    "aa_pil":          (50.5,  "AA kalem pil — boy"),
}

# Köşe okuma gürültüsü için deneyimsel değerler (piksel, std)
TIPIK_SIGMA_PX: dict[str, float] = {
    "aruco": 0.4,
    "elle": 1.5,
}


def kare_dunya(kenar_mm: float, merkez_mm=(0.0, 0.0)) -> np.ndarray:
    """Merkezi verilen, kenarı `kenar_mm` olan karenin köşeleri (sol üstten saat yönü)."""
    y = kenar_mm / 2.0
    cx, cy = merkez_mm
    return np.array([
        [cx - y, cy - y],
        [cx + y, cy - y],
        [cx + y, cy + y],
        [cx - y, cy + y],
    ])


def dikdortgen_dunya(genislik_mm: float, yukseklik_mm: float, merkez_mm=(0.0, 0.0)) -> np.ndarray:
    gx, gy = genislik_mm / 2.0, yukseklik_mm / 2.0
    cx, cy = merkez_mm
    return np.array([
        [cx - gx, cy - gy],
        [cx + gx, cy - gy],
        [cx + gx, cy + gy],
        [cx - gx, cy + gy],
    ])


def bilinen_nesne(ad: str, merkez_mm=(0.0, 0.0)) -> np.ndarray:
    """Tablodaki standart nesnenin dünya köşeleri."""
    if ad not in BILINEN_NESNELER:
        raise KeyError(f"Bilinmeyen referans '{ad}'. Seçenekler: {sorted(BILINEN_NESNELER)}")
    g, y = BILINEN_NESNELER[ad]
    return dikdortgen_dunya(g, y, merkez_mm)


def aruco_bul(goruntu, kenar_mm: float, *, sozluk: str = "DICT_4X4_50",
              isaret_id: int | None = None):
    """
    Görüntüdeki ArUco işaretini bulur.

    Döner: (dunya_mm, resim_px, isaret_id)
    Köşe sırası ArUco'nun kendi sırası (sol üst, sağ üst, sağ alt, sol alt) ile
    `kare_dunya` sırası aynı tutulmuştur.
    """
    import cv2  # yalnız bu yol için gerekli; çekirdek matematik cv2'siz çalışır

    if goruntu.ndim == 3:
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    else:
        gri = goruntu

    sozluk_nesnesi = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, sozluk))
    parametreler = cv2.aruco.DetectorParameters()
    # Alt piksel köşe iyileştirmesi: belirsizlik iddiamız buna dayanıyor
    parametreler.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dedektor = cv2.aruco.ArucoDetector(sozluk_nesnesi, parametreler)

    koseler, kimlikler, _ = dedektor.detectMarkers(gri)
    if kimlikler is None or len(kimlikler) == 0:
        raise ValueError("Görüntüde ArUco işareti bulunamadı.")

    kimlikler = kimlikler.ravel()
    if isaret_id is None:
        secim = 0
    else:
        eslesme = np.nonzero(kimlikler == isaret_id)[0]
        if len(eslesme) == 0:
            raise ValueError(f"{isaret_id} kimlikli işaret yok. Bulunanlar: {kimlikler.tolist()}")
        secim = int(eslesme[0])

    resim_px = koseler[secim].reshape(4, 2).astype(float)
    return kare_dunya(kenar_mm), resim_px, int(kimlikler[secim])

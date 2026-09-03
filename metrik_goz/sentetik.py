"""
Sentetik sahne üreteci — doğrulamanın omurgası.

Gerçek fotoğrafta "doğru cevap" yoktur; olsa zaten ölçmeye gerek kalmazdı.
Bu yüzden belirsizlik iddiasını kanıtlamanın tek yolu, doğru cevabı bizim
koyduğumuz sahneler üretmek: kamerayı biz yerleştiriyoruz, referansı biz
koyuyoruz, ölçülecek mesafeyi biz biliyoruz. Sonra sisteme sadece pikselleri
verip ne kadar isabet ettiğine ve güven aralığının gerçekten tutup tutmadığına
bakıyoruz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sahne:
    """Doğru cevabı bilinen sentetik ölçüm sahnesi."""

    H_gercek: np.ndarray          # dünya (mm) -> resim (px), gürültüsüz
    referans_dunya: np.ndarray    # referans köşeleri, mm
    referans_px: np.ndarray       # referans köşeleri, px (gürültüsüz)
    boyut_px: tuple[int, int]     # (yükseklik, genişlik)

    def izdusur(self, dunya_noktalari) -> np.ndarray:
        d = np.atleast_2d(np.asarray(dunya_noktalari, dtype=float))
        h = np.hstack([d, np.ones((len(d), 1))]) @ self.H_gercek.T
        return h[:, :2] / h[:, 2:3]

    def gorunur_mu(self, px) -> bool:
        yuk, gen = self.boyut_px
        px = np.atleast_2d(px)
        return bool(np.all((px[:, 0] >= 0) & (px[:, 0] < gen) &
                           (px[:, 1] >= 0) & (px[:, 1] < yuk)))


def kamera_homografisi(
    *,
    odak_px: float,
    merkez_px: tuple[float, float],
    mesafe_mm: float,
    egim_derece: float,
    azimut_derece: float = 0.0,
) -> np.ndarray:
    """
    z=0 düzlemi ile görüntü arasındaki homografi.

    `egim_derece` = 0 tam tepeden (nadir) bakış; büyüdükçe bakış yatıklaşır ve
    perspektif kısalma artar. Ölçüm hatasının en çok büyüdüğü parametre budur,
    o yüzden doğrulamada süpürülüyor.
    """
    egim = np.radians(egim_derece)
    azimut = np.radians(azimut_derece)

    # Kamera konumu: düzlemin üstünde, verilen eğim ve azimutta
    C = mesafe_mm * np.array([
        np.sin(egim) * np.cos(azimut),
        np.sin(egim) * np.sin(azimut),
        np.cos(egim),
    ])

    zc = -C / np.linalg.norm(C)                      # kamera ileri ekseni: orijine bak
    dunya_yukari = np.array([0.0, 0.0, 1.0])
    if abs(zc @ dunya_yukari) > 0.999:               # tam nadir: başka bir yukarı seç
        dunya_yukari = np.array([0.0, 1.0, 0.0])
    xc = np.cross(dunya_yukari, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)

    R = np.vstack([xc, yc, zc])                      # dünya -> kamera
    t = -R @ C

    K = np.array([
        [odak_px, 0.0, merkez_px[0]],
        [0.0, odak_px, merkez_px[1]],
        [0.0, 0.0, 1.0],
    ])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return H / H[2, 2]


def sahne_kur(
    *,
    referans_boyut_mm: float = 100.0,
    referans_merkez_mm: tuple[float, float] = (0.0, 0.0),
    odak_px: float = 1400.0,
    boyut_px: tuple[int, int] = (1080, 1920),
    mesafe_mm: float = 1200.0,
    egim_derece: float = 25.0,
    azimut_derece: float = 0.0,
) -> Sahne:
    """
    Tipik bir sahne: masaya konmuş kare referans, ondan biraz uzakta ölçülecek
    şeyler. Varsayılanlar bir mutfak tezgâhı senaryosuna karşılık geliyor
    (1,2 m mesafe, hafif eğik bakış, 100 mm'lik referans).
    """
    yuk, gen = boyut_px
    H = kamera_homografisi(
        odak_px=odak_px,
        merkez_px=(gen / 2.0, yuk / 2.0),
        mesafe_mm=mesafe_mm,
        egim_derece=egim_derece,
        azimut_derece=azimut_derece,
    )
    yari = referans_boyut_mm / 2.0
    cx, cy = referans_merkez_mm
    ref_dunya = np.array([
        [cx - yari, cy - yari],
        [cx + yari, cy - yari],
        [cx + yari, cy + yari],
        [cx - yari, cy + yari],
    ])
    h = np.hstack([ref_dunya, np.ones((4, 1))]) @ H.T
    ref_px = h[:, :2] / h[:, 2:3]

    return Sahne(H_gercek=H, referans_dunya=ref_dunya, referans_px=ref_px, boyut_px=boyut_px)


def gurultule(noktalar, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Piksel koordinatlarına bağımsız Gauss gürültüsü ekler."""
    noktalar = np.asarray(noktalar, dtype=float)
    return noktalar + rng.normal(0.0, sigma_px, size=noktalar.shape)

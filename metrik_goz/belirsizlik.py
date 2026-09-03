"""
Belirsizlik yayılımı — bu paketin asıl iddiası.

Bir ölçüm sistemi "412 mm" diyorsa bu tek başına bilgi değil; "412 ± 9 mm,
%95 güven" bilgi.

Hata iki ayrı yerden geliyor ve İKİSİNİ birden saymazsak aralık yalancı
biçimde dar çıkar:

  1) Referans köşelerinin piksel gürültüsü  -> homografi yanlış kurulur
  2) Ölçtüğün noktaların piksel gürültüsü   -> doğru homografide yanlış yer

Çoğu "tek fotoğraftan ölçüm" kodu yalnızca birincisini sayar (ya da hiçbirini)
ve bu yüzden %95 dediği aralık gerçekte %60 tutar. Buradaki kapsama testi
(`testler/test_kapsama.py`) tam olarak bunu kontrol ediyor.

İki yöntem:
  Monte Carlo : her iki gürültüyü de örnekle, homografiyi yeniden kur, ölç.
                Yavaş ama varsayımsız — referans doğrusu bu.
  Analitik    : birinci mertebeden yayılım. Hızlı; MC'ye karşı doğrulanıyor.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .homografi import Homografi, _jacobian  # analitik Jacobian'ı paylaşıyoruz


@dataclass
class Olcum:
    """Bir ölçüm ve dürüst hata payı."""

    deger: float
    std: float
    alt: float
    ust: float
    guven: float
    yontem: str
    birim: str = "mm"

    def __str__(self) -> str:
        return (f"{self.deger:.1f} ± {self.std:.1f} {self.birim} "
                f"(%{self.guven * 100:.0f}: {self.alt:.1f}–{self.ust:.1f})")

    @property
    def bagil_hata(self) -> float:
        return float(self.std / abs(self.deger)) if self.deger else float("inf")


# ------------------------------------------------------------------ kovaryans
def parametre_kovaryansi(dunya_mm, resim_px, H: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Homografinin 8 parametresinin kovaryansı: sigma_px^2 * (J^T J)^-1.

    Artıklardan değil, dışarıdan verilen köşe gürültüsünden hesaplıyoruz.
    Sebebi: 4 köşeli bir referansta 8 artık ve 8 parametre var, yani
    serbestlik derecesi sıfır — artıklar tanımı gereği sıfıra yakın çıkar ve
    onlardan okunan belirsizlik anlamsız biçimde küçük olur. Köşe gürültüsü
    ise ölçülebilir bir büyüklük (ArUco köşesi için tipik 0,3–0,7 piksel).
    """
    dunya = np.asarray(dunya_mm, dtype=float)
    resim = np.asarray(resim_px, dtype=float)
    p = (H / H[2, 2]).ravel()[:8]
    J = _jacobian(p, dunya, resim)
    JtJ = J.T @ J
    try:
        ters = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        ters = np.linalg.pinv(JtJ)
    return (sigma_px ** 2) * ters


# ------------------------------------------------------------------ analitik
def analitik(
    homografi: Homografi,
    dunya_mm,
    resim_px,
    olcum_fn,
    noktalar_px=None,
    *,
    sigma_px: float = 0.5,
    sigma_nokta_px: float | None = None,
    guven: float = 0.95,
    birim: str = "mm",
) -> Olcum:
    """
    Birinci mertebeden yayılım:
        var = grad_p^T Cov_p grad_p  +  sigma_nokta^2 * ||grad_x||^2

    olcum_fn(homografi, noktalar_px) -> float
    `noktalar_px` None ise ölçüm tıklanan noktalara bağlı değildir (örneğin
    maske tabanlı geçit ölçümü) ve yalnızca homografi belirsizliği taşınır.
    """
    if sigma_nokta_px is None:
        sigma_nokta_px = sigma_px

    Kov = parametre_kovaryansi(dunya_mm, resim_px, homografi.H, sigma_px)
    p0 = (homografi.H / homografi.H[2, 2]).ravel()[:8]
    noktalar = None if noktalar_px is None else np.asarray(noktalar_px, dtype=float)
    g0 = float(olcum_fn(homografi, noktalar))

    # Homografi parametrelerine göre gradyan
    grad_p = np.zeros(8)
    for i in range(8):
        h = 1e-6 * max(1.0, abs(p0[i]))
        ileri, geri = p0.copy(), p0.copy()
        ileri[i] += h
        geri[i] -= h
        grad_p[i] = (olcum_fn(_homografiden(ileri, homografi), noktalar)
                     - olcum_fn(_homografiden(geri, homografi), noktalar)) / (2 * h)
    var = float(grad_p @ Kov @ grad_p)

    # Tıklanan noktaların gürültüsü
    if noktalar is not None and sigma_nokta_px > 0:
        h = 1e-4
        kareler = 0.0
        duz = noktalar.ravel()
        for i in range(duz.size):
            ileri, geri = duz.copy(), duz.copy()
            ileri[i] += h
            geri[i] -= h
            tur = (olcum_fn(homografi, ileri.reshape(noktalar.shape))
                   - olcum_fn(homografi, geri.reshape(noktalar.shape))) / (2 * h)
            kareler += tur ** 2
        var += (sigma_nokta_px ** 2) * kareler

    std = float(np.sqrt(max(var, 0.0)))
    z = _z_degeri(guven)
    return Olcum(g0, std, g0 - z * std, g0 + z * std, guven, "analitik", birim)


# ------------------------------------------------------------------ Monte Carlo
def monte_carlo(
    dunya_mm,
    resim_px,
    olcum_fn,
    noktalar_px=None,
    *,
    sigma_px: float = 0.5,
    sigma_nokta_px: float | None = None,
    n: int = 400,
    guven: float = 0.95,
    tohum: int | None = 0,
    birim: str = "mm",
    kur_fn=None,
) -> Olcum:
    """
    Hem referans gözlemlerini hem ölçülen noktaları bozup her seferinde
    homografiyi yeniden kurar. Sonuç ölçümün gerçek örneklem dağılımıdır.

    Güven aralığı yüzdeliklerden alınır; dağılım simetrik olmasa bile
    (uzak ekstrapolasyonda olmuyor) doğru kalır.

    `kur_fn(bozuk_resim_px) -> Homografi`: referans MODELİNİ dışarıdan verir.
    Varsayılan dört nokta eşlemesi; ama tek bir bilinen uzunluktan kurulan
    benzerlik modelinde (`Homografi.olcekten`) gözlem iki noktadan ibaret ve
    kalan iki köşe onlardan TÜRETİLİYOR. O iki köşeyi bağımsız bozmak olmayan
    bir gürültü uydurmak olurdu; hangi sayıların gerçekten gözlem olduğunu
    modelin kendisi bilir, bu yüzden kurulumu ona bırakıyoruz.

    `kur_fn` verildiğinde `dunya_mm` None olabilir.
    """
    if sigma_nokta_px is None:
        sigma_nokta_px = sigma_px

    if n < 2:
        raise ValueError("Monte Carlo için en az 2 örnek gerekiyor.")

    dunya = None if dunya_mm is None else np.asarray(dunya_mm, dtype=float)
    resim = np.asarray(resim_px, dtype=float)
    noktalar = None if noktalar_px is None else np.asarray(noktalar_px, dtype=float)
    rng = np.random.default_rng(tohum)

    if kur_fn is None:
        if dunya is None:
            raise ValueError("kur_fn verilmediğinde dunya_mm gerekli.")
        iyilestir = len(dunya) >= 5
        kur_fn = lambda gozlem: Homografi.kur(dunya, gozlem, iyilestir=iyilestir,
                                              kovaryans=False)

    # Gürültüyü döngü içinde tek tek değil, tek seferde üretiyoruz: aynı sayılar,
    # çağrı başına yeniden kurulan üretici maliyeti yok.
    ref_gurultu = rng.normal(0.0, sigma_px, size=(n, *resim.shape))
    nokta_gurultu = (None if noktalar is None or sigma_nokta_px <= 0
                     else rng.normal(0.0, sigma_nokta_px, size=(n, *noktalar.shape)))

    ornekler = np.empty(n)
    basarili = 0
    for i in range(n):
        bozuk_ref = resim + ref_gurultu[i]
        if noktalar is None:
            bozuk_nokta = None
        elif nokta_gurultu is None:
            bozuk_nokta = noktalar
        else:
            bozuk_nokta = noktalar + nokta_gurultu[i]
        try:
            ornekler[basarili] = float(olcum_fn(kur_fn(bozuk_ref), bozuk_nokta))
            basarili += 1
        except (ValueError, np.linalg.LinAlgError):
            continue

    if basarili < max(20, n // 10):
        raise RuntimeError("Monte Carlo örneklerinin çoğu başarısız — homografi dejenereye yakın.")
    ornekler = ornekler[:basarili]

    kuyruk = (1.0 - guven) / 2.0
    alt, ust = np.percentile(ornekler, [100 * kuyruk, 100 * (1 - kuyruk)])
    return Olcum(
        deger=float(np.mean(ornekler)),
        std=float(np.std(ornekler, ddof=1)),
        alt=float(alt),
        ust=float(ust),
        guven=guven,
        yontem=f"monte_carlo(n={basarili})",
        birim=birim,
    )


# ------------------------------------------------------------------ yardımcı
def _homografiden(p: np.ndarray, sablon: Homografi) -> Homografi:
    """8 parametreden geçici Homografi (gradyan hesabı için)."""
    H = np.append(p, 1.0).reshape(3, 3)
    return Homografi(
        H=H, H_ters=np.linalg.inv(H), rms_px=sablon.rms_px,
        kovaryans=None, referans_kutu=sablon.referans_kutu, yakinsadi=sablon.yakinsadi,
    )


def _z_degeri(guven: float) -> float:
    """Çift taraflı normal kritik değer. Standart kütüphane yetiyor, scipy gerekmez."""
    if not 0.0 < guven < 1.0:
        raise ValueError(f"Güven 0 ile 1 arasında olmalı, {guven} verildi.")
    return float(NormalDist().inv_cdf(0.5 + guven / 2.0))

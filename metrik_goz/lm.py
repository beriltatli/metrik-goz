"""
Levenberg-Marquardt — elle yazılmış sönümlemeli en küçük kareler çözücüsü.

Neden hazır `scipy.optimize.least_squares` kullanmıyoruz: bu paketin bütün
belirsizlik iddiası çözücünün ürettiği kovaryansa dayanıyor. Kovaryansın
nereden geldiğini bilmiyorsak "±3 cm" demeye hakkımız yok. Bu yüzden
çözücü de bizim.

Çözülen problem:
    min_p  ||r(p)||^2
Gauss-Newton adımı (J^T J) dp = -J^T r denklemini çözer; LM bunu
(J^T J + lambda * diag(J^T J)) dp = -J^T r  haline getirir. lambda büyükse
adım gradyan inişine, küçükse Gauss-Newton'a yaklaşır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sonuc:
    """LM çözümünün sonucu."""

    p: np.ndarray                      # bulunan parametreler
    maliyet: float                     # son 0.5 * ||r||^2
    artiklar: np.ndarray               # son artık vektörü
    yakinsadi: bool
    adim_sayisi: int
    durma_nedeni: str
    kovaryans: np.ndarray | None = None   # parametrelerin kovaryansı
    gecmis: list[float] = field(default_factory=list)

    @property
    def rms(self) -> float:
        """Artıkların karekök ortalaması — ölçüm biriminde hata."""
        return float(np.sqrt(np.mean(self.artiklar ** 2)))


def sayisal_jacobian(artik_fn, p: np.ndarray, adim: float = 1e-7) -> np.ndarray:
    """Merkezi farkla Jacobian. Analitik Jacobian'ı doğrulamak için de kullanılır."""
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(artik_fn(p), dtype=float)
    J = np.zeros((r0.size, p.size))
    for i in range(p.size):
        h = adim * max(1.0, abs(p[i]))
        ileri, geri = p.copy(), p.copy()
        ileri[i] += h
        geri[i] -= h
        J[:, i] = (np.asarray(artik_fn(ileri)) - np.asarray(artik_fn(geri))) / (2 * h)
    return J


def coz(
    artik_fn,
    p0,
    jacobian_fn=None,
    *,
    maks_adim: int = 100,
    lambda0: float = 1e-3,
    tol_maliyet: float = 1e-12,
    tol_adim: float = 1e-12,
    tol_gradyan: float = 1e-12,
    kovaryans_hesapla: bool = True,
) -> Sonuc:
    """
    Artık fonksiyonunu en küçük kareler anlamında minimize eder.

    artik_fn(p) -> (m,) artık vektörü
    jacobian_fn(p) -> (m, n) Jacobian; verilmezse merkezi farkla hesaplanır.

    Kovaryans, yakınsama noktasında  s^2 * (J^T J)^-1  ile kestirilir;
    burada s^2 = ||r||^2 / (m - n) artık varyansıdır. Yani ölçüm gürültüsünü
    dışarıdan varsaymak yerine uydurma artığından okuyoruz.
    """
    p = np.asarray(p0, dtype=float).copy()
    jac = jacobian_fn if jacobian_fn is not None else (lambda q: sayisal_jacobian(artik_fn, q))

    r = np.asarray(artik_fn(p), dtype=float)
    maliyet = 0.5 * float(r @ r)
    lam = lambda0
    gecmis = [maliyet]
    neden = "maks_adim"
    yakinsadi = False
    adim_no = 0          # maks_adim=0 ise döngü hiç dönmez; yine de raporluyoruz

    for adim_no in range(1, maks_adim + 1):
        J = np.asarray(jac(p), dtype=float)
        g = J.T @ r                       # gradyan
        if np.max(np.abs(g)) < tol_gradyan:
            neden, yakinsadi = "gradyan", True
            break

        H = J.T @ J
        kosegen = np.diag(np.maximum(np.diag(H), 1e-12))

        # Kabul edilen bir adım bulana kadar sönümlemeyi büyüt.
        kabul = False
        for _ in range(30):
            try:
                dp = np.linalg.solve(H + lam * kosegen, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            p_yeni = p + dp
            r_yeni = np.asarray(artik_fn(p_yeni), dtype=float)
            maliyet_yeni = 0.5 * float(r_yeni @ r_yeni)

            if maliyet_yeni < maliyet:
                # Adım işe yaradı: sönümlemeyi gevşet, Gauss-Newton'a yaklaş.
                azalma = maliyet - maliyet_yeni
                p, r, maliyet = p_yeni, r_yeni, maliyet_yeni
                lam = max(lam * 0.3, 1e-12)
                kabul = True
                gecmis.append(maliyet)
                if azalma < tol_maliyet or np.linalg.norm(dp) < tol_adim:
                    neden, yakinsadi = "maliyet" if azalma < tol_maliyet else "adim", True
                break

            lam *= 10.0

        if not kabul:
            neden, yakinsadi = "sonumleme_doydu", True
            break
        if yakinsadi:
            break

    # Kovaryans son parametrede değerlendirilmiş Jacobian ister. Döngü bir adım
    # kabul ettiyse elimizdeki J bir önceki p'ye ait, o yüzden yeniden kuruyoruz.
    kov = _kovaryans(jac(p), r) if kovaryans_hesapla else None

    return Sonuc(
        p=p, maliyet=maliyet, artiklar=r, yakinsadi=yakinsadi,
        adim_sayisi=adim_no, durma_nedeni=neden, kovaryans=kov, gecmis=gecmis,
    )


def _kovaryans(J: np.ndarray, r: np.ndarray) -> np.ndarray | None:
    """s^2 (J^T J)^-1. Serbestlik derecesi kalmadıysa None döner."""
    m, n = J.shape
    sd = m - n
    if sd <= 0:
        return None
    s2 = float(r @ r) / sd
    H = J.T @ J
    try:
        return s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return s2 * np.linalg.pinv(H)

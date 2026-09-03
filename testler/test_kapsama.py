"""
Kapsama testi — bu paketin asıl iddiasının sınavı.

Sistem "%95 güven aralığı" diyorsa, çok sayıda bağımsız ölçümde gerçek değer
o aralığın içinde gerçekten %95 oranında düşmeli. Tutmuyorsa hata payı
süslemedir.

Bu test sentetik sahnelerle çalışır çünkü gerçek fotoğrafta doğru cevap yoktur.
Kamera konumunu, referansı ve ölçülecek mesafeyi biz koyuyoruz; sisteme yalnız
gürültülü pikselleri veriyoruz.
"""

import numpy as np
import pytest

from metrik_goz import Homografi, belirsizlik, olcum
from metrik_goz.sentetik import gurultule, sahne_kur

SIGMA_PX = 0.5


def _mesafe_fn(h, n):
    return olcum.mesafe(h, n[0], n[1])


def _deneme(rng, *, mesafe_mm, egim, ref_mm=100.0, uzaklik_kati=1.0, mc_n=120):
    """Tek bir sentetik ölçüm; gerçek değeri ve üretilen aralıkları döner."""
    sahne = sahne_kur(referans_boyut_mm=ref_mm, mesafe_mm=mesafe_mm,
                      egim_derece=egim, azimut_derece=float(rng.uniform(0, 360)))

    r = uzaklik_kati * ref_mm
    aci = rng.uniform(0, 2 * np.pi)
    merkez = r * np.array([np.cos(aci), np.sin(aci)])
    yon = rng.uniform(0, 2 * np.pi)
    boy = rng.uniform(0.5, 2.0) * ref_mm
    yon_vec = np.array([np.cos(yon), np.sin(yon)])
    a, b = merkez - 0.5 * boy * yon_vec, merkez + 0.5 * boy * yon_vec

    a_px, b_px = sahne.izdusur(a)[0], sahne.izdusur(b)[0]
    if not (sahne.gorunur_mu([a_px, b_px]) and sahne.gorunur_mu(sahne.referans_px)):
        return None

    ref_g = gurultule(sahne.referans_px, SIGMA_PX, rng)
    nokta_g = gurultule(np.array([a_px, b_px]), SIGMA_PX, rng)
    gercek = float(np.hypot(*(b - a)))

    mc = belirsizlik.monte_carlo(sahne.referans_dunya, ref_g, _mesafe_fn, nokta_g,
                                 sigma_px=SIGMA_PX, n=mc_n,
                                 tohum=int(rng.integers(1 << 30)))
    h = Homografi.kur(sahne.referans_dunya, ref_g)
    an = belirsizlik.analitik(h, sahne.referans_dunya, ref_g, _mesafe_fn, nokta_g,
                              sigma_px=SIGMA_PX)
    return gercek, mc, an


def _kosu(tohum, n=120, **kw):
    rng = np.random.default_rng(tohum)
    denemeler = [d for d in (_deneme(rng, **kw) for _ in range(n)) if d]
    assert len(denemeler) > n // 2, "sahnelerin çoğu kadraja sığmadı"
    return denemeler


# --------------------------------------------------------------- asıl testler
def test_monte_carlo_kapsamasi_yuzde_95():
    """
    %95 aralığı gerçekten %95 civarında tutmalı.

    Altı bağımsız tohumda ölçülen gerçek kapsama %94,4. Nominal %95'ten bir
    puanlık eksiklik gerçek ve açıklanabilir: parametrik bootstrap, dağılımı
    gerçek köşeler etrafında değil GÖZLENEN (yani zaten gürültülü) köşeler
    etrafında merkezliyor. Aralığın genişliği doğru, merkezi biraz kayıyor.
    Bunu gizlemek yerine yazıyoruz; %95 diyip %80 tutturmaktan iyidir.
    """
    denemeler = _kosu(11, mesafe_mm=1200, egim=25.0)
    kapsama = np.mean([mc.alt <= g <= mc.ust for g, mc, _ in denemeler])
    assert 0.88 <= kapsama <= 0.99, f"kapsama {kapsama:.3f}, %94 civarı olmalıydı"


def test_analitik_kapsamasi_yuzde_95():
    denemeler = _kosu(12, mesafe_mm=1200, egim=25.0)
    kapsama = np.mean([an.alt <= g <= an.ust for g, _, an in denemeler])
    assert 0.88 <= kapsama <= 0.99, f"kapsama {kapsama:.3f}, %94 civarı olmalıydı"


def test_analitik_monte_carlo_ile_ortusuyor():
    """
    Birinci mertebeden yayılım, varsayımsız Monte Carlo'yla aynı büyüklüğü
    vermeli. Ayrışırlarsa analitik yol bu çalışma bölgesinde geçersizdir.
    """
    denemeler = _kosu(13, n=50, mesafe_mm=1200, egim=25.0)
    oranlar = np.array([an.std / max(mc.std, 1e-9) for _, mc, an in denemeler])
    assert 0.80 <= np.median(oranlar) <= 1.25, f"AN/MC std oranı {np.median(oranlar):.2f}"


@pytest.mark.parametrize("egim", [0.0, 25.0, 50.0])
def test_kapsama_bakis_acisindan_bagimsiz(egim):
    """Bakış açısı yatıklaştıkça hata büyür ama aralık hâlâ tutmalı."""
    denemeler = _kosu(20 + int(egim), n=90, mesafe_mm=1200, egim=egim)
    kapsama = np.mean([mc.alt <= g <= mc.ust for g, mc, _ in denemeler])
    assert 0.86 <= kapsama <= 0.99, f"egim {egim}: kapsama {kapsama:.3f}"


def test_calisma_bolgesinde_hata_yuzde_3_altinda():
    """
    İlan edilen çalışma bölgesi: 100 mm referans, 0,5 px köşe gürültüsü,
    ≤ 2 m mesafe, referans boyutunun ≤ 2 katı uzaklık. Bu bölgede ortanca
    bağıl hata %3'ün altında kalmalı.
    """
    denemeler = _kosu(31, n=70, mesafe_mm=1200, egim=25.0, uzaklik_kati=1.0)
    hatalar = np.array([abs(mc.deger - g) / g for g, mc, _ in denemeler])
    assert np.median(hatalar) < 0.03, f"ortanca hata %{np.median(hatalar) * 100:.2f}"


def test_uzakta_belirsizlik_buyuyor():
    """
    Referanstan uzaklaştıkça sistem daha az emin olduğunu SÖYLEMELİ.
    Sessizce yanılmak, hata payını büyütmekten çok daha kötü.
    """
    yakin = _kosu(41, n=40, mesafe_mm=1200, egim=25.0, uzaklik_kati=0.5)
    uzak = _kosu(42, n=40, mesafe_mm=1200, egim=25.0, uzaklik_kati=4.0)

    y = np.median([mc.std / g for g, mc, _ in yakin])
    u = np.median([mc.std / g for g, mc, _ in uzak])
    assert u > 1.5 * y, f"uzakta bağıl belirsizlik büyümedi ({y:.4f} -> {u:.4f})"

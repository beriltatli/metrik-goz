"""LM çözücüsünün ve analitik Jacobian'ın doğruluğu."""

import numpy as np
import pytest

from metrik_goz import lm
from metrik_goz.homografi import _artiklar, _jacobian
from metrik_goz.sentetik import sahne_kur


def test_bilinen_probleme_yakinsiyor():
    """y = a*exp(b*x) uydurması: LM doğru parametreleri bulmalı."""
    x = np.linspace(0, 2, 40)
    gercek = np.array([2.5, -0.9])
    y = gercek[0] * np.exp(gercek[1] * x)

    sonuc = lm.coz(lambda p: p[0] * np.exp(p[1] * x) - y, [1.0, -0.1])

    assert sonuc.yakinsadi
    np.testing.assert_allclose(sonuc.p, gercek, rtol=1e-6)
    assert sonuc.maliyet < 1e-18


def test_gurultulu_veride_kovaryans_makul():
    """Gürültü büyüdükçe parametre belirsizliği de büyümeli."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2, 200)
    y0 = 2.5 * np.exp(-0.9 * x)

    stdler = []
    for sigma in (0.01, 0.05):
        y = y0 + rng.normal(0, sigma, x.size)
        s = lm.coz(lambda p: p[0] * np.exp(p[1] * x) - y, [1.0, -0.1])
        stdler.append(np.sqrt(np.diag(s.kovaryans))[0])

    assert stdler[1] > stdler[0] * 2


def test_analitik_jacobian_sayisalla_ayni():
    """Homografi Jacobian'ı elle türetildi; sayısal türevle tutmalı."""
    sahne = sahne_kur(egim_derece=30.0)
    dunya = sahne.referans_dunya
    resim = sahne.referans_px
    p = (sahne.H_gercek / sahne.H_gercek[2, 2]).ravel()[:8]

    J_analitik = _jacobian(p, dunya, resim)
    J_sayisal = lm.sayisal_jacobian(lambda q: _artiklar(q, dunya, resim), p)

    olcek = np.maximum(np.abs(J_sayisal).max(axis=0), 1e-9)
    fark = np.abs(J_analitik - J_sayisal) / olcek
    assert fark.max() < 1e-5, f"en büyük bağıl fark {fark.max():.2e}"


@pytest.mark.parametrize("egim", [0.0, 20.0, 45.0, 60.0])
def test_jacobian_farkli_acilarda_tutarli(egim):
    sahne = sahne_kur(egim_derece=egim)
    dunya, resim = sahne.referans_dunya, sahne.referans_px
    p = (sahne.H_gercek / sahne.H_gercek[2, 2]).ravel()[:8]

    J_a = _jacobian(p, dunya, resim)
    J_s = lm.sayisal_jacobian(lambda q: _artiklar(q, dunya, resim), p)
    olcek = np.maximum(np.abs(J_s).max(axis=0), 1e-9)
    assert (np.abs(J_a - J_s) / olcek).max() < 1e-5

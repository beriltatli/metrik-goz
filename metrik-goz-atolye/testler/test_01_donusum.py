"""
ADIM 1'in testleri.

Çalıştır:  pytest testler/test_01_donusum.py -v

Hepsi yeşile dönene kadar adım 1 bitmiş sayılmaz.
"""

import numpy as np
import pytest

from metrik_goz.homografi import _homojen, _uygula


# --------------------------------------------------------------- _homojen
def test_homojen_sutun_ekliyor():
    noktalar = np.array([[3.0, 4.0], [7.0, 1.0]])
    sonuc = _homojen(noktalar)

    assert sonuc.shape == (2, 3), "çıktı (N, 3) olmalı"
    np.testing.assert_allclose(sonuc, [[3, 4, 1], [7, 1, 1]])


def test_homojen_tek_nokta():
    sonuc = _homojen(np.array([[5.0, 9.0]]))
    np.testing.assert_allclose(sonuc, [[5, 9, 1]])


# --------------------------------------------------------------- _uygula
def test_birim_matris_hicbir_sey_yapmaz():
    """H birim matris ise noktalar aynen çıkmalı. En basit kontrol."""
    noktalar = np.array([[10.0, 20.0], [-3.0, 7.5]])
    np.testing.assert_allclose(_uygula(np.eye(3), noktalar), noktalar)


def test_oteleme():
    """
    Son sütun öteleme yapar:
        [[1, 0, 5],
         [0, 1, -2],
         [0, 0, 1]]
    her noktayı x'te +5, y'de -2 kaydırmalı.
    """
    H = np.array([[1.0, 0.0, 5.0],
                  [0.0, 1.0, -2.0],
                  [0.0, 0.0, 1.0]])
    noktalar = np.array([[0.0, 0.0], [10.0, 10.0]])
    np.testing.assert_allclose(_uygula(H, noktalar), [[5.0, -2.0], [15.0, 8.0]])


def test_olcekleme():
    """Köşegendeki 2 ve 3, x'i iki, y'yi üç katına çıkarmalı."""
    H = np.diag([2.0, 3.0, 1.0])
    np.testing.assert_allclose(_uygula(H, [[4.0, 5.0]]), [[8.0, 15.0]])


def test_perspektif_bolme():
    """
    İşin can alıcı kısmı: alt satırdaki 0.01, x büyüdükçe w'yi büyütür ve
    noktayı geri çeker.

    Nokta (100, 50) için:
        w = 0.01 * 100 + 1 = 2
        sonuç = (100/2, 50/2) = (50, 25)
    """
    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.01, 0.0, 1.0]])
    np.testing.assert_allclose(_uygula(H, [[100.0, 50.0]]), [[50.0, 25.0]])


def test_tek_nokta_da_calisiyor():
    """(2,) biçiminde tek bir nokta verilse de (1, 2) dönmeli."""
    sonuc = _uygula(np.eye(3), [3.0, 8.0])
    assert sonuc.shape == (1, 2)
    np.testing.assert_allclose(sonuc, [[3.0, 8.0]])


def test_kare_yamuga_donusuyor():
    """
    Asıl gösteri: perspektifli bir H, kareyi yamuğa çeviriyor.
    Üst kenar alt kenardan dar çıkmalı — fotoğrafta uzaklaşan yol gibi.
    """
    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.004, 1.0]])
    kare = np.array([[-100.0, 0.0], [100.0, 0.0],
                     [100.0, 200.0], [-100.0, 200.0]])
    y = _uygula(H, kare)

    alt_genislik = y[1, 0] - y[0, 0]
    ust_genislik = y[2, 0] - y[3, 0]
    assert ust_genislik < alt_genislik, "üst kenar daralmalıydı"
    assert alt_genislik == pytest.approx(200.0)
    assert ust_genislik == pytest.approx(200.0 / 1.8)

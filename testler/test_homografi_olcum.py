"""Homografi kurulumu ve ölçüm katmanının doğruluğu."""

import numpy as np
import pytest

from metrik_goz import Homografi, olcum
from metrik_goz.sentetik import gurultule, sahne_kur


def test_gurultusuzde_tam_isabet():
    """Gürültü yoksa ölçüm gerçek değeri neredeyse tam vermeli."""
    sahne = sahne_kur(egim_derece=25.0)
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)

    a_dunya, b_dunya = np.array([-150.0, 60.0]), np.array([220.0, -40.0])
    gercek = float(np.hypot(*(b_dunya - a_dunya)))
    olculen = olcum.mesafe(h, sahne.izdusur(a_dunya)[0], sahne.izdusur(b_dunya)[0])

    assert abs(olculen - gercek) / gercek < 1e-9


def test_alan_dogru():
    sahne = sahne_kur(egim_derece=35.0)
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)

    kare = np.array([[-80.0, -80.0], [80.0, -80.0], [80.0, 80.0], [-80.0, 80.0]])
    gercek = 160.0 ** 2
    olculen = olcum.alan(h, sahne.izdusur(kare))

    assert abs(olculen - gercek) / gercek < 1e-9


def test_lm_dlt_den_iyi():
    """Gürültü altında LM iyileştirmesi yeniden izdüşüm hatasını düşürmeli."""
    rng = np.random.default_rng(3)
    sahne = sahne_kur(egim_derece=35.0)

    # 5+ nokta: LM'nin iyileştirecek serbestlik derecesi olsun
    ek_dunya = np.array([[-200.0, 150.0], [180.0, -170.0], [40.0, 210.0]])
    dunya = np.vstack([sahne.referans_dunya, ek_dunya])
    resim = gurultule(sahne.izdusur(dunya), 1.0, rng)

    sadece_dlt = Homografi.kur(dunya, resim, iyilestir=False)
    lm_ile = Homografi.kur(dunya, resim, iyilestir=True)

    assert lm_ile.rms_px <= sadece_dlt.rms_px + 1e-12
    assert lm_ile.yakinsadi


def test_dort_noktadan_az_reddediliyor():
    with pytest.raises(ValueError):
        Homografi.kur(np.zeros((3, 2)), np.zeros((3, 2)))


def test_ekstrapolasyon_uyarisi_artiyor():
    """Referanstan uzaklaştıkça uyarı değeri büyümeli."""
    sahne = sahne_kur(referans_boyut_mm=100.0, egim_derece=20.0)
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)

    yakin = h.duzlem_disi_uyarisi(sahne.izdusur([[30.0, 0.0]])[0])
    uzak = h.duzlem_disi_uyarisi(sahne.izdusur([[400.0, 0.0]])[0])

    assert yakin == pytest.approx(0.0, abs=1e-6)
    assert uzak > 3.0


def test_yerel_olcek_uzakta_buyuyor():
    """
    Eğik bakışta uzaktaki piksel daha çok milimetreye karşılık gelir.

    Örnekleme derinlik yönünde yapılmalı: `sahne_kur` varsayılan azimutta
    kamerayı +X tarafına koyuyor, yani derinlik X ile değişiyor, Y ile değil.
    Y boyunca örneklersek iki nokta da aynı derinlikte kalır ve ölçek 1e-13
    mertebesinde farkla eşit çıkar — testin ölçtüğü şey kaybolur.
    """
    sahne = sahne_kur(egim_derece=55.0)
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)

    yakin_px = sahne.izdusur([[200.0, 0.0]])[0]     # kameraya yakın taraf
    uzak_px = sahne.izdusur([[-400.0, 0.0]])[0]     # uzak taraf
    assert sahne.gorunur_mu([yakin_px, uzak_px])

    assert h.olcek_mm_px(uzak_px) > 1.5 * h.olcek_mm_px(yakin_px)


def test_en_dar_gecit_bilinen_koridoru_buluyor():
    """
    Dünya düzleminde 300 mm geniş, ortasında 120 mm'ye daralan bir koridor
    kuruyoruz ve sistemin daralmayı bulmasını bekliyoruz.
    """
    sahne = sahne_kur(egim_derece=15.0, mesafe_mm=2500.0, referans_boyut_mm=200.0)
    yuk, gen = sahne.boyut_px

    # Koridoru dünya koordinatında tanımla, piksel maskesine boya
    yy, xx = np.mgrid[0:yuk, 0:gen]
    px = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    h_gercek_ters = np.linalg.inv(sahne.H_gercek)
    hn = np.hstack([px, np.ones((len(px), 1))]) @ h_gercek_ters.T
    dunya = hn[:, :2] / hn[:, 2:3]
    X, Y = dunya[:, 0], dunya[:, 1]

    yari = np.where(np.abs(X) < 250.0, 60.0, 150.0)     # ortada 120 mm, dışta 300 mm
    serbest = (np.abs(Y) < yari) & (np.abs(X) < 900.0)
    maske = serbest.reshape(yuk, gen)

    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)
    gecit = olcum.en_dar_gecit(h, maske, eksen=[1.0, 0.0], adim_mm=25.0, ornek_mm=4.0)

    assert abs(gecit.genislik_mm - 120.0) < 12.0
    assert gecit.gecer_mi(100.0)
    assert not gecit.gecer_mi(150.0)


# ----------------------------------------------------------------- benzerlik modeli
def test_olcekten_tepeden_bakista_tam_isabet():
    """
    Tam tepeden bakışta (nadir) ölçek düzlemin her yerinde aynı; tek bilinen
    uzunluktan kurulan benzerlik modeli orada KUSURSUZ olmalı — referanstan ne
    kadar uzakta ölçtüğün fark etmemeli. Modelin geçerlilik iddiası bu.
    """
    sahne = sahne_kur(referans_boyut_mm=26.15, egim_derece=0.0, mesafe_mm=520.0)
    uc = sahne.izdusur([[-13.075, 0.0], [13.075, 0.0]])
    h = Homografi.olcekten(uc[0], uc[1], 26.15)
    assert h.model == "benzerlik"

    for uzaklik in (0.0, 100.0, 250.0):
        a, b = np.array([uzaklik, -60.0]), np.array([uzaklik + 90.0, 40.0])
        gercek = float(np.hypot(*(b - a)))
        olculen = olcum.mesafe(h, sahne.izdusur(a)[0], sahne.izdusur(b)[0])
        assert abs(olculen - gercek) / gercek < 1e-9


def test_olcekten_egik_bakista_yaniliyor():
    """
    Aynı model eğik bakışta yanılmalı — ve bu bir kusur değil, ilan edilmiş
    sınır. Sessizce doğru sanılması tehlikeli olan tam bu durum.
    """
    sahne = sahne_kur(referans_boyut_mm=26.15, egim_derece=35.0, mesafe_mm=520.0)
    uc = sahne.izdusur([[-13.075, 0.0], [13.075, 0.0]])
    h = Homografi.olcekten(uc[0], uc[1], 26.15)

    a, b = np.array([200.0, 0.0]), np.array([320.0, 0.0])
    gercek = float(np.hypot(*(b - a)))
    olculen = olcum.mesafe(h, sahne.izdusur(a)[0], sahne.izdusur(b)[0])
    assert abs(olculen - gercek) / gercek > 0.05


def test_olcekten_dejenere_girdiyi_reddediyor():
    with pytest.raises(ValueError):
        Homografi.olcekten([10.0, 10.0], [10.0, 10.0], 26.15)
    with pytest.raises(ValueError):
        Homografi.olcekten([0.0, 0.0], [50.0, 0.0], 0.0)


# ----------------------------------------------------------------- kutu ölçümü
def test_kutu_kenarlari_ve_alani_dogru():
    sahne = sahne_kur(egim_derece=30.0)
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)

    en, boy = 146.7, 71.5
    kose = np.array([[-en / 2, -boy / 2], [en / 2, -boy / 2],
                     [en / 2, boy / 2], [-en / 2, boy / 2]])
    k = olcum.kutu(h, sahne.izdusur(kose))

    assert k.en_mm == pytest.approx(en, rel=1e-9)
    assert k.boy_mm == pytest.approx(boy, rel=1e-9)
    assert k.alan_mm2 == pytest.approx(en * boy, rel=1e-9)
    assert k.kosegen_mm == pytest.approx(np.hypot(en, boy), rel=1e-9)
    # Projektif model perspektifi düzelttiği için sapma sıfır olmalı.
    assert k.dikdortgenlik < 1e-9


def test_kutu_dikdortgenligi_egikligi_ele_veriyor():
    """
    `dikdortgenlik`, kullanıcının bilmediği eğimin gözlenebilir vekili: ölçek
    modelinde eğik bakış nesneyi yamuk gösterir, karşılıklı kenarlar ayrışır.
    Uyarı katmanı bu sayıya dayanıyor, o yüzden eğimle büyümesi şart.
    """
    en, boy = 146.7, 71.5
    kose = np.array([[-en / 2, -boy / 2], [en / 2, -boy / 2],
                     [en / 2, boy / 2], [-en / 2, boy / 2]]) + np.array([150.0, 0.0])

    sapmalar = []
    for egim in (0.0, 15.0, 35.0):
        sahne = sahne_kur(referans_boyut_mm=26.15, egim_derece=egim, mesafe_mm=520.0)
        uc = sahne.izdusur([[-13.075, 0.0], [13.075, 0.0]])
        h = Homografi.olcekten(uc[0], uc[1], 26.15)
        sapmalar.append(olcum.kutu(h, sahne.izdusur(kose)).dikdortgenlik)

    assert sapmalar[0] < 1e-9
    assert sapmalar[0] < sapmalar[1] < sapmalar[2]


def test_kutu_dort_kose_istiyor():
    sahne = sahne_kur()
    h = Homografi.kur(sahne.referans_dunya, sahne.referans_px)
    with pytest.raises(ValueError):
        olcum.kutu(h, sahne.izdusur([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))

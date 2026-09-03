"""
Web panelinin arka ucu.

Buradaki asıl soru "uç nokta 200 döndü mü" değil — panelin ürettiği sayının
komut satırının ürettiği sayıyla AYNI olması. Aynı çekirdeği iki ayrı yerden
çağırıyoruz; bir gün ayrışırlarsa kullanıcı hangisine güveneceğini bilemez.
"""

import io

import numpy as np
import pytest

flask = pytest.importorskip("flask")
cv2 = pytest.importorskip("cv2")

from metrik_goz import Homografi, belirsizlik, olcum, referans
from metrik_goz.ornek import ornek_sahne
from metrik_goz.web.sunucu import Depo, uygulama_kur


@pytest.fixture(scope="module")
def istemci():
    return uygulama_kur().test_client()


@pytest.fixture(scope="module")
def tezgah(istemci):
    """Bir kez üretilen örnek sahne: ArUco'suyla birlikte."""
    sahne = istemci.post("/api/ornek", json={"ad": "tezgah"}).get_json()
    aruco = istemci.post("/api/aruco", json={"gorsel_id": sahne["kimlik"],
                                             "kenar_mm": 100}).get_json()
    return sahne, aruco


def _ref(aruco, kenar=100):
    return {"tur": "aruco", "kenar_mm": kenar, "koseler": aruco["koseler"]}


# ------------------------------------------------------------------ temel
def test_panel_aciliyor(istemci):
    yanit = istemci.get("/")
    assert yanit.status_code == 200
    govde = yanit.get_data(as_text=True)
    assert 'id="tuval"' in govde and "panel.js" in govde


def test_gorsel_yukle_ve_servis_et(istemci):
    goruntu = np.full((240, 320, 3), 200, np.uint8)
    veri = cv2.imencode(".png", goruntu)[1].tobytes()
    yanit = istemci.post("/api/gorsel",
                         data={"dosya": (io.BytesIO(veri), "kare.png")})
    assert yanit.status_code == 200
    ozet = yanit.get_json()
    assert (ozet["genislik"], ozet["yukseklik"]) == (320, 240)

    servis = istemci.get(ozet["url"])
    assert servis.status_code == 200
    # Sunucunun ölçtüğü ızgara ile tarayıcıya gidenin boyutu aynı olmalı;
    # ayrışırlarsa kullanıcının tıkladığı yer ile ölçülen yer farklı olur.
    geri = cv2.imdecode(np.frombuffer(servis.data, np.uint8), cv2.IMREAD_COLOR)
    assert geri.shape[:2] == (240, 320)


def test_depo_en_eskiyi_dusuruyor():
    from metrik_goz.web.sunucu import Gorsel, IstekHatasi

    depo = Depo(kapasite=2)
    for i in range(3):
        depo.ekle(Gorsel(kimlik=str(i), dizi=np.zeros((4, 4, 3), np.uint8),
                         veri=b"", mime="image/png", ad=f"{i}.png"))
    assert depo.al("2").kimlik == "2"
    with pytest.raises(IstekHatasi):
        depo.al("0")


# ------------------------------------------------------------------ ölçüm
def test_panel_ile_kutuphane_ayni_sayiyi_veriyor(istemci, tezgah):
    """Panelin döndürdüğü değer, aynı girdiyle kütüphanenin döndürdüğüdür."""
    sahne, aruco = tezgah
    noktalar = sahne["ipucu"]["mesafe"]
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": sahne["kimlik"], "referans": _ref(aruco),
        "olcum": {"tur": "mesafe", "noktalar": noktalar},
        "sigma_px": 0.4, "mc_n": 200, "guven": 0.95,
    }).get_json()

    dunya = referans.kare_dunya(100.0)
    resim = np.asarray(aruco["koseler"], float)
    beklenen = belirsizlik.monte_carlo(
        dunya, resim, lambda h, n: olcum.mesafe(h, n[0], n[1]),
        np.asarray(noktalar, float), sigma_px=0.4, n=200)

    assert yanit["olcum"]["deger"] == pytest.approx(beklenen.deger, rel=1e-12)
    assert yanit["olcum"]["alt"] == pytest.approx(beklenen.alt, rel=1e-12)


def test_olcum_gercek_degeri_yakaliyor(istemci, tezgah):
    """Sentetik sahnenin doğru cevabı güven aralığının içinde olmalı."""
    sahne, aruco = tezgah
    for tur, anahtar in (("mesafe", "mesafe"), ("alan", "alan")):
        yanit = istemci.post("/api/olc", json={
            "gorsel_id": sahne["kimlik"], "referans": _ref(aruco),
            "olcum": {"tur": tur, "noktalar": sahne["ipucu"][anahtar]},
            "mc_n": 300,
        }).get_json()
        gercek = sahne["gercek"][anahtar]["deger"]
        o = yanit["olcum"]
        assert o["alt"] <= gercek <= o["ust"], f"{tur}: {gercek} ∉ [{o['alt']}, {o['ust']}]"
        assert o["birim"] == sahne["gercek"][anahtar]["birim"]


def test_gecit_karari_alt_uca_gore(istemci):
    """
    Karar nokta tahminine değil aralığın ALT ucuna bakmalı: geçemeyeceği yola
    girmek, geçebileceği yolu kaçırmaktan pahalı.
    """
    sahne = istemci.post("/api/ornek", json={"ad": "gecit"}).get_json()
    aruco = istemci.post("/api/aruco", json={"gorsel_id": sahne["kimlik"],
                                             "kenar_mm": 200}).get_json()
    ortak = {"gorsel_id": sahne["kimlik"], "referans": _ref(aruco, 200), "mc_n": 60}

    yanit = istemci.post("/api/olc", json={**ortak, "olcum": {
        "tur": "gecit", "noktalar": sahne["ipucu"]["gecit"], "ayak_izi_mm": 480,
    }}).get_json()
    gecit = yanit["gecit"]
    assert gecit["karar"]["gecer"] is True
    assert yanit["olcum"]["alt"] <= 520.0 <= yanit["olcum"]["ust"]
    assert len(gecit["profil_mm"]) == len(gecit["istasyonlar_mm"])
    assert gecit["kenar_payi_mm"] > 0

    # Alt uç ile nokta tahmini arasına düşen bir ayak izi: karar GEÇMEZ olmalı.
    arada = (yanit["olcum"]["alt"] + yanit["olcum"]["deger"]) / 2.0
    sinir = istemci.post("/api/olc", json={**ortak, "olcum": {
        "tur": "gecit", "noktalar": sahne["ipucu"]["gecit"], "ayak_izi_mm": arada,
    }}).get_json()["gecit"]["karar"]
    assert sinir["nokta_tahmini_gecer"] is True
    assert sinir["gecer"] is False


def test_uzak_olcum_uyari_uretiyor(istemci, tezgah):
    """Referanstan uzaklaşınca sistem kötüleştiğini söylemeli, sessiz kalmamalı."""
    sahne, aruco = tezgah
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": sahne["kimlik"], "referans": _ref(aruco),
        "olcum": {"tur": "mesafe", "noktalar": [[20, 20], [1380, 880]]},
        "mc_n": 100,
    }).get_json()
    assert any(u["seviye"] == "yuksek" for u in yanit["uyarilar"])


def test_elle_referans_daha_genis_aralik(istemci, tezgah):
    """Elle tıklanan köşe daha gürültülü; aralık ArUco'dan geniş çıkmalı."""
    sahne, aruco = tezgah
    ortak = {"gorsel_id": sahne["kimlik"], "mc_n": 200,
             "olcum": {"tur": "mesafe", "noktalar": sahne["ipucu"]["mesafe"]}}
    otomatik = istemci.post("/api/olc", json={**ortak, "referans": _ref(aruco)}).get_json()
    elle = istemci.post("/api/olc", json={**ortak, "referans": {
        "tur": "kare", "kenar_mm": 100, "koseler": aruco["koseler"]}}).get_json()
    assert elle["olcum"]["std"] > 2 * otomatik["olcum"]["std"]


# ------------------------------------------------------------------ hatalar
@pytest.mark.parametrize("govde, beklenen", [
    ({"gorsel_id": "yok"}, 404),
    ({"olcum": {"tur": "mesafe", "noktalar": [[1, 1]]}}, 400),
    ({"olcum": {"tur": "hacim", "noktalar": [[1, 1], [2, 2]]}}, 400),
    ({"sigma_px": -1}, 400),
    ({"referans": {"tur": "nesne", "nesne": "yok", "koseler": [[0, 0]] * 4}}, 400),
])
def test_kotu_istekler_aciklamali_hata(istemci, tezgah, govde, beklenen):
    sahne, aruco = tezgah
    tam = {"gorsel_id": sahne["kimlik"], "referans": _ref(aruco),
           "olcum": {"tur": "mesafe", "noktalar": sahne["ipucu"]["mesafe"]}, **govde}
    yanit = istemci.post("/api/olc", json=tam)
    assert yanit.status_code == beklenen
    assert yanit.get_json()["hata"]


def test_bozuk_gorsel_reddediliyor(istemci):
    yanit = istemci.post("/api/gorsel",
                         data={"dosya": (io.BytesIO(b"jpeg degil"), "x.jpg")})
    assert yanit.status_code == 400
    assert "okunamadı" in yanit.get_json()["hata"]


def test_isaretsiz_goruntude_aruco_hatasi(istemci):
    duz = cv2.imencode(".png", np.full((200, 200, 3), 128, np.uint8))[1].tobytes()
    ozet = istemci.post("/api/gorsel",
                        data={"dosya": (io.BytesIO(duz), "duz.png")}).get_json()
    yanit = istemci.post("/api/aruco", json={"gorsel_id": ozet["kimlik"],
                                             "kenar_mm": 50})
    assert yanit.status_code == 400
    assert "bulunamadı" in yanit.get_json()["hata"]


# ------------------------------------------------------------------ örnek sahne
@pytest.mark.parametrize("ad", ["tezgah", "gecit"])
def test_ornek_sahnede_isaret_bulunuyor(ad):
    """
    Örnek sahnedeki işaret gerçekten tespit edilebilmeli.

    Bu göründüğünden kırılgan: kamera düzleme üstten baktığı için dünya->görüntü
    dönüşümü yönelim çeviriyor ve işaret aynalanmış yapıştırılırsa hiçbir
    dedektör onu tanımıyor.
    """
    sahne = ornek_sahne(ad)
    kenar = sahne["referans"]["kenar_mm"]
    dunya, resim, _ = referans.aruco_bul(sahne["goruntu"], kenar)
    h = Homografi.kur(dunya, resim)

    kenarlar = [np.hypot(*(resim[i] - resim[(i + 1) % 4])) for i in range(4)]
    assert min(kenarlar) > 20, "işaret ölçüm için fazla küçük görünüyor"
    assert h.rms_px < 1e-6

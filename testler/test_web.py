"""
Web panelinin arka ucu.

Buradaki asıl soru "uç nokta 200 döndü mü" değil — panelin ürettiği sayının
kütüphanenin ürettiği sayıyla AYNI olması. Aynı çekirdeği iki ayrı yerden
çağırıyoruz; bir gün ayrışırlarsa kullanıcı hangisine güveneceğini bilemez.
"""

import io

import numpy as np
import pytest

flask = pytest.importorskip("flask")
cv2 = pytest.importorskip("cv2")

from metrik_goz import Homografi, belirsizlik, olcum, referans
from metrik_goz import ornek as ornek_modulu
from metrik_goz.ornek import ornek_sahne
from metrik_goz.web.sunucu import Depo, uygulama_kur

ELLE = referans.TIPIK_SIGMA_PX["elle"]


@pytest.fixture(scope="module")
def istemci():
    return uygulama_kur().test_client()


@pytest.fixture(scope="module")
def masa(istemci):
    """Panelin asıl senaryosu: masada telefon, yanında 1 TL. Tepeden çekim."""
    return istemci.post("/api/ornek", json={"ad": "duz"}).get_json()


@pytest.fixture(scope="module")
def gecit_sahnesi(istemci):
    """ArUco'lu sahne: projektif referans ailesini sınayan testler için."""
    sahne = istemci.post("/api/ornek", json={"ad": "gecit"}).get_json()
    aruco = istemci.post("/api/aruco", json={"gorsel_id": sahne["kimlik"],
                                             "kenar_mm": 200}).get_json()
    return sahne, aruco


def _olcek_ref(sahne):
    """Tek bilinen uzunluktan kurulan benzerlik referansı (panelin varsayılanı)."""
    return {"tur": "olcek", "ad": sahne["referans"]["ad"],
            "noktalar": sahne["ipucu"]["referans"]}


def _aruco_ref(aruco, kenar=200):
    return {"tur": "aruco", "kenar_mm": kenar, "koseler": aruco["koseler"]}


def _kur_fn(uzunluk_mm):
    return lambda g: Homografi.olcekten(g[0], g[1], uzunluk_mm)


# ------------------------------------------------------------------ temel
def test_panel_aciliyor(istemci):
    yanit = istemci.get("/")
    assert yanit.status_code == 200
    govde = yanit.get_data(as_text=True)
    assert 'id="tuval"' in govde and "panel.js" in govde


def test_panel_yalniz_surebildigi_ornekleri_gosteriyor(istemci):
    """
    Panel tek akış sürüyor (nesnenin ölçüsü); geçit sahnesi o akışa uymuyor.

    Düğmesini yine de basarsak kullanıcı hiçbir şeyin çalışmadığı bir sahneye
    düşer — sahne listesi ile panelin sürebildiği liste ayrı tutuluyor.
    """
    govde = istemci.get("/").get_data(as_text=True)
    for ad in ornek_modulu.PANEL_SAHNELERI:
        assert f'data-ornek="{ad}"' in govde
    assert 'data-ornek="gecit"' not in govde
    assert "gecit" in ornek_modulu.SAHNELER          # API'de duruyor


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
def test_panel_ile_kutuphane_ayni_sayiyi_veriyor(istemci, masa):
    """Panelin döndürdüğü değer, aynı girdiyle kütüphanenin döndürdüğüdür."""
    koseler = masa["ipucu"]["kutu"]
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": masa["kimlik"], "referans": _olcek_ref(masa),
        "olcum": {"tur": "kutu", "noktalar": koseler},
        "sigma_px": ELLE, "mc_n": 200, "guven": 0.95,
    }).get_json()

    uzunluk_mm = referans.BILINEN_UZUNLUKLAR[masa["referans"]["ad"]][0]
    resim = np.asarray(masa["ipucu"]["referans"], float)
    beklenen = belirsizlik.monte_carlo(
        None, resim, lambda h, n: olcum.kutu(h, n).en_mm,
        np.asarray(koseler, float), sigma_px=ELLE, n=200,
        kur_fn=_kur_fn(uzunluk_mm))

    assert yanit["olcum"]["ad"] == "en"
    assert yanit["olcum"]["deger"] == pytest.approx(beklenen.deger, rel=1e-12)
    assert yanit["olcum"]["alt"] == pytest.approx(beklenen.alt, rel=1e-12)


def test_olcum_gercek_degeri_yakaliyor(istemci, masa):
    """Tepeden çekilmiş sahnede doğru cevap güven aralığının içinde olmalı."""
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": masa["kimlik"], "referans": _olcek_ref(masa),
        "olcum": {"tur": "kutu", "noktalar": masa["ipucu"]["kutu"]},
        "mc_n": 400,
    }).get_json()

    olculer = {o["ad"]: o for o in yanit["olculer"]}
    assert set(olculer) == {"en", "boy", "alan"}
    for ad, o in olculer.items():
        gercek = masa["gercek"][ad]
        assert o["birim"] == gercek["birim"]
        assert o["alt"] <= gercek["deger"] <= o["ust"], \
            f"{ad}: {gercek['deger']} ∉ [{o['alt']}, {o['ust']}]"


def test_egik_cekimde_perspektif_uyarisi(istemci):
    """
    Benzerlik modelinin tek gerçek zaafı perspektif ve sistematik yanlılık hata
    payının İÇİNDE DEĞİL. Sistem bunu sessizce yutmamalı: eğik sahnede ölçüm
    gerçek değeri ıskalıyor, o yüzden yüksek seviyeli uyarı şart.
    """
    sahne = istemci.post("/api/ornek", json={"ad": "egik"}).get_json()
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": sahne["kimlik"], "referans": _olcek_ref(sahne),
        "olcum": {"tur": "kutu", "noktalar": sahne["ipucu"]["kutu"]},
        "mc_n": 200,
    }).get_json()

    en = yanit["olculer"][0]
    gercek = sahne["gercek"]["en"]["deger"]
    assert not (en["alt"] <= gercek <= en["ust"]), "eğik sahne aralığı tutmamalı"

    yuksek = [u["metin"] for u in yanit["uyarilar"] if u["seviye"] == "yuksek"]
    assert yuksek and "perspektif" in " ".join(yuksek).lower()
    assert yanit["kutu"]["dikdortgenlik"] > 0.06


def test_benzerlik_modelinde_analitik_yerine_monte_carlo(istemci, masa):
    """
    Analitik yayılım sekiz projektif parametrenin kovaryansına dayanıyor;
    benzerlik modelinde o parametrelerin dördü serbest bile değil. Sessizce
    yanlış sayı üretmek yerine yöntemi düşürüp bunu söylemeli.
    """
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": masa["kimlik"], "referans": _olcek_ref(masa),
        "olcum": {"tur": "kutu", "noktalar": masa["ipucu"]["kutu"]},
        "yontem": "analitik", "mc_n": 100,
    }).get_json()

    assert yanit["olcum"]["yontem"].startswith("monte_carlo")
    assert any("Monte Carlo" in u["metin"] for u in yanit["uyarilar"])


def test_gecit_karari_alt_uca_gore(istemci, gecit_sahnesi):
    """
    Karar nokta tahminine değil aralığın ALT ucuna bakmalı: geçemeyeceği yola
    girmek, geçebileceği yolu kaçırmaktan pahalı.
    """
    sahne, aruco = gecit_sahnesi
    ortak = {"gorsel_id": sahne["kimlik"], "referans": _aruco_ref(aruco), "mc_n": 60}

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


def test_uzak_olcum_uyari_uretiyor(istemci, gecit_sahnesi):
    """
    Projektif referanstan uzaklaşınca sistem kötüleştiğini söylemeli.

    Uzaklık riski projektif ailenin zaafı: homografi referans köşelerine
    oturtuluyor, ondan uzaklaştıkça parametre hatası büyüyerek taşınıyor.
    """
    sahne, aruco = gecit_sahnesi
    yanit = istemci.post("/api/olc", json={
        "gorsel_id": sahne["kimlik"], "referans": _aruco_ref(aruco),
        "olcum": {"tur": "mesafe", "noktalar": [[20, 20], [1380, 880]]},
        "mc_n": 100,
    }).get_json()
    assert any(u["seviye"] == "yuksek" for u in yanit["uyarilar"])


def test_elle_referans_daha_genis_aralik(istemci, gecit_sahnesi):
    """Elle tıklanan köşe daha gürültülü; aralık ArUco'dan geniş çıkmalı."""
    sahne, aruco = gecit_sahnesi
    ortak = {"gorsel_id": sahne["kimlik"], "mc_n": 200,
             "olcum": {"tur": "mesafe", "noktalar": [[400, 500], [1000, 520]]}}
    otomatik = istemci.post("/api/olc",
                            json={**ortak, "referans": _aruco_ref(aruco)}).get_json()
    elle = istemci.post("/api/olc", json={**ortak, "referans": {
        "tur": "kare", "kenar_mm": 200, "koseler": aruco["koseler"]}}).get_json()
    assert elle["olcum"]["std"] > 2 * otomatik["olcum"]["std"]


# ------------------------------------------------------------------ hatalar
@pytest.mark.parametrize("govde, beklenen", [
    ({"gorsel_id": "yok"}, 404),
    ({"olcum": {"tur": "kutu", "noktalar": [[1, 1]]}}, 400),
    ({"olcum": {"tur": "hacim", "noktalar": [[1, 1], [2, 2]]}}, 400),
    ({"sigma_px": -1}, 400),
    ({"yontem": "kehanet"}, 400),
    ({"referans": {"tur": "nesne", "nesne": "yok", "koseler": [[0, 0]] * 4}}, 400),
    ({"referans": {"tur": "olcek", "ad": "yok", "noktalar": [[0, 0], [9, 9]]}}, 400),
    ({"referans": {"tur": "olcek", "uzunluk_mm": 26.15, "noktalar": [[0, 0]]}}, 400),
])
def test_kotu_istekler_aciklamali_hata(istemci, masa, govde, beklenen):
    tam = {"gorsel_id": masa["kimlik"], "referans": _olcek_ref(masa),
           "olcum": {"tur": "kutu", "noktalar": masa["ipucu"]["kutu"]}, **govde}
    yanit = istemci.post("/api/olc", json=tam)
    assert yanit.status_code == beklenen
    assert yanit.get_json()["hata"]


def test_bilinmeyen_ornek_aciklamali_hata(istemci):
    yanit = istemci.post("/api/ornek", json={"ad": "tezgah"})
    assert yanit.status_code == 400
    assert "duz" in yanit.get_json()["hata"]


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
def test_gecit_sahnesinde_isaret_bulunuyor():
    """
    Örnek sahnedeki işaret gerçekten tespit edilebilmeli.

    Bu göründüğünden kırılgan: kamera düzleme üstten baktığı için dünya->görüntü
    dönüşümü yönelim çeviriyor ve işaret aynalanmış yapıştırılırsa hiçbir
    dedektör onu tanımıyor.
    """
    sahne = ornek_sahne("gecit")
    kenar = sahne["referans"]["kenar_mm"]
    dunya, resim, _ = referans.aruco_bul(sahne["goruntu"], kenar)
    h = Homografi.kur(dunya, resim)

    kenarlar = [np.hypot(*(resim[i] - resim[(i + 1) % 4])) for i in range(4)]
    assert min(kenarlar) > 20, "işaret ölçüm için fazla küçük görünüyor"
    assert h.rms_px < 1e-6


@pytest.mark.parametrize("ad, en_cok_hata", [("duz", 0.03), ("egik", 1.0)])
def test_ornek_ipuclari_sahnenin_iddiasini_tutuyor(ad, en_cok_hata):
    """
    İpucu noktaları gürültüsüz ve doğru cevap biliniyor; kalan hata modelin
    kendi hatasıdır. `duz` ilan edilen %3'ün altında kalmalı — `egik` ise
    kalmamalı, sahnenin bütün anlamı bu farkı göstermek.
    """
    sahne = ornek_sahne(ad)
    ref = np.asarray(sahne["ipucu"]["referans"], float)
    h = Homografi.olcekten(ref[0], ref[1], sahne["referans"]["uzunluk_mm"])
    k = olcum.kutu(h, np.asarray(sahne["ipucu"]["kutu"], float))

    gercek = sahne["gercek"]["en"]["deger"]
    hata = abs(k.en_mm - gercek) / gercek
    assert h.model == "benzerlik"
    if ad == "duz":
        assert hata < en_cok_hata
        assert k.dikdortgenlik < 0.02
    else:
        assert hata > 0.10, "eğik sahne belirgin biçimde yanılmalı"
        assert k.dikdortgenlik > 0.06, "yanılma gözlenebilir olmalı"

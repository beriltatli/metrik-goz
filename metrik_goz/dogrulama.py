"""
Sentetik doğrulama: iddia edilen sayıları üreten ve grafikleyen modül.

README'deki hiçbir sayı elle yazılmıyor; hepsi burada üretilip
`dogrulama/sonuclar.json` dosyasına yazılıyor. Böylece kod değişince sayıların
eskimesi mümkün olmuyor.

    python -m metrik_goz.cli dogrula --cikti dogrulama
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .homografi import Homografi
from . import belirsizlik, olcum, referans
from .sentetik import gurultule, sahne_kur

# Doğrulanmış kategorik palet (renk körlüğü güvenli, açık zeminde geçti)
MAVI, TURUNCU, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
YUZEY, IZGARA = "#fcfcfb", "#e6e5e1"

SIGMA_PX = 0.5
REF_MM = 100.0

# Benzerlik (tek uzunluk) modelinin taraması: panelin varsayılan senaryosu —
# masadaki 1 TL ile masadaki telefonu ölçmek.
BENZERLIK_REF_MM = 26.15                 # 1 TL, çap
BENZERLIK_NESNE_MM = (146.7, 71.5)       # telefon: en, boy
BENZERLIK_EGIMLER = (0.0, 5.0, 10.0, 20.0, 30.0)
BENZERLIK_UZAKLIKLAR = (0.0, 2.0, 4.0, 6.0)   # referans çapının katı
EN_AZ_HUCRE = 8            # kadraj dışı kalan hücreleri ortalamaya sokma
# Sunucunun "yüksek" uyarı eşiği; taramanın işi tam olarak bunu sınamak.
DIKDORTGENLIK_ESIGI = 0.06
CIDDI_HATA = 0.05


# ------------------------------------------------------------------ deneyler
def _tek_olcum(rng, *, mesafe_mm, egim, uzaklik_kati, mc_n=200):
    sahne = sahne_kur(referans_boyut_mm=REF_MM, mesafe_mm=mesafe_mm,
                      egim_derece=egim, azimut_derece=float(rng.uniform(0, 360)))
    r = uzaklik_kati * REF_MM
    aci = rng.uniform(0, 2 * np.pi)
    merkez = r * np.array([np.cos(aci), np.sin(aci)])
    yon = rng.uniform(0, 2 * np.pi)
    boy = rng.uniform(0.5, 2.0) * REF_MM
    v = np.array([np.cos(yon), np.sin(yon)])
    a, b = merkez - 0.5 * boy * v, merkez + 0.5 * boy * v

    a_px, b_px = sahne.izdusur(a)[0], sahne.izdusur(b)[0]
    if not (sahne.gorunur_mu([a_px, b_px]) and sahne.gorunur_mu(sahne.referans_px)):
        return None

    ref_g = gurultule(sahne.referans_px, SIGMA_PX, rng)
    nokta_g = gurultule(np.array([a_px, b_px]), SIGMA_PX, rng)
    gercek = float(np.hypot(*(b - a)))
    fn = lambda h, n: olcum.mesafe(h, n[0], n[1])

    mc = belirsizlik.monte_carlo(sahne.referans_dunya, ref_g, fn, nokta_g,
                                 sigma_px=SIGMA_PX, n=mc_n,
                                 tohum=int(rng.integers(1 << 30)))
    h = Homografi.kur(sahne.referans_dunya, ref_g)
    an = belirsizlik.analitik(h, sahne.referans_dunya, ref_g, fn, nokta_g, sigma_px=SIGMA_PX)
    return dict(gercek=gercek, mc=mc, an=an,
                icinde=bool(mc.alt <= gercek <= mc.ust),
                hata=abs(mc.deger - gercek) / gercek)


def _kosu(tohum, n, **kw):
    rng = np.random.default_rng(tohum)
    return [d for d in (_tek_olcum(rng, **kw) for _ in range(n)) if d]


# ------------------------------------------------------------------ grafikler
def _eksen_duzen(ax, baslik, altbaslik=None, xlabel=None, ylabel=None):
    ax.set_title(baslik, fontsize=13, fontweight="600", color=INK, loc="left",
                 pad=30 if altbaslik else 12)
    if altbaslik:
        ax.text(0, 1.045, altbaslik, transform=ax.transAxes, fontsize=10.5,
                color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(IZGARA)


def _grafik_kapsama(kosullar, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adlar = [k["ad"] for k in kosullar]
    deger = [k["kapsama"] for k in kosullar]
    y = np.arange(len(adlar))

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(adlar) + 2.1))
    fig.patch.set_facecolor(YUZEY)
    ax.set_facecolor(YUZEY)

    ax.axvline(0.95, color=TURUNCU, linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    ax.text(0.95, len(adlar) - 0.35, " nominal %95", color=TURUNCU, fontsize=10,
            fontweight="600", va="center")

    ax.hlines(y, 0.85, deger, color=IZGARA, linewidth=1.4, zorder=1)
    ax.scatter(deger, y, s=70, color=MAVI, zorder=3,
               edgecolor=YUZEY, linewidth=1.5)
    # Etiketler hizalı bir sütunda: nominal çizgiyle çakışmasınlar
    for yi, d in zip(y, deger):
        ax.text(1.0, yi, f"%{d * 100:.1f}", color=INK, fontsize=10,
                va="center", ha="right", fontweight="600")

    ax.set_yticks(y, adlar)
    ax.set_xlim(0.85, 1.012)
    ax.set_ylim(-0.6, len(adlar) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"%{v * 100:.0f}")
    ax.grid(axis="x", color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    _eksen_duzen(ax, "Güven aralığı gerçekten tutuyor mu",
                 "Her koşulda bağımsız sentetik ölçümler · gerçek değerin %95 aralığında "
                 "çıkma oranı", xlabel="kapsama")
    fig.tight_layout()
    fig.savefig(yol, dpi=170, facecolor=YUZEY)
    plt.close(fig)


def _grafik_hata_uzaklik(seri, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([s["uzaklik"] for s in seri])
    orta = np.array([s["ortanca"] for s in seri]) * 100
    p10 = np.array([s["p10"] for s in seri]) * 100
    p90 = np.array([s["p90"] for s in seri]) * 100

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(YUZEY)
    ax.set_facecolor(YUZEY)

    ax.fill_between(x, p10, p90, color=MAVI, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(x, orta, color=MAVI, linewidth=2.0, marker="o", markersize=7,
            markeredgecolor=YUZEY, markeredgewidth=1.5, zorder=3)

    ax.axhline(3.0, color=TURUNCU, linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    ax.text(x[0], 3.0, " ilan edilen sınır %3", color=TURUNCU, fontsize=10,
            fontweight="600", va="bottom")
    ax.text(x[-1], orta[-1], f"  ortanca %{orta[-1]:.1f}", color=INK, fontsize=10,
            fontweight="600", va="center")
    ax.text(x[-1], p90[-1], f"  p90 %{p90[-1]:.1f}", color=INK2, fontsize=9.5, va="center")

    ax.set_xlim(x[0] - 0.15, x[-1] + 0.9)
    ax.set_ylim(0, max(p90.max() * 1.15, 4))
    ax.grid(axis="y", color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    _eksen_duzen(ax, "Hata, referanstan uzaklaştıkça büyüyor",
                 "1,2 m mesafe · 25° bakış · 100 mm referans · 0,5 px köşe gürültüsü",
                 xlabel="ölçülen yerin referansa uzaklığı (referans boyutunun katı)",
                 ylabel="bağıl hata (%)")
    fig.tight_layout()
    fig.savefig(yol, dpi=170, facecolor=YUZEY)
    plt.close(fig)


def _grafik_analitik_mc(denemeler, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mc = np.array([d["mc"].std for d in denemeler])
    an = np.array([d["an"].std for d in denemeler])
    ust = max(mc.max(), an.max()) * 1.08

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor(YUZEY)
    ax.set_facecolor(YUZEY)

    ax.plot([0, ust], [0, ust], color=INK3, linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
    ax.text(ust * 0.97, ust * 0.97, "eşit  ", color=INK3, fontsize=10,
            ha="right", va="top", rotation=45, rotation_mode="anchor")
    ax.scatter(mc, an, s=46, color=MAVI, alpha=0.65, zorder=3,
               edgecolor=YUZEY, linewidth=1.0)

    oran = np.median(an / np.maximum(mc, 1e-9))
    ax.text(0.04, 0.94, f"ortanca oran {oran:.2f}", transform=ax.transAxes,
            fontsize=11, color=INK, fontweight="600")

    ax.set_xlim(0, ust)
    ax.set_ylim(0, ust)
    ax.grid(color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    _eksen_duzen(ax, "Hızlı yol yavaş yolla aynı sonucu veriyor",
                 "Her nokta bir ölçüm · analitik yayılım vs Monte Carlo",
                 xlabel="Monte Carlo std (mm)", ylabel="analitik std (mm)")
    fig.tight_layout()
    fig.savefig(yol, dpi=170, facecolor=YUZEY)
    plt.close(fig)


# --------------------------------------------------- benzerlik (tek uzunluk) modeli
def _en_genis_cap(sahne, merkez_mm, yaricap_mm: float) -> np.ndarray:
    """
    Dairenin görüntüdeki EN GENİŞ yerinin iki ucu.

    Yuvarlak referansın izdüşümü perspektifte elips; hangi çapı okuduğun fark
    eder. En geniş yer (elipsin büyük ekseni) kısalmaya uğramamış olan çaptır —
    kullanıcı da doğal olarak oradan tıklar. Buradaki tarama tam olarak o iyi
    niyetli kullanıcıyı taklit ediyor; kötü bir çap seçerek modeli haksız yere
    kötü göstermek doğrulamayı işe yaramaz kılardı.
    """
    aci = np.linspace(0, np.pi, 360, endpoint=False)
    yon = np.column_stack([np.cos(aci), np.sin(aci)])
    a = sahne.izdusur(np.asarray(merkez_mm, float) + yaricap_mm * yon)
    b = sahne.izdusur(np.asarray(merkez_mm, float) - yaricap_mm * yon)
    k = int(np.argmax(np.hypot(*(a - b).T)))
    return np.array([a[k], b[k]])


def _benzerlik_tek(rng, *, egim, uzaklik_kati, mc_n=100):
    """
    Bir deneme: masada bir para, ondan `uzaklik_kati` çap uzakta bir telefon.

    İki sayı birden ölçülüyor ve karıştırılmamaları şart:

    yanlilik — GÜRÜLTÜSÜZ ölçümün hatası, yani modelin kendi hatası. Tek
               uzunluktan kurulan ölçek perspektifi düzeltmediği için eğimle
               büyüyen bu terim sistematiktir; hata payı onu KAPSAMIYOR.
    hata     — kullanıcının gerçekten göreceği hata: yanlılık + tıklama
               gürültüsü. 26 mm'lik para görüntüde ~75 px olduğundan tepeden
               çekimde bile birkaç yüzdelik gürültü payı kalıyor.

    İkisini tek sayıda toplarsak "tepeden çekimde model kusursuz" iddiası
    gürültünün altında kaybolur ve tarama hiçbir şey kanıtlamaz.
    """
    sahne = sahne_kur(referans_boyut_mm=BENZERLIK_REF_MM, odak_px=1500.0,
                      boyut_px=(900, 1300), mesafe_mm=520.0, egim_derece=egim,
                      azimut_derece=float(rng.uniform(0, 360)))
    en_mm, boy_mm = BENZERLIK_NESNE_MM

    aci = rng.uniform(0, 2 * np.pi)
    merkez = uzaklik_kati * BENZERLIK_REF_MM * np.array([np.cos(aci), np.sin(aci)])
    t = rng.uniform(0, 2 * np.pi)
    donme = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    kose_mm = np.array([[-en_mm / 2, -boy_mm / 2], [en_mm / 2, -boy_mm / 2],
                        [en_mm / 2, boy_mm / 2], [-en_mm / 2, boy_mm / 2]]) @ donme.T + merkez

    kose_px = sahne.izdusur(kose_mm)
    ref_px = _en_genis_cap(sahne, (0.0, 0.0), BENZERLIK_REF_MM / 2.0)
    if not (sahne.gorunur_mu(kose_px) and sahne.gorunur_mu(ref_px)):
        return None

    kur_fn = lambda g: Homografi.olcekten(g[0], g[1], BENZERLIK_REF_MM)
    yanlilik = abs(olcum.kutu(kur_fn(ref_px), kose_px).en_mm - en_mm) / en_mm

    # Para da telefon da elle tıklanıyor; ArUco'nun alt piksel doğruluğu yok.
    sigma = referans.TIPIK_SIGMA_PX["elle"]
    ref_g = gurultule(ref_px, sigma, rng)
    kose_g = gurultule(kose_px, sigma, rng)
    kutu = olcum.kutu(kur_fn(ref_g), kose_g)
    mc = belirsizlik.monte_carlo(
        None, ref_g, lambda h, n: olcum.kutu(h, n).en_mm, kose_g,
        sigma_px=sigma, n=mc_n, tohum=int(rng.integers(1 << 30)), kur_fn=kur_fn)

    return dict(gercek=en_mm, mc=mc,
                yanlilik=float(yanlilik),
                hata=abs(mc.deger - en_mm) / en_mm,
                icinde=bool(mc.alt <= en_mm <= mc.ust),
                # Kullanıcının ekranında görünen sapma: gürültülü köşelerden.
                dikdortgenlik=float(kutu.dikdortgenlik))


def benzerlik_taramasi(n: int = 40, *, tohum: int = 900) -> dict:
    """
    Tek bilinen uzunluktan kurulan BENZERLİK modelinin nerede geçerli olduğu.

    Bu model perspektifi düzeltmiyor — panelin uyarı metinleri bu taramanın
    ölçtüğü üç şeye dayanıyor:

    1) Yanlılığı üreten şey uzaklık değil, EĞİM. Tam tepeden çekimde ölçek
       düzlemin her yerinde aynı olduğu için para nerede durursa dursun
       yanlılık sıfıra iniyor; bakış yatıklaştıkça büyüyor ve uzaklık onu
       çarpan olarak büyütüyor. Projektif modelde durum tersine: orada
       uzaklığın kendisi risk.
    2) Sistematik yanlılık hata payının İÇİNDE DEĞİL. Bu yüzden eğim büyürken
       kapsama çöküyor; aralığın genişliği doğru, merkezi kayıyor. Gizlemek
       yerine ölçüp yazıyoruz — panelin "hata payı bunu kapsamıyor" cümlesi
       tam olarak buradan geliyor.
    3) Kullanıcı eğimi bilmiyor, ama ölçtüğü nesnenin karşılıklı kenarlarının
       ne kadar farklı çıktığını sistem görüyor. `Kutu.dikdortgenlik` bu yüzden
       eğimin gözlenebilir vekili; taramanın asıl sınavı, ciddi yanlılığın ne
       kadarını bu vekilin eşiği yakaladığı.
    """
    rng = np.random.default_rng(tohum)
    hucreler, denemeler = [], []
    for egim in BENZERLIK_EGIMLER:
        for uzaklik in BENZERLIK_UZAKLIKLAR:
            d = [x for x in (_benzerlik_tek(rng, egim=egim, uzaklik_kati=uzaklik)
                             for _ in range(n)) if x]
            # Nesne kadrajdan taştığında elde birkaç deneme kalıyor;
            # üç örnekten çıkan bir ortanca sayı gibi görünür, bilgi değil.
            if len(d) < EN_AZ_HUCRE:
                continue
            denemeler.extend(d)
            hatalar = np.array([x["hata"] for x in d])
            hucreler.append(dict(
                egim=egim, uzaklik=uzaklik, n=len(d),
                ortanca_yanlilik=float(np.median([x["yanlilik"] for x in d])),
                ortanca_hata=float(np.median(hatalar)),
                p90_hata=float(np.percentile(hatalar, 90)),
                ortanca_dikdortgenlik=float(np.median([x["dikdortgenlik"] for x in d])),
                kapsama=float(np.mean([x["icinde"] for x in d])),
            ))

    yanlilik = np.array([d["yanlilik"] for d in denemeler])
    sapma = np.array([d["dikdortgenlik"] for d in denemeler])
    ciddi = yanlilik > CIDDI_HATA
    uyarilan = sapma > DIKDORTGENLIK_ESIGI

    tepeden = [h for h in hucreler if h["egim"] == 0.0]
    yatik = [h for h in hucreler if h["egim"] == BENZERLIK_EGIMLER[-1]]
    return dict(
        hucreler=hucreler,
        n=len(denemeler),
        esik=DIKDORTGENLIK_ESIGI,
        ciddi_hata_esigi=CIDDI_HATA,
        # Ciddi yanlılığın kaçı uyarı üretiyor: uyarının asıl işi bu.
        yakalama=float(np.mean(uyarilan[ciddi])) if ciddi.any() else float("nan"),
        yanlis_alarm=float(np.mean(uyarilan[~ciddi])) if (~ciddi).any() else float("nan"),
        # Tepeden çekimde yanlılık uzaklıktan bağımsız olarak sıfıra inmeli.
        tepeden_p90_yanlilik=float(max(h["ortanca_yanlilik"] for h in tepeden)),
        tepeden_kapsama=float(np.mean([h["kapsama"] for h in tepeden])),
        yatik_ortanca_yanlilik=float(max(h["ortanca_yanlilik"] for h in yatik)),
        yatik_kapsama=float(np.mean([h["kapsama"] for h in yatik])),
        en_kotu_ortanca_yanlilik=float(max(h["ortanca_yanlilik"] for h in hucreler)),
    )


def _grafik_benzerlik(tarama, yol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.0),
                                 gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.patch.set_facecolor(YUZEY)
    renkler = [MAVI, AQUA, TURUNCU, INK3]

    for renk, uzaklik in zip(renkler, BENZERLIK_UZAKLIKLAR):
        seri = [h for h in tarama["hucreler"] if h["uzaklik"] == uzaklik]
        if not seri:
            continue
        x = [h["egim"] for h in seri]
        y = [h["ortanca_yanlilik"] * 100 for h in seri]
        ax.plot(x, y, color=renk, linewidth=2.0, marker="o", markersize=6,
                markeredgecolor=YUZEY, markeredgewidth=1.4, zorder=3)
        ax.text(x[-1], y[-1], f"  {uzaklik:g}× uzakta", color=renk, fontsize=9.5,
                va="center", fontweight="600")

    ax.set_facecolor(YUZEY)
    ax.axhline(3.0, color=INK3, linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
    ax.text(0, 3.0, " ilan edilen sınır %3", color=INK3, fontsize=9.5,
            va="bottom", fontweight="600")
    ax.set_xlim(-1.5, BENZERLIK_EGIMLER[-1] + 9)
    ax.grid(axis="y", color=IZGARA, linewidth=0.8)
    ax.set_axisbelow(True)
    _eksen_duzen(ax, "Yanlılığı üreten şey eğim, uzaklık değil",
                 "Tek uzunluktan ölçek · 1 TL çapıyla telefon · gürültüsüz, yani "
                 "modelin kendi hatası",
                 xlabel="kamera eğimi (derece, 0 = tam tepeden)",
                 ylabel="ortanca sistematik yanlılık (%)")

    sapma = [h["ortanca_dikdortgenlik"] * 100 for h in tarama["hucreler"]]
    hata = [h["ortanca_yanlilik"] * 100 for h in tarama["hucreler"]]
    bx.set_facecolor(YUZEY)
    bx.axvline(tarama["esik"] * 100, color=TURUNCU, linewidth=1.6,
               linestyle=(0, (5, 3)), zorder=1)
    bx.text(tarama["esik"] * 100, 0.0, " uyarı eşiği", color=TURUNCU,
            fontsize=9.5, va="bottom", fontweight="600")
    bx.scatter(sapma, hata, s=54, color=MAVI, alpha=0.8, zorder=3,
               edgecolor=YUZEY, linewidth=1.2)
    bx.text(0.04, 0.93, f"ciddi yanlılığın %{tarama['yakalama'] * 100:.0f}'i "
                        f"uyarı üretiyor", transform=bx.transAxes, fontsize=10.5,
            color=INK, fontweight="600")
    bx.grid(color=IZGARA, linewidth=0.8)
    bx.set_axisbelow(True)
    _eksen_duzen(bx, "Eğimin gözlenebilir vekili",
                 "Karşılıklı kenar uyumsuzluğu, eğimi bilmeden eğimi ele veriyor",
                 xlabel="dikdörtgenlik sapması (%)", ylabel="ortanca yanlılık (%)")

    fig.tight_layout(w_pad=3.0)
    fig.savefig(yol, dpi=170, facecolor=YUZEY)
    plt.close(fig)


# ------------------------------------------------------------------ ana akış
def dogrulamayi_calistir(cikti_dizini: str = "dogrulama", n: int = 140) -> dict:
    yol = pathlib.Path(cikti_dizini)
    yol.mkdir(parents=True, exist_ok=True)

    kosullar = []
    tanimlar = [
        ("mesafe 0,6 m", dict(mesafe_mm=600, egim=25.0, uzaklik_kati=1.0)),
        ("mesafe 1,2 m", dict(mesafe_mm=1200, egim=25.0, uzaklik_kati=1.0)),
        ("mesafe 2,0 m", dict(mesafe_mm=2000, egim=25.0, uzaklik_kati=1.0)),
        ("mesafe 3,0 m", dict(mesafe_mm=3000, egim=25.0, uzaklik_kati=1.0)),
        ("bakış 0° (tepeden)", dict(mesafe_mm=1200, egim=0.0, uzaklik_kati=1.0)),
        ("bakış 40°", dict(mesafe_mm=1200, egim=40.0, uzaklik_kati=1.0)),
        ("bakış 55° (çok yatık)", dict(mesafe_mm=1200, egim=55.0, uzaklik_kati=1.0)),
        ("uzaklık 4× referans", dict(mesafe_mm=1200, egim=25.0, uzaklik_kati=4.0)),
    ]
    for i, (ad, kw) in enumerate(tanimlar):
        d = _kosu(200 + i, n, **kw)
        hatalar = np.array([x["hata"] for x in d])
        kosullar.append(dict(
            ad=ad, n=len(d),
            kapsama=float(np.mean([x["icinde"] for x in d])),
            ortanca_hata=float(np.median(hatalar)),
            p90_hata=float(np.percentile(hatalar, 90)),
        ))

    uzaklik_serisi = []
    for i, k in enumerate((0.5, 1.0, 2.0, 3.0, 4.0)):
        d = _kosu(300 + i, n, mesafe_mm=1200, egim=25.0, uzaklik_kati=k)
        h = np.array([x["hata"] for x in d])
        uzaklik_serisi.append(dict(uzaklik=k, n=len(d),
                                   ortanca=float(np.median(h)),
                                   p10=float(np.percentile(h, 10)),
                                   p90=float(np.percentile(h, 90))))

    karsilastirma = _kosu(400, 90, mesafe_mm=1200, egim=25.0, uzaklik_kati=1.0)
    benzerlik = benzerlik_taramasi(max(20, n // 3))

    _grafik_kapsama(kosullar, yol / "kapsama.png")
    _grafik_hata_uzaklik(uzaklik_serisi, yol / "hata_uzaklik.png")
    _grafik_analitik_mc(karsilastirma, yol / "analitik_vs_mc.png")
    _grafik_benzerlik(benzerlik, yol / "benzerlik.png")

    oranlar = np.array([d["an"].std / max(d["mc"].std, 1e-9) for d in karsilastirma])
    ozet = dict(
        sigma_px=SIGMA_PX,
        referans_mm=REF_MM,
        kosullar=kosullar,
        uzaklik_serisi=uzaklik_serisi,
        analitik_mc_oran_ortanca=float(np.median(oranlar)),
        ortalama_kapsama=float(np.mean([k["kapsama"] for k in kosullar])),
        benzerlik=benzerlik,
    )
    (yol / "sonuclar.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"Üretildi: {yol}/kapsama.png, hata_uzaklik.png, analitik_vs_mc.png, "
          f"benzerlik.png, sonuclar.json")
    print(f"  ortalama kapsama    : %{ozet['ortalama_kapsama'] * 100:.1f}")
    print(f"  analitik/MC std oranı: {ozet['analitik_mc_oran_ortanca']:.2f}")
    for k in kosullar:
        print(f"  {k['ad']:22s} kapsama %{k['kapsama']*100:5.1f}  "
              f"ortanca hata %{k['ortanca_hata']*100:5.2f}")
    print("  benzerlik modeli (tek uzunluk, perspektif düzeltilmiyor):")
    print(f"    tepeden çekimde yanlılık : %{benzerlik['tepeden_p90_yanlilik'] * 100:.3f} "
          f"(uzaklıktan bağımsız), kapsama %{benzerlik['tepeden_kapsama'] * 100:.1f}")
    print(f"    {BENZERLIK_EGIMLER[-1]:.0f}° eğimde yanlılık   : "
          f"%{benzerlik['yatik_ortanca_yanlilik'] * 100:.1f}, "
          f"kapsama %{benzerlik['yatik_kapsama'] * 100:.1f} — sistematik yanlılık "
          f"hata payının içinde değil")
    print(f"    dikdörtgenlik uyarısı    : ciddi yanlılığın "
          f"%{benzerlik['yakalama'] * 100:.0f}'ini yakalıyor, "
          f"%{benzerlik['yanlis_alarm'] * 100:.0f} yanlış alarm")
    return ozet

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
from . import belirsizlik, olcum
from .sentetik import gurultule, sahne_kur

# Doğrulanmış kategorik palet (renk körlüğü güvenli, açık zeminde geçti)
MAVI, TURUNCU, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
YUZEY, IZGARA = "#fcfcfb", "#e6e5e1"

SIGMA_PX = 0.5
REF_MM = 100.0


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

    _grafik_kapsama(kosullar, yol / "kapsama.png")
    _grafik_hata_uzaklik(uzaklik_serisi, yol / "hata_uzaklik.png")
    _grafik_analitik_mc(karsilastirma, yol / "analitik_vs_mc.png")

    oranlar = np.array([d["an"].std / max(d["mc"].std, 1e-9) for d in karsilastirma])
    ozet = dict(
        sigma_px=SIGMA_PX,
        referans_mm=REF_MM,
        kosullar=kosullar,
        uzaklik_serisi=uzaklik_serisi,
        analitik_mc_oran_ortanca=float(np.median(oranlar)),
        ortalama_kapsama=float(np.mean([k["kapsama"] for k in kosullar])),
    )
    (yol / "sonuclar.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    print(f"Üretildi: {yol}/kapsama.png, hata_uzaklik.png, analitik_vs_mc.png, sonuclar.json")
    print(f"  ortalama kapsama    : %{ozet['ortalama_kapsama'] * 100:.1f}")
    print(f"  analitik/MC std oranı: {ozet['analitik_mc_oran_ortanca']:.2f}")
    for k in kosullar:
        print(f"  {k['ad']:22s} kapsama %{k['kapsama']*100:5.1f}  "
              f"ortanca hata %{k['ortanca_hata']*100:5.2f}")
    return ozet

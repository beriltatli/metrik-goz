"""
Komut satırı arayüzü.

    metrik-goz kutu   foto.jpg --olcek-ad 1_tl --uc 300,410 --uc 372,410 \
                               --kose ... (nesnenin 4 köşesi)
    metrik-goz mesafe foto.jpg --aruco 100 --nokta 120,340 --nokta 610,355
    metrik-goz alan   foto.jpg --aruco 100 --nokta ... (en az 3 nokta)
    metrik-goz gecit  foto.jpg --aruco 100 --maske serbest.png --ayak-izi 480
    metrik-goz dogrula --cikti dogrulama/
    metrik-goz panel  --port 8000
    metrik-goz ornek  --sahne duz --cikti ornekler/

Referans iki aileden biri olabilir ve fark ölçümün ne kadarına güvenebileceğini
belirliyor: `--olcek` tek bir bilinen UZUNLUK'tan benzerlik modeli kurar
(perspektif düzeltilmez), `--aruco` / `--nesne` / `--kose` dört noktadan
projektif model kurar (perspektif düzeltilir). Panel de aynı ayrımı yapıyor.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, NamedTuple

import numpy as np

from .homografi import Homografi
from . import belirsizlik, olcum, ornek, referans


def _nokta(metin: str) -> tuple[float, float]:
    try:
        x, y = metin.split(",")
        return float(x), float(y)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Nokta 'x,y' biçiminde olmalı: '{metin}'")


class Referans(NamedTuple):
    """
    Çözülmüş referans: hangi noktalar gözlem, model nasıl kuruluyor.

    `kur_fn` verildiğinde homografiyi model kendisi kuruyor. Benzerlik modelinde
    gözlem iki noktadan ibaret ve kalan köşeler onlardan türetiliyor; o türetilmiş
    köşeleri Monte Carlo'da bağımsız bozmak olmayan bir gürültü uydurmak olurdu.
    """

    dunya: np.ndarray | None
    resim: np.ndarray
    sigma: float
    ad: str
    kur_fn: Callable[[np.ndarray], Homografi] | None = None

    def kur(self) -> Homografi:
        if self.kur_fn is not None:
            return self.kur_fn(self.resim)
        return Homografi.kur(self.dunya, self.resim)


def _referansi_coz(args, goruntu) -> Referans:
    if args.aruco is not None:
        dunya, resim, kimlik = referans.aruco_bul(goruntu, args.aruco)
        return Referans(dunya, resim, referans.TIPIK_SIGMA_PX["aruco"], f"ArUco #{kimlik}")

    olcek_mm = args.olcek
    if args.olcek_ad is not None:
        olcek_mm, aciklama = referans.BILINEN_UZUNLUKLAR[args.olcek_ad]
    else:
        aciklama = f"{olcek_mm:g} mm bilinen uzunluk" if olcek_mm else ""
    if olcek_mm is not None:
        if len(args.uc) != 2:
            raise SystemExit("Ölçek referansı için tam 2 --uc gerekiyor "
                             "(bilinen uzunluğun iki ucu).")
        resim = np.array(args.uc, float)
        return Referans(None, resim, referans.TIPIK_SIGMA_PX["elle"], aciklama,
                        kur_fn=lambda g, u=olcek_mm: Homografi.olcekten(g[0], g[1], u))

    if args.nesne is not None:
        if len(args.kose) != 4:
            raise SystemExit("Bilinen nesne için tam 4 --kose vermelisin.")
        dunya = referans.bilinen_nesne(args.nesne)
        return Referans(dunya, np.array(args.kose, float),
                        referans.TIPIK_SIGMA_PX["elle"], args.nesne)

    raise SystemExit("Referans gerekli: --aruco KENAR_MM, --olcek-ad AD --uc x,y ×2, "
                     "--olcek UZUNLUK_MM --uc x,y ×2 ya da --nesne AD --kose x,y ×4")


def _goruntu_oku(yol: str):
    import cv2
    g = cv2.imread(yol)
    if g is None:
        raise SystemExit(f"Görüntü okunamadı: {yol}")
    return g


def _olc(ref: Referans, h: Homografi, fn, noktalar, args, *, birim: str = "mm"):
    """Belirsizliği referans modelinin kendi kurulumuyla yayar."""
    return belirsizlik.monte_carlo(
        ref.dunya, ref.resim, fn, noktalar,
        sigma_px=ref.sigma, n=args.mc, birim=birim, kur_fn=ref.kur_fn,
    )


def _uyari_bas(h: Homografi, noktalar, *, dikdortgenlik: float | None = None) -> None:
    if h.model == "benzerlik":
        # Benzerlik modelinin tek gerçek zaafı perspektif; uzaklık değil.
        if dikdortgenlik is not None and dikdortgenlik > 0.06:
            print(f"  UYARI: karşılıklı kenarlar %{dikdortgenlik * 100:.1f} farklı "
                  f"ölçülüyor — fotoğraf eğik. Tek uzunluktan kurulan ölçek "
                  f"perspektifi DÜZELTMEZ ve hata payı bunu kapsamıyor.", file=sys.stderr)
        else:
            print("  Not: tek uzunluktan ölçek kuruldu, perspektif düzeltilmiyor. "
                  "Fotoğraf nesnenin tam üstünden çekildiyse sonuç doğru.", file=sys.stderr)
    elif noktalar is not None:
        en_uzak = max(h.duzlem_disi_uyarisi(n) for n in noktalar)
        if en_uzak > 2.0:
            print(f"  UYARI: ölçüm referansın {en_uzak:.1f} kutu boyu dışında. "
                  f"Referansı ölçtüğün şeye yaklaştır.", file=sys.stderr)
    if h.rms_px > 1.5:
        print(f"  UYARI: yeniden izdüşüm hatası {h.rms_px:.2f} px — "
              f"referans köşeleri iyi oturmamış olabilir.", file=sys.stderr)


def _baslik(ref: Referans, h: Homografi) -> None:
    print(f"Referans: {ref.ad}   model: {h.model}   "
          f"yeniden izdüşüm RMS: {h.rms_px:.2f} px")


def komut_kutu(args) -> None:
    """Dört köşesi işaretlenen nesnenin eni, boyu ve alanı — panelin akışı."""
    goruntu = _goruntu_oku(args.goruntu)
    ref = _referansi_coz(args, goruntu)
    if len(args.kose_nesne) != 4:
        raise SystemExit("kutu için tam 4 --kose-nesne gerekiyor.")

    h = ref.kur()
    noktalar = np.array(args.kose_nesne, float)
    k = olcum.kutu(h, noktalar)

    # Üçü de aynı tohumdan: "en" ve "boy" birbiriyle tutarlı bir dünyada ölçülüyor.
    en = _olc(ref, h, lambda hh, nn: olcum.kutu(hh, nn).en_mm, noktalar, args)
    boy = _olc(ref, h, lambda hh, nn: olcum.kutu(hh, nn).boy_mm, noktalar, args)
    alan_ = _olc(ref, h, lambda hh, nn: olcum.kutu(hh, nn).alan_mm2 / 100.0, noktalar,
                 args, birim="cm²")

    _baslik(ref, h)
    print(f"En:       {en}")
    print(f"Boy:      {boy}")
    print(f"Alan:     {alan_}")
    _uyari_bas(h, noktalar, dikdortgenlik=k.dikdortgenlik)


def komut_mesafe(args) -> None:
    goruntu = _goruntu_oku(args.goruntu)
    ref = _referansi_coz(args, goruntu)
    if len(args.nokta) != 2:
        raise SystemExit("mesafe için tam 2 --nokta gerekiyor.")

    h = ref.kur()
    noktalar = np.array(args.nokta, float)
    sonuc = _olc(ref, h, lambda hh, nn: olcum.mesafe(hh, nn[0], nn[1]), noktalar, args)

    _baslik(ref, h)
    print(f"Mesafe:   {sonuc}")
    _uyari_bas(h, noktalar)


def komut_alan(args) -> None:
    goruntu = _goruntu_oku(args.goruntu)
    ref = _referansi_coz(args, goruntu)
    if len(args.nokta) < 3:
        raise SystemExit("alan için en az 3 --nokta gerekiyor.")

    h = ref.kur()
    noktalar = np.array(args.nokta, float)
    sonuc = _olc(ref, h, lambda hh, nn: olcum.alan(hh, nn) / 100.0,   # mm^2 -> cm^2
                 noktalar, args, birim="cm²")

    _baslik(ref, h)
    print(f"Alan:     {sonuc}")
    _uyari_bas(h, noktalar)


def komut_gecit(args) -> None:
    import cv2
    goruntu = _goruntu_oku(args.goruntu)
    ref = _referansi_coz(args, goruntu)

    maske_img = cv2.imread(args.maske, cv2.IMREAD_GRAYSCALE)
    if maske_img is None:
        raise SystemExit(f"Maske okunamadı: {args.maske}")
    maske = maske_img > 127

    h = ref.kur()
    gecit = olcum.en_dar_gecit(h, maske, adim_mm=args.adim, ornek_mm=args.ornek)

    # Eksen Monte Carlo boyunca sabit: her örnekte yeniden PCA yapmak ölçülen
    # şeyi değiştirir, belirsizliği değil.
    sonuc = belirsizlik.monte_carlo(
        ref.dunya, ref.resim,
        lambda hh, _: olcum.en_dar_gecit(hh, maske, eksen=gecit.eksen,
                                         adim_mm=args.adim,
                                         ornek_mm=args.ornek).genislik_mm,
        None, sigma_px=ref.sigma, n=min(args.mc, 60), kur_fn=ref.kur_fn,
    )
    _baslik(ref, h)
    print(f"En dar geçit: {sonuc}")
    if args.ayak_izi is not None:
        karar = "GEÇER" if sonuc.alt >= args.ayak_izi else "GEÇMEZ"
        print(f"{args.ayak_izi:.0f} mm ayak izi için karar: {karar}")
        print("  (karar aralığın ALT ucuna göre veriliyor — geçemeyeceği yola "
              "girmek, geçebileceği yolu kaçırmaktan pahalı)")


def komut_dogrula(args) -> None:
    from .dogrulama import dogrulamayi_calistir
    dogrulamayi_calistir(args.cikti)


def komut_panel(args) -> None:
    try:
        from .web import uygulama_kur
    except ImportError as hata:
        raise SystemExit("Panel için Flask gerekiyor:  pip install 'metrik-goz[web]'") from hata
    uygulama = uygulama_kur()
    print(f"metrik-goz paneli:  http://{args.host}:{args.port}")
    uygulama.run(host=args.host, port=args.port, debug=args.hata_ayikla, threaded=True)


def komut_ornek(args) -> None:
    """Sentetik örnek sahneyi diske yazar — doğru cevabı yanına JSON olarak."""
    import json
    import pathlib

    import cv2

    cikti = pathlib.Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    adlar = sorted(ornek.SAHNELER) if args.sahne == "hepsi" else [args.sahne]

    for ad in adlar:
        sahne = ornek.ornek_sahne(ad)
        goruntu_yolu = cikti / sahne["ad"]
        cv2.imwrite(str(goruntu_yolu), sahne["goruntu"])

        yan = {k: v for k, v in sahne.items() if k != "goruntu"}
        yan["ipucu"] = {a: np.asarray(b).round(2).tolist()
                        for a, b in sahne["ipucu"].items()}
        veri_yolu = goruntu_yolu.with_suffix(".json")
        veri_yolu.write_text(json.dumps(yan, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Üretildi: {goruntu_yolu}  ve  {veri_yolu}")
        for olcu_ad, g in sahne["gercek"].items():
            print(f"  gerçek {olcu_ad:8s}: {g['deger']:.1f} {g['birim']}  ({g['aciklama']})")


def ana(argv=None) -> None:
    ayristirici = argparse.ArgumentParser(
        prog="metrik-goz",
        description="Tek fotoğraftan gerçek ölçü, dürüst hata payıyla.",
    )
    alt = ayristirici.add_subparsers(dest="komut", required=True)

    def ortak(p):
        p.add_argument("goruntu")
        p.add_argument("--aruco", type=float, metavar="KENAR_MM",
                       help="ArUco işaretinin kenar uzunluğu (mm)")
        p.add_argument("--olcek", type=float, metavar="UZUNLUK_MM",
                       help="Bilinen tek uzunluk (mm); iki ucu --uc ile verilir")
        p.add_argument("--olcek-ad", choices=sorted(referans.BILINEN_UZUNLUKLAR),
                       help="Tablodaki bilinen uzunluk (örn. 1_tl)")
        p.add_argument("--uc", type=_nokta, action="append", default=[],
                       metavar="X,Y", help="Bilinen uzunluğun ucu (2 kez verilir)")
        p.add_argument("--nesne", choices=sorted(referans.BILINEN_NESNELER),
                       help="Standart boyutlu referans nesne")
        p.add_argument("--kose", type=_nokta, action="append", default=[],
                       help="Bilinen nesnenin köşesi (4 kez verilir)")
        p.add_argument("--mc", type=int, default=400, help="Monte Carlo örnek sayısı")

    p_kutu = alt.add_parser("kutu", help="Dört köşesi işaretlenen nesnenin en/boy/alanı")
    ortak(p_kutu)
    p_kutu.add_argument("--kose-nesne", type=_nokta, action="append", default=[],
                        metavar="X,Y", help="Ölçülen nesnenin köşesi (4 kez verilir)")
    p_kutu.set_defaults(fn=komut_kutu)

    p_mesafe = alt.add_parser("mesafe", help="İki nokta arası mesafe")
    ortak(p_mesafe)
    p_mesafe.add_argument("--nokta", type=_nokta, action="append", default=[])
    p_mesafe.set_defaults(fn=komut_mesafe)

    p_alan = alt.add_parser("alan", help="Çokgenin alanı")
    ortak(p_alan)
    p_alan.add_argument("--nokta", type=_nokta, action="append", default=[])
    p_alan.set_defaults(fn=komut_alan)

    p_gecit = alt.add_parser("gecit", help="Serbest alandaki en dar geçit")
    ortak(p_gecit)
    p_gecit.add_argument("--maske", required=True, help="Serbest alan maskesi (beyaz = geçilebilir)")
    p_gecit.add_argument("--ayak-izi", type=float, help="Aracın genişliği (mm)")
    p_gecit.add_argument("--adim", type=float, default=20.0)
    p_gecit.add_argument("--ornek", type=float, default=5.0)
    p_gecit.set_defaults(fn=komut_gecit)

    p_dog = alt.add_parser("dogrula", help="Sentetik doğrulamayı çalıştır ve grafikleri üret")
    p_dog.add_argument("--cikti", default="dogrulama")
    p_dog.set_defaults(fn=komut_dogrula)

    p_panel = alt.add_parser("panel", help="Web panelini başlat (sürükle-bırak arayüz)")
    p_panel.add_argument("--host", default="127.0.0.1")
    p_panel.add_argument("--port", type=int, default=8000)
    p_panel.add_argument("--hata-ayikla", action="store_true",
                         help="Flask hata ayıklama kipi (yalnız geliştirme)")
    p_panel.set_defaults(fn=komut_panel)

    p_ornek = alt.add_parser("ornek", help="Doğru cevabı bilinen sentetik örnek sahne üret")
    p_ornek.add_argument("--sahne", default="duz",
                         choices=sorted(ornek.SAHNELER) + ["hepsi"])
    p_ornek.add_argument("--cikti", default="ornekler")
    p_ornek.set_defaults(fn=komut_ornek)

    args = ayristirici.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    ana()

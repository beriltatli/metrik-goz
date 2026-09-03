"""
Örnek sahne üreteci — panelin "elimde fotoğraf yok" hâli için.

Panele bir fotoğraf sürüklemeden de sistemin ne yaptığı görülebilmeli. Burada
üretilen görüntüler sentetik: kamerayı, referansı ve ölçülecek şeyi biz
koyuyoruz, dolayısıyla DOĞRU CEVABI biliyoruz. Panel bu doğru cevabı ölçümün
yanına yazıyor — güven aralığının gerçekten tuttuğunu tek bakışta görüyorsun.

Gerçek fotoğrafta bu mümkün değil; orada yalnız ölçüm ve hata payı var. Örnek
sahneler tam olarak o eksik olan referansı sağlamak için burada.
"""

from __future__ import annotations

import numpy as np

from .sentetik import sahne_kur

# Panelle aynı palet (dogrulama.py ile ortak)
_ZEMIN = (238, 236, 232)          # BGR
_MASA = (206, 214, 224)
_IZGARA = (188, 196, 206)
_ENGEL = (86, 84, 82)
_SERBEST = (214, 222, 214)
_HEDEF = (52, 104, 235)           # BGR ≈ #eb6834'ün tersi: turuncu
_HEDEF2 = (122, 175, 27)


def _dunya_haritasi(H: np.ndarray, boyut_px: tuple[int, int]):
    """
    Her piksel için düzlem üzerindeki dünya koordinatı.

    Kamera arkasına düşen pikseller (projektif ölçek işaret değiştirir) geçersiz
    işaretlenir; oralara boyamak sahneyi düzlemin yanlış yarısıyla doldururdu.
    """
    yuk, gen = boyut_px
    yy, xx = np.mgrid[0:yuk, 0:gen]
    px = np.stack([xx.ravel(), yy.ravel(), np.ones(yuk * gen)], axis=1).astype(float)
    hn = px @ np.linalg.inv(H).T
    w = hn[:, 2]

    # Orijin kamera önünde; oradaki işaret "önde" demek.
    onde_isaret = np.sign(np.array([0.0, 0.0, 1.0]) @ np.linalg.inv(H).T[:, 2] or 1.0)
    gecerli = (np.abs(w) > 1e-9) & (np.sign(w) == onde_isaret)

    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    X = (hn[:, 0] / w).reshape(yuk, gen)
    Y = (hn[:, 1] / w).reshape(yuk, gen)
    return X, Y, gecerli.reshape(yuk, gen)


def _dunyayi_izdusur(H: np.ndarray, dunya) -> np.ndarray:
    d = np.atleast_2d(np.asarray(dunya, dtype=float))
    h = np.hstack([d, np.ones((len(d), 1))]) @ H.T
    return h[:, :2] / h[:, 2:3]


def _isaretli_alan(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _aruco_bas(tuval: np.ndarray, H: np.ndarray, kenar_mm: float,
               isaret_id: int = 7, sozluk: str = "DICT_4X4_50") -> None:
    """
    ArUco işaretini düzleme perspektifiyle yapıştırır.

    İki incelik var:

    1) Kamera düzleme üstten baktığı için dünya -> görüntü dönüşümü yönelim
       ÇEVİRİYOR (bir yansıma içeriyor). Köşeleri olduğu gibi eşlersek işaret
       görüntüye aynalanmış düşer ve hiçbir dedektör onu tanımaz — aynalanmış
       kod sözlükte yok. Hedef dörtgenin işaretli alanına bakıp gerekirse
       eşlemeyi çeviriyoruz. Sonuçta dünya çerçevesi `kare_dunya` sırasına göre
       aynalanmış oluyor; yansıma bir izometri olduğundan ölçülen uzunluk, alan
       ve genişlik değerleri bundan etkilenmiyor.

    2) İşaretin çevresine beyaz bir sessiz bölge bırakıyoruz — basılı bir kâğıt
       gibi. ArUco'nun köşe bulması dış siyah kenarın etrafındaki kontrasta
       dayanıyor; zemin koyulaşırsa tespit sessizce düşer.
    """
    import cv2

    s = 400
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, sozluk))
    isaret = cv2.cvtColor(cv2.aruco.generateImageMarker(d, isaret_id, s),
                          cv2.COLOR_GRAY2BGR)

    yari = kenar_mm / 2.0
    yuk, gen = tuval.shape[:2]
    kaynak = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)

    def yapistir(kaynak_img, kose_yari):
        # ArUco köşe sırası: sol üst, sağ üst, sağ alt, sol alt.
        dunya = np.array([[-kose_yari, -kose_yari], [kose_yari, -kose_yari],
                          [kose_yari, kose_yari], [-kose_yari, kose_yari]])
        hedef = _dunyayi_izdusur(H, dunya).astype(np.float32)
        if _isaretli_alan(hedef) * _isaretli_alan(kaynak) < 0:
            hedef = hedef[[1, 0, 3, 2]]
        M = cv2.getPerspectiveTransform(kaynak, hedef)
        warp = cv2.warpPerspective(kaynak_img, M, (gen, yuk), flags=cv2.INTER_LINEAR)
        maske = cv2.warpPerspective(np.full(kaynak_img.shape[:2], 255, np.uint8), M,
                                    (gen, yuk), flags=cv2.INTER_NEAREST)
        tuval[maske > 127] = warp[maske > 127]

    yapistir(np.full((s, s, 3), 250, np.uint8), yari * 1.35)   # sessiz bölge
    yapistir(isaret, yari)


def _disk_bas(tuval, H, merkez_mm, yaricap_mm, renk, etiket=None, *, nisan=False):
    """Düzlem üzerinde dairesel bir nesne (perspektifte elips olur)."""
    import cv2

    aci = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cevre = np.asarray(merkez_mm) + yaricap_mm * np.column_stack([np.cos(aci), np.sin(aci)])
    h = np.hstack([cevre, np.ones((len(cevre), 1))]) @ H.T
    px = (h[:, :2] / h[:, 2:3]).astype(np.int32)
    cv2.fillPoly(tuval, [px], renk, lineType=cv2.LINE_AA)

    h0 = np.append(merkez_mm, 1.0) @ H.T
    m = (h0[:2] / h0[2]).astype(int)
    if nisan:
        cv2.drawMarker(tuval, tuple(m), (255, 255, 255), cv2.MARKER_CROSS, 18, 2,
                       cv2.LINE_AA)
    if etiket:
        cv2.putText(tuval, etiket, (m[0] + 14, m[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return m.astype(float)


# Telefon: ölçülecek nesne. Boyutları referans listesinde YOK — kullanıcının
# gerçekten bilmediği bir şeyi ölçüyor olması demonun bütün anlamı.
TELEFON_EN, TELEFON_BOY = 146.7, 71.5
PARA_MM = 26.15                      # 1 TL, çap


def _en_genis_capin_uclari(H: np.ndarray, merkez_mm, yaricap_mm: float):
    """
    Dairenin görüntüdeki EN GENİŞ yerinin iki ucu.

    Yuvarlak referansın perspektifteki izdüşümü elips; hangi çapı ölçtüğün fark
    eder. En geniş yer (elipsin büyük ekseni) kısalmaya uğramamış olan çaptır,
    yani doğru cevabı veren tek çap. Kullanıcı da doğal olarak oradan ölçer;
    ipucu noktalarını oraya koyuyoruz.
    """
    aci = np.linspace(0, np.pi, 720, endpoint=False)
    yon = np.column_stack([np.cos(aci), np.sin(aci)])
    a = _dunyayi_izdusur(H, np.asarray(merkez_mm) + yaricap_mm * yon)
    b = _dunyayi_izdusur(H, np.asarray(merkez_mm) - yaricap_mm * yon)
    k = int(np.argmax(np.hypot(*(a - b).T)))
    return np.array([a[k], b[k]])


def _masa(egim_derece: float, ad: str, aciklama: str):
    """Masaya konmuş bir telefon ve yanındaki 1 TL — panelin asıl senaryosu."""
    import cv2

    boyut = (900, 1300)
    sahne = sahne_kur(referans_boyut_mm=PARA_MM, odak_px=1500.0, boyut_px=boyut,
                      mesafe_mm=520.0, egim_derece=egim_derece, azimut_derece=0.0)
    H = sahne.H_gercek
    X, Y, gecerli = _dunya_haritasi(H, boyut)

    tuval = np.full((*boyut, 3), _ZEMIN, np.uint8)
    masa = gecerli & (np.abs(X) < 420) & (np.abs(Y) < 300)
    tuval[masa] = _MASA
    doku = masa & (np.minimum(Y % 90, 90 - Y % 90) < 1.2)      # ahşap çizgileri
    tuval[doku] = _IZGARA

    # Telefon: köşeleri keskin, tıklanabilir
    merkez = np.array([-40.0, 0.0])
    kose_mm = np.array([[-TELEFON_EN / 2, -TELEFON_BOY / 2],
                        [TELEFON_EN / 2, -TELEFON_BOY / 2],
                        [TELEFON_EN / 2, TELEFON_BOY / 2],
                        [-TELEFON_EN / 2, TELEFON_BOY / 2]]) + merkez
    kose_px = _dunyayi_izdusur(H, kose_mm)
    cv2.fillPoly(tuval, [kose_px.astype(np.int32)], (46, 44, 42), cv2.LINE_AA)
    ekran = _dunyayi_izdusur(H, (kose_mm - merkez) * 0.9 + merkez)
    cv2.fillPoly(tuval, [ekran.astype(np.int32)], (28, 26, 25), cv2.LINE_AA)

    # 1 TL: telefonun hemen yanında, masanın üstünde
    para_merkez = np.array([110.0, 0.0])
    _disk_bas(tuval, H, para_merkez, PARA_MM / 2, (86, 158, 196))
    _disk_bas(tuval, H, para_merkez, PARA_MM / 2 * 0.72, (104, 178, 214))
    para_uclari = _en_genis_capin_uclari(H, para_merkez, PARA_MM / 2)

    return dict(
        goruntu=tuval,
        ad=f"ornek-{ad}.png",
        aciklama=aciklama,
        referans=dict(tur="olcek", ad="1_tl", uzunluk_mm=PARA_MM),
        varsayilan_olcum="kutu",
        gercek={
            "en": dict(deger=TELEFON_EN, birim="mm", aciklama="telefonun uzun kenarı"),
            "boy": dict(deger=TELEFON_BOY, birim="mm", aciklama="telefonun kısa kenarı"),
            "alan": dict(deger=TELEFON_EN * TELEFON_BOY / 100.0, birim="cm²",
                         aciklama="telefonun yüzü"),
        },
        ipucu=dict(referans=para_uclari.tolist(), kutu=kose_px.tolist()),
    )


def _duz(tohum: int):
    return _masa(1.5, "duz",
                 "Masaya konmuş telefon ve yanında 1 TL. Fotoğraf neredeyse tam "
                 "tepeden çekilmiş — bu, basit ölçek modelinin doğru çalıştığı hâl.")


def _egik(tohum: int):
    return _masa(26.0, "egik",
                 "Aynı sahne, ama fotoğraf eğik çekilmiş. Tek uzunluktan kurulan "
                 "ölçek perspektifi düzeltemez; sistemin bunu fark edip fark "
                 "etmediğine bak.")


def _gecit(tohum: int):
    """Enkaz senaryosu: ortada daralan serbest koridor."""
    boyut = (900, 1400)
    sahne = sahne_kur(referans_boyut_mm=200.0, odak_px=1100.0, boyut_px=boyut,
                      mesafe_mm=2600.0, egim_derece=18.0, azimut_derece=0.0)
    H = sahne.H_gercek
    X, Y, gecerli = _dunya_haritasi(H, boyut)

    DAR_MM, GENIS_MM = 520.0, 900.0
    yari = np.where(np.abs(X) < 420.0, DAR_MM / 2.0, GENIS_MM / 2.0)
    koridor = gecerli & (np.abs(Y) < yari) & (np.abs(X) < 1500.0)
    saha = gecerli & (np.abs(X) < 1800.0) & (np.abs(Y) < 1300.0)

    tuval = np.full((*boyut, 3), _ZEMIN, np.uint8)
    tuval[saha] = _ENGEL
    tuval[koridor] = _SERBEST

    _aruco_bas(tuval, H, 200.0, isaret_id=3)

    # Serbest koridorun köşeleri: "çokgeni buraya çiz" ipucu
    cokgen_mm = np.array([
        [-1400, -GENIS_MM / 2], [-420, -GENIS_MM / 2], [-420, -DAR_MM / 2],
        [420, -DAR_MM / 2], [420, -GENIS_MM / 2], [1400, -GENIS_MM / 2],
        [1400, GENIS_MM / 2], [420, GENIS_MM / 2], [420, DAR_MM / 2],
        [-420, DAR_MM / 2], [-420, GENIS_MM / 2], [-1400, GENIS_MM / 2],
    ], dtype=float)
    h = np.hstack([cokgen_mm, np.ones((len(cokgen_mm), 1))]) @ H.T
    cokgen_px = h[:, :2] / h[:, 2:3]

    return dict(
        goruntu=tuval,
        ad="ornek-gecit.png",
        aciklama="Enkaz koridoru: 200 mm ArUco referansı, ortada daralan serbest alan. "
                 "Serbest alanın çevresini çokgen olarak çiz.",
        referans=dict(tur="aruco", kenar_mm=200.0),
        varsayilan_olcum="gecit",
        gercek={
            "gecit": dict(deger=DAR_MM, birim="mm", aciklama="koridorun en dar yeri"),
        },
        ipucu=dict(gecit=cokgen_px.tolist()),
        ayak_izi_mm=480.0,
    )


SAHNELER = {"duz": _duz, "egik": _egik, "gecit": _gecit}

# Panelin gösterdiği örnekler: hepsi "nesnenin ölçüsü" akışına uyuyor.
PANEL_SAHNELERI = ["duz", "egik"]


def ornek_sahne(ad: str = "tezgah", *, tohum: int = 0) -> dict:
    """Adı verilen örnek sahneyi üretir; doğru cevabı da birlikte döner."""
    if ad not in SAHNELER:
        raise KeyError(f"Bilinmeyen örnek '{ad}'. Seçenekler: {sorted(SAHNELER)}")
    return SAHNELER[ad](tohum)

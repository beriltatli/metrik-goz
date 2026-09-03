"""
DENEY 1 — "1 piksel kaç mm?" sorusunun görüntü boyunca değiştiğini kendi
kodunla gör.

Bu dosya SENİN yazdığın `_uygula` fonksiyonunu kullanıyor. Adım 1'i
bitirmeden çalışmaz. Bitirince çalıştır:

    python deneyler/deney_01_olcek.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from metrik_goz.homografi import _uygula

# Masaya 1 metre yukarıdan, 45 derece eğik bakan bir kamera.
# (Bu matrisin nereden geldiğini adım 2'de sen kuracaksın; şimdilik hazır.)
H = np.array([
    [-6.78822510e-01, -1.40000000e+00,  9.60000000e+02],
    [-1.37178716e+00,  0.00000000e+00,  5.40000000e+02],
    [-7.07106781e-04,  0.00000000e+00,  1.00000000e+00]
])
H = H / H[2, 2]
H_ters = np.linalg.inv(H)

KAMERA = 1000 * np.array([np.sin(np.radians(45)), 0.0, np.cos(np.radians(45))])


def dunyaya(px):
    """Piksel -> masa üstündeki nokta (mm). Senin _uygula'nı kullanıyor."""
    return _uygula(H_ters, px)[0]


def mm_basina_piksel(px):
    """O pikselin civarında 1 pikselin kaç mm'ye denk geldiği."""
    merkez = dunyaya(px)
    sag = dunyaya([px[0] + 1, px[1]]) - merkez
    asagi = dunyaya([px[0], px[1] + 1]) - merkez
    return float(np.sqrt(abs(np.linalg.det(np.column_stack([sag, asagi])))))


print("Kamera masaya 1 m yukarıdan 45° eğik bakıyor · odak 1400 px\n")
print(f"{'piksel':>12} {'yerdeki x (mm)':>16} {'kameraya uzaklık':>18} {'1 piksel = ? mm':>17}")
print("-" * 68)
for v in (980, 860, 700, 540, 400, 300):
    m = dunyaya([960, v])
    uzaklik = np.linalg.norm(np.array([m[0], m[1], 0.0]) - KAMERA)
    print(f"{'(960,' + str(v) + ')':>12} {m[0]:>15.0f} {uzaklik:>15.0f} mm {mm_basina_piksel([960, v]):>15.2f}")

a = mm_basina_piksel([960, 980])
b = mm_basina_piksel([960, 300])
print("-" * 68)
print(f"En uzak piksel {a:.2f} mm, en yakın piksel {b:.2f} mm  ->  {a / b:.1f} kat fark.")
print("\nTek bir 'mm/piksel' sayısı yazsaydık, görüntünün bir ucunda")
print(f"%{abs(a / b - 1) * 100:.0f} hata yapardık. Homografi tam da bunu çözüyor.")

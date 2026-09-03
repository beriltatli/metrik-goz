"""
metrik-goz — tek fotoğraftan gerçek ölçü, dürüst hata payıyla.

Temel kullanım:

    from metrik_goz import Homografi, olcum, belirsizlik

    h = Homografi.kur(referans_dunya_mm, referans_resim_px)
    mm = olcum.mesafe(h, (120, 340), (610, 355))

    sonuc = belirsizlik.monte_carlo(
        referans_dunya_mm, referans_resim_px,
        lambda hh, nn: olcum.mesafe(hh, nn[0], nn[1]),
        noktalar_px=[(120, 340), (610, 355)],
        sigma_px=0.5,
    )
    print(sonuc)     # 412.3 ± 8.7 mm (%95: 395.1–429.4)

Sınır: ölçülen her şey referansla aynı düzlem üzerinde olmalı.
"""

from .homografi import Homografi, dlt
from .olcum import Gecit, Kutu, alan, en_dar_gecit, kutu, mesafe, uzunluk
from .belirsizlik import Olcum, analitik, monte_carlo, parametre_kovaryansi
from . import lm, olcum, belirsizlik, sentetik, referans

__all__ = [
    "Homografi", "dlt",
    "Gecit", "Kutu", "mesafe", "uzunluk", "alan", "kutu", "en_dar_gecit",
    "Olcum", "monte_carlo", "analitik", "parametre_kovaryansi",
    "lm", "olcum", "belirsizlik", "sentetik", "referans",
]

__version__ = "0.1.0"

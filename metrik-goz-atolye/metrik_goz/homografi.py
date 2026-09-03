"""
Düzlem homografisi.

ADIM 1 — Burada iki fonksiyon yazacaksın: `_homojen` ve `_uygula`.
Gerisi sonraki adımlarda gelecek.
"""

from __future__ import annotations

import numpy as np


def _homojen(noktalar: np.ndarray) -> np.ndarray:
    """
    (N, 2) biçimindeki noktalara 1'lerden oluşan üçüncü bir sütun ekler.

        [[3, 4],          [[3, 4, 1],
         [7, 1]]    ->     [7, 1, 1]]

    Yani (N, 2) girer, (N, 3) çıkar.

    İpucu: np.ones ve np.hstack işini görür.
    """
    raise NotImplementedError("ADIM 1a — bunu sen yazacaksın")


def _uygula(H: np.ndarray, noktalar) -> np.ndarray:
    """
    3x3 homografi matrisini noktalara uygular. (N, 2) girer, (N, 2) çıkar.

    Üç aşama:
      1) Noktaları homojen yap:            (N, 2) -> (N, 3)
      2) H ile çarp:                        her nokta için  H @ [x, y, 1]
         (Dikkat: noktalar satır satır duruyor, o yüzden  noktalar @ H.T)
      3) Üçüncü bileşene bölerek geri dön:  [a, b, w] -> [a/w, b/w]

    Üçüncü adım perspektifin ta kendisi. Uzaktaki noktalarda w büyür ve
    nokta merkeze doğru büzülür — fotoğrafta yolun ufukta daralmasının
    sebebi bu bölme.

    w sıfıra çok yakınsa nokta "sonsuzda" demektir (ufuk çizgisi üstünde);
    böyle bir noktayı ölçemeyiz. Şimdilik çok küçük w'leri 1e-12'ye
    kırpman yeter, ileride düzgün bir kontrol ekleyeceğiz.

    İpucu: np.atleast_2d ile tek nokta verilse de (1, 2) hâline getir,
    böylece fonksiyon hem tek nokta hem nokta listesiyle çalışır.
    """
    raise NotImplementedError("ADIM 1b — bunu sen yazacaksın")

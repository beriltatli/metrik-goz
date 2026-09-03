# metrik-goz — atölye

Bu klasörde projeyi **sen** yazıyorsun. Her adımda ne yazacağın bir docstring
olarak duruyor, testleri hazır. `pytest` kırmızıdan yeşile döndüğünde o adım
bitmiş demektir.

## Kurulum (bir kez)

```bash
cd metrik-goz-atolye
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pip install -e .
```

Kontrol: `pytest -q` çalışmalı (şu an testler kırmızı olacak, normal).

## Yol haritası

| Adım | Konu | Ne yazacaksın |
|---|---|---|
| **1** | Homojen koordinatlar ve projektif dönüşüm | `_homojen`, `_uygula` |
| 2 | Sentetik sahne: doğru cevabı bilinen test verisi | `kamera_homografisi`, `sahne_kur` |
| 3 | DLT — 4 noktadan homografi | `dlt` |
| 4 | Hartley normalizasyonu — neden şart | `_normalizasyon_matrisi` |
| 5 | Yeniden izdüşüm hatası — DLT neden yetmiyor | `_artiklar` |
| 6 | Jacobian'ı elle türetmek | `_jacobian` |
| 7 | Levenberg–Marquardt çözücüsü | `lm.coz` |
| 8 | Ölçüm katmanı: mesafe, uzunluk, alan | `mesafe`, `alan` |
| 9 | En dar geçit taraması | `en_dar_gecit` |
| 10 | Monte Carlo ile belirsizlik | `monte_carlo` |
| 11 | Analitik yayılım ve kovaryans | `parametre_kovaryansi`, `analitik` |
| 12 | Kapsama testi — asıl sınav | doğrulama |

## Kurallar

- Önce kendin dene. Cevap anahtarına bakmadan en az 15 dakika uğraş.
- Test yeşile döndükten sonra anahtarla karşılaştır ve **farkı anla**.
- Anlamadığın satırı bırakma; sor. Mülakatta sorulacak olan tam o satır.

"""
Web paneli — aynı çekirdeği tarayıcıdan kullanmak için.

Panel yeni bir ölçüm yolu AÇMIYOR: gördüğün her sayı `metrik_goz.olcum` ve
`metrik_goz.belirsizlik` içindeki, testleri geçen aynı fonksiyonlardan geliyor.
Buradaki katman yalnız "görüntü + tıklanan noktalar" girdisini toplayıp sonucu
okunur biçimde geri veriyor.

    from metrik_goz.web import uygulama_kur
    uygulama_kur().run(port=8000)

ya da:

    metrik-goz panel --port 8000
"""

from .sunucu import uygulama_kur

__all__ = ["uygulama_kur"]

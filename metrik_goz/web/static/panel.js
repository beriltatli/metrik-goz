/*
  metrik-göz paneli — tarayıcı tarafı.

  Tek kural: burada hiçbir şey ölçülmüyor. Tuval yalnız piksel koordinatı
  topluyor, sunucu ölçüyor. Ölçümün iki ayrı yerde iki farklı biçimde
  hesaplanması, bir gün ikisinin ayrışması demektir.

  Koordinat düzlemleri:
    görüntü px  — sunucunun ölçtüğü ızgara, tıklamalar buna çevrilir
    tuval css px — ekranda gördüğümüz yer  (gorunum ile dönüşür)
*/
"use strict";

const VERI = JSON.parse(document.getElementById("sunucu-verisi").textContent);
const $ = (kimlik) => document.getElementById(kimlik);

const RENK = getComputedStyle(document.documentElement);
const renk = (ad, yedek) => (RENK.getPropertyValue(ad).trim() || yedek);

const OLCUM_KURALI = {
  mesafe:  { enAz: 2, enCok: 2,        kapali: false,
    ipucu: "Aralarındaki mesafeyi istediğin iki noktayı tıkla." },
  uzunluk: { enAz: 2, enCok: Infinity, kapali: false,
    ipucu: "Kırık çizginin köşelerini sırayla tıkla. Toplam boy ölçülür." },
  alan:    { enAz: 3, enCok: Infinity, kapali: true,
    ipucu: "Çokgenin köşelerini sırayla tıkla; son köşe ilkine kendiliğinden bağlanır." },
  gecit:   { enAz: 3, enCok: Infinity, kapali: true,
    ipucu: "Geçilebilir (serbest) alanın çevresini çokgen olarak çiz. En dar kesit " +
           "bulunur ve karar güven aralığının ALT ucuna göre verilir." },
};

const REF_IPUCU = {
  aruco: "İşareti otomatik bul; köşeler alt piksel doğrulukta okunur ve elle " +
         "düzenlenmez.",
  kare: "Dört köşeyi sırayla tıkla: sol üst → sağ üst → sağ alt → sol alt.",
  dikdortgen: "Dört köşeyi sırayla tıkla: sol üst → sağ üst → sağ alt → sol alt.",
  nesne: "Dört köşeyi sırayla tıkla: sol üst → sağ üst → sağ alt → sol alt. " +
         "Nesne referansla aynı düzlemde, düz durmalı.",
};

const durum = {
  gorsel: null,          // {kimlik, url, genislik, yukseklik, ad}
  img: null,             // HTMLImageElement
  mod: "olcum",          // "referans" | "olcum" — ArUco varsayılan, köşe tıklanmıyor
  ref: { tur: "aruco", koseler: [], etiket: null, otomatik: false },
  olcum: { tur: "mesafe", noktalar: [] },
  gorunum: { o: 1, x: 0, y: 0 },
  sonuc: null,
  demo: null,            // örnek sahnenin doğru cevabı ve ipucu noktaları
  imlec: null,
  suruklenen: null,
  kaydirma: null,
  bosluk: false,
  calisiyor: false,
};

const tuval = $("tuval");
const ctx = tuval.getContext("2d");
const buyutec = $("buyutec");
const bctx = buyutec.getContext("2d");
const sarmal = $("tuval-sarmal");

/* ================================================================ görünüm */
function tuvaliOlcekle() {
  const dpr = window.devicePixelRatio || 1;
  const g = sarmal.clientWidth, y = sarmal.clientHeight;
  if (!g || !y) return;
  tuval.width = Math.round(g * dpr);
  tuval.height = Math.round(y * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ciz();
}

function sigdir() {
  if (!durum.gorsel) return;
  const g = sarmal.clientWidth, y = sarmal.clientHeight;
  const o = Math.min(g / durum.gorsel.genislik, y / durum.gorsel.yukseklik) * 0.96;
  durum.gorunum = {
    o,
    x: (g - durum.gorsel.genislik * o) / 2,
    y: (y - durum.gorsel.yukseklik * o) / 2,
  };
  ciz();
}

function yakinlastir(carpan, merkezCss) {
  if (!durum.gorsel) return;
  const gor = durum.gorunum;
  const yeni = Math.min(40, Math.max(0.02, gor.o * carpan));
  const m = merkezCss || { x: sarmal.clientWidth / 2, y: sarmal.clientHeight / 2 };
  // Yakınlaştırma imlecin altındaki görüntü noktasını sabit tutmalı.
  gor.x = m.x - (m.x - gor.x) * (yeni / gor.o);
  gor.y = m.y - (m.y - gor.y) * (yeni / gor.o);
  gor.o = yeni;
  ciz();
}

const ekrana = (p) => ({ x: p[0] * durum.gorunum.o + durum.gorunum.x,
                         y: p[1] * durum.gorunum.o + durum.gorunum.y });

function cssKonum(olay) {
  const k = tuval.getBoundingClientRect();
  return { x: olay.clientX - k.left, y: olay.clientY - k.top };
}

function goruntuye(css) {
  return [(css.x - durum.gorunum.x) / durum.gorunum.o,
          (css.y - durum.gorunum.y) / durum.gorunum.o];
}

/* ================================================================ çizim */
function ciz() {
  const g = sarmal.clientWidth, y = sarmal.clientHeight;
  ctx.clearRect(0, 0, g, y);
  if (!durum.img) return;

  const gor = durum.gorunum;
  ctx.save();
  ctx.translate(gor.x, gor.y);
  ctx.scale(gor.o, gor.o);
  ctx.imageSmoothingEnabled = gor.o < 3;      // piksel seviyesinde net kalsın
  ctx.drawImage(durum.img, 0, 0);
  ctx.restore();

  cizCokgen(durum.ref.koseler, renk("--mavi", "#2a78d6"), true, "K");
  cizOlcum();
  cizGecitSonucu();
  cizOlcuEtiketleri();
}

function cizCokgen(noktalar, cizgiRengi, kapali, onEk) {
  if (!noktalar.length) return;
  const p = noktalar.map(ekrana);

  if (p.length > 1) {
    ctx.save();
    ctx.strokeStyle = cizgiRengi;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(p[0].x, p[0].y);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
    if (kapali && p.length > 2) {
      ctx.closePath();
      ctx.fillStyle = cizgiRengi + "22";
      ctx.fill();
    }
    ctx.stroke();
    ctx.restore();
  }
  p.forEach((nokta, i) => nokta && cizNokta(nokta, cizgiRengi, onEk + (i + 1)));
}

function cizNokta(p, dolgu, etiket) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = dolgu;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = renk("--yuzey", "#fff");
  ctx.stroke();
  if (etiket) {
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.textBaseline = "middle";
    const g = ctx.measureText(etiket).width;
    ctx.fillStyle = renk("--yuzey", "#fff");
    ctx.globalAlpha = 0.88;
    ctx.fillRect(p.x + 8, p.y - 8, g + 6, 15);
    ctx.globalAlpha = 1;
    ctx.fillStyle = dolgu;
    ctx.fillText(etiket, p.x + 11, p.y);
  }
  ctx.restore();
}

function cizOlcum() {
  const kural = OLCUM_KURALI[durum.olcum.tur];
  cizCokgen(durum.olcum.noktalar, renk("--turuncu", "#eb6834"), kural.kapali, "");
}

function cizOlcuEtiketleri() {
  const s = durum.sonuc;
  if (!s || !s.parcalar || !durum.olcum.noktalar.length) return;
  const p = durum.olcum.noktalar.map(ekrana);
  ctx.save();
  ctx.font = "600 12px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  s.parcalar.forEach((mm, i) => {
    if (!p[i] || !p[i + 1]) return;
    const x = (p[i].x + p[i + 1].x) / 2, y = (p[i].y + p[i + 1].y) / 2;
    const metin = mm >= 1000 ? (mm / 1000).toFixed(3) + " m" : mm.toFixed(1) + " mm";
    const g = ctx.measureText(metin).width;
    ctx.fillStyle = renk("--yuzey", "#fff");
    ctx.globalAlpha = 0.9;
    ctx.fillRect(x - g / 2 - 5, y - 9, g + 10, 18);
    ctx.globalAlpha = 1;
    ctx.fillStyle = renk("--turuncu", "#eb6834");
    ctx.fillText(metin, x, y);
  });
  ctx.restore();
}

function cizGecitSonucu() {
  const g = durum.sonuc && durum.sonuc.gecit;
  if (!g || !g.cizgi_px) return;
  const [a, b] = g.cizgi_px.map(ekrana);
  ctx.save();
  ctx.strokeStyle = renk("--kirmizi", "#cc3b2f");
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  const metin = g.genislik_mm.toFixed(0) + " mm";
  ctx.font = "700 13px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
  const w = ctx.measureText(metin).width;
  ctx.fillStyle = renk("--kirmizi", "#cc3b2f");
  ctx.fillRect(mx - w / 2 - 6, my - 22, w + 12, 19);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(metin, mx, my - 12.5);
  ctx.restore();
}

/* ---------------------------------------------------------------- büyüteç */
function buyuteciCiz(css) {
  if (!durum.img || durum.kaydirma) { buyutec.style.display = "none"; return; }
  const KAT = 6, B = 132;
  const ip = goruntuye(css);
  const yari = B / (2 * KAT);
  bctx.save();
  bctx.clearRect(0, 0, B, B);
  bctx.beginPath();
  bctx.arc(B / 2, B / 2, B / 2, 0, Math.PI * 2);
  bctx.clip();
  bctx.fillStyle = renk("--yuzey2", "#eee");
  bctx.fillRect(0, 0, B, B);
  bctx.imageSmoothingEnabled = false;
  bctx.drawImage(durum.img, ip[0] - yari, ip[1] - yari, 2 * yari, 2 * yari, 0, 0, B, B);
  bctx.strokeStyle = renk("--turuncu", "#eb6834");
  bctx.lineWidth = 1;
  bctx.beginPath();
  bctx.moveTo(B / 2, B / 2 - 12); bctx.lineTo(B / 2, B / 2 + 12);
  bctx.moveTo(B / 2 - 12, B / 2); bctx.lineTo(B / 2 + 12, B / 2);
  bctx.stroke();
  bctx.restore();

  buyutec.style.display = "block";
  const sag = css.x > sarmal.clientWidth - 170;
  buyutec.style.left = (sag ? css.x - 150 : css.x + 18) + "px";
  buyutec.style.top = Math.min(Math.max(css.y - 66, 6),
                               sarmal.clientHeight - 138) + "px";
}

/* ================================================================ noktalar */
function aktifListe() {
  return durum.mod === "referans" ? durum.ref.koseler : durum.olcum.noktalar;
}

function noktaEkle(p) {
  if (durum.mod === "referans") {
    if (durum.ref.tur === "aruco") {
      bilgi("ArUco köşeleri otomatik bulunuyor. Elle işaretlemek için referans " +
            "türünü Kare / Dikdörtgen / Bilinen nesne yap.");
      return;
    }
    if (durum.ref.koseler.length >= 4) {
      bilgi("Referans için tam 4 köşe gerekiyor; fazlasını eklemeden önce Temizle.");
      return;
    }
    durum.ref.koseler.push(p);
    durum.ref.otomatik = false;
    if (durum.ref.koseler.length === 4) modSec("olcum");
  } else {
    const kural = OLCUM_KURALI[durum.olcum.tur];
    if (durum.olcum.noktalar.length >= kural.enCok) {
      durum.olcum.noktalar.shift();          // mesafe: en eski noktayı düşür
    }
    durum.olcum.noktalar.push(p);
  }
  durum.sonuc = null;
  guncelle();
}

function yakinNokta(css) {
  const esik = 11;
  const listeler = [
    { liste: durum.ref.koseler, kilit: durum.ref.tur === "aruco" },
    { liste: durum.olcum.noktalar, kilit: false },
  ];
  for (const { liste, kilit } of listeler) {
    if (kilit) continue;
    for (let i = liste.length - 1; i >= 0; i--) {
      const e = ekrana(liste[i]);
      if (Math.hypot(e.x - css.x, e.y - css.y) <= esik) return { liste, i };
    }
  }
  return null;
}

function geriAl() {
  const liste = aktifListe();
  if (!liste.length) return;
  liste.pop();
  if (durum.mod === "referans") durum.ref.otomatik = false;
  durum.sonuc = null;
  guncelle();
}

function temizle() {
  aktifListe().length = 0;
  if (durum.mod === "referans") { durum.ref.otomatik = false; durum.ref.etiket = null; }
  durum.sonuc = null;
  guncelle();
}

/* ================================================================ olaylar */
tuval.addEventListener("pointerdown", (o) => {
  if (!durum.img) return;
  const css = cssKonum(o);
  if (o.button === 1 || o.button === 2 || durum.bosluk) {
    durum.kaydirma = { css, gor: { ...durum.gorunum } };
    tuval.classList.add("kaydiriyor");
    tuval.setPointerCapture(o.pointerId);
    o.preventDefault();
    return;
  }
  if (o.button !== 0) return;
  const hedef = yakinNokta(css);
  if (hedef) {
    durum.suruklenen = { ...hedef, tasindi: false };
    tuval.setPointerCapture(o.pointerId);
  } else {
    noktaEkle(goruntuye(css));
  }
});

tuval.addEventListener("pointermove", (o) => {
  const css = cssKonum(o);
  durum.imlec = css;
  if (durum.kaydirma) {
    durum.gorunum.x = durum.kaydirma.gor.x + (css.x - durum.kaydirma.css.x);
    durum.gorunum.y = durum.kaydirma.gor.y + (css.y - durum.kaydirma.css.y);
    ciz();
  } else if (durum.suruklenen) {
    durum.suruklenen.liste[durum.suruklenen.i] = goruntuye(css);
    durum.suruklenen.tasindi = true;
    durum.sonuc = null;
    ciz();
  }
  imlecBilgisi(css);
  buyuteciCiz(css);
});

function suruklemeyiBitir(o) {
  if (durum.kaydirma) {
    durum.kaydirma = null;
    tuval.classList.remove("kaydiriyor");
  }
  if (durum.suruklenen) {
    const tasindi = durum.suruklenen.tasindi;
    durum.suruklenen = null;
    if (tasindi) guncelle();
  }
  if (o && tuval.hasPointerCapture(o.pointerId)) tuval.releasePointerCapture(o.pointerId);
}
tuval.addEventListener("pointerup", suruklemeyiBitir);
tuval.addEventListener("pointercancel", suruklemeyiBitir);
tuval.addEventListener("contextmenu", (o) => o.preventDefault());
tuval.addEventListener("pointerleave", () => {
  buyutec.style.display = "none";
  durum.imlec = null;
  imlecBilgisi(null);
});

tuval.addEventListener("wheel", (o) => {
  if (!durum.img) return;
  o.preventDefault();
  yakinlastir(Math.exp(-o.deltaY * 0.0022), cssKonum(o));
}, { passive: false });

window.addEventListener("keydown", (o) => {
  const yaziyor = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (o.code === "Space" && !yaziyor) {
    durum.bosluk = true;
    tuval.classList.add("kaydir");
    o.preventDefault();
    return;
  }
  if (yaziyor) {
    if (o.key === "Enter") { o.preventDefault(); calistir(); }
    return;
  }
  if ((o.metaKey || o.ctrlKey) && o.key.toLowerCase() === "z") { o.preventDefault(); geriAl(); }
  else if (o.key === "Backspace" || o.key === "Delete") { o.preventDefault(); geriAl(); }
  else if (o.key === "1") modSec("referans");
  else if (o.key === "2") modSec("olcum");
  else if (o.key.toLowerCase() === "f") sigdir();
  else if (o.key === "Enter") calistir();
});
window.addEventListener("keyup", (o) => {
  if (o.code === "Space") { durum.bosluk = false; tuval.classList.remove("kaydir"); }
});

window.addEventListener("resize", tuvaliOlcekle);

/* ---------------------------------------------------------------- dosya */
["dragenter", "dragover"].forEach((ad) =>
  window.addEventListener(ad, (o) => {
    if (!o.dataTransfer || ![...o.dataTransfer.types].includes("Files")) return;
    o.preventDefault();
    $("birak").classList.remove("gizli");
    $("birak").classList.add("uzerinde");
  }));

["dragleave", "drop"].forEach((ad) =>
  window.addEventListener(ad, (o) => {
    if (ad === "dragleave" && o.relatedTarget) return;
    $("birak").classList.remove("uzerinde");
    if (durum.gorsel) $("birak").classList.add("gizli");
  }));

window.addEventListener("drop", (o) => {
  const dosya = o.dataTransfer && o.dataTransfer.files[0];
  if (!dosya) return;
  o.preventDefault();
  gorselYukle(dosya);
});

window.addEventListener("paste", (o) => {
  const oge = [...(o.clipboardData ? o.clipboardData.items : [])]
    .find((x) => x.type.startsWith("image/"));
  if (oge) gorselYukle(oge.getAsFile());
});

$("dosya-sec").onclick = () => $("dosya").click();
$("dosya").onchange = (o) => o.target.files[0] && gorselYukle(o.target.files[0]);
$("geri-al").onclick = geriAl;
$("temizle").onclick = temizle;
$("sigdir").onclick = sigdir;
$("yakinlas").onclick = () => yakinlastir(1.3);
$("uzaklas").onclick = () => yakinlastir(1 / 1.3);
$("mod-referans").onclick = () => modSec("referans");
$("mod-olcum").onclick = () => modSec("olcum");
$("calistir").onclick = calistir;
$("aruco-bul").onclick = arucoBul;

document.querySelectorAll("[data-ornek]").forEach((d) =>
  d.onclick = () => ornekYukle(d.dataset.ornek));

$("ref-turleri").onclick = (o) => {
  const d = o.target.closest("[data-ref]");
  if (!d) return;
  durum.ref = { tur: d.dataset.ref, koseler: [], etiket: null, otomatik: false };
  durum.sonuc = null;
  $("sigma").value = d.dataset.ref === "aruco" ? VERI.sigmalar.aruco : VERI.sigmalar.elle;
  modSec(d.dataset.ref === "aruco" ? "olcum" : "referans");
};

$("olcum-turleri").onclick = (o) => {
  const d = o.target.closest("[data-olcum]");
  if (!d) return;
  const tur = d.dataset.olcum;
  // Örnek sahnedeysek ve bu ölçüm için hazır noktalar varsa doğrudan koyuyoruz:
  // sentetik sahnede doğru cevabı bildiğimiz nokta kümesi zaten belli.
  const hazir = durum.demo && durum.demo.ipucu && durum.demo.ipucu[tur];
  durum.olcum = { tur, noktalar: hazir ? hazir.map((p) => [p[0], p[1]]) : [] };
  durum.sonuc = null;
  modSec("olcum");
};

function modSec(mod) {
  durum.mod = mod;
  guncelle();
}

/* ================================================================ arayüz */
function bilgi(metin) {
  $("durum-metni").textContent = metin;
}

function imlecBilgisi(css) {
  if (!css || !durum.gorsel) {
    $("imlec-px").textContent = "";
    $("imlec-mm").textContent = "";
    return;
  }
  const p = goruntuye(css);
  $("imlec-px").textContent = `px ${p[0].toFixed(1)}, ${p[1].toFixed(1)}`;
  const olcek = durum.sonuc && durum.sonuc.homografi.olcek_mm_px[0];
  $("imlec-mm").textContent = olcek ? `1. noktada ${olcek.toFixed(3)} mm/px` : "";
}

function guncelle() {
  document.querySelectorAll("[data-ref]").forEach((d) =>
    d.classList.toggle("secili", d.dataset.ref === durum.ref.tur));
  document.querySelectorAll("[data-olcum]").forEach((d) =>
    d.classList.toggle("secili", d.dataset.olcum === durum.olcum.tur));
  ["aruco", "kare", "dikdortgen", "nesne"].forEach((ad) =>
    $("ref-" + ad).hidden = durum.ref.tur !== ad);
  $("gecit-ayarlari").hidden = durum.olcum.tur !== "gecit";

  $("mod-referans").classList.toggle("secili", durum.mod === "referans");
  $("mod-olcum").classList.toggle("secili", durum.mod === "olcum");
  $("ref-ipucu").textContent = REF_IPUCU[durum.ref.tur];
  $("olcum-ipucu").textContent = OLCUM_KURALI[durum.olcum.tur].ipucu;
  $("olcek-rozeti").textContent = "%" + Math.round(durum.gorunum.o * 100);

  const kose = durum.ref.koseler.length;
  $("ref-rozeti").textContent = durum.ref.etiket
    ? durum.ref.etiket : (kose ? `${kose}/4 köşe` : "yok");
  const n = durum.olcum.noktalar.length;
  $("olcum-rozeti").textContent = `${n} nokta`;

  const eksik = eksikNe();
  $("calistir").disabled = !!eksik || durum.calisiyor;
  $("calistir-ipucu").textContent = eksik ||
    "Hazır. Enter tuşu da çalıştırır.";
  ciz();
}

function eksikNe() {
  if (!durum.gorsel) return "Önce bir görüntü yükle.";
  if (durum.ref.tur === "aruco" && !durum.ref.koseler.length)
    return "ArUco işaretini bul (ya da elle referans türü seç).";
  if (durum.ref.tur !== "aruco" && durum.ref.koseler.length !== 4)
    return `Referans için 4 köşe gerekiyor (${durum.ref.koseler.length} var).`;
  const kural = OLCUM_KURALI[durum.olcum.tur];
  const n = durum.olcum.noktalar.length;
  if (n < kural.enAz) return `${durum.olcum.tur} için en az ${kural.enAz} nokta gerekiyor (${n} var).`;
  return null;
}

/* ================================================================ sunucu */
async function istek(yol, secenek) {
  const yanit = await fetch(yol, secenek);
  const govde = await yanit.json().catch(() => ({}));
  if (!yanit.ok) throw new Error(govde.hata || `Sunucu hatası (${yanit.status}).`);
  return govde;
}

async function gorselYukle(dosya) {
  bilgi("Görüntü yükleniyor…");
  try {
    const form = new FormData();
    form.append("dosya", dosya);
    const g = await istek("/api/gorsel", { method: "POST", body: form });
    await gorseliBagla(g);
    durum.demo = null;
    bilgi(`${g.ad} · ${g.genislik}×${g.yukseklik} px`);
  } catch (h) {
    hataGoster(h.message);
    bilgi("Yükleme başarısız.");
  }
}

function gorseliBagla(g) {
  return new Promise((coz, red) => {
    const img = new Image();
    img.onload = () => {
      durum.gorsel = g;
      durum.img = img;
      durum.ref.koseler = [];
      durum.ref.etiket = null;
      durum.olcum.noktalar = [];
      durum.sonuc = null;
      $("birak").classList.add("gizli");
      sigdir();
      guncelle();
      coz();
    };
    img.onerror = () => red(new Error("Görüntü tarayıcıda açılamadı."));
    img.src = g.url;
  });
}

async function arucoBul() {
  if (!durum.gorsel) return hataGoster("Önce bir görüntü yükle.");
  bilgi("ArUco işareti aranıyor…");
  try {
    const y = await istek("/api/aruco", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gorsel_id: durum.gorsel.kimlik,
        kenar_mm: Number($("aruco-kenar").value),
        sozluk: $("aruco-sozluk").value,
      }),
    });
    durum.ref = { tur: "aruco", koseler: y.koseler, etiket: y.etiket, otomatik: true };
    $("sigma").value = y.sigma_px;
    durum.sonuc = null;
    modSec("olcum");
    bilgi(`${y.etiket} bulundu.`);
  } catch (h) {
    hataGoster(h.message);
    bilgi("İşaret bulunamadı.");
  }
}

async function ornekYukle(ad) {
  bilgi("Örnek sahne üretiliyor…");
  try {
    const s = await istek("/api/ornek", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ad }),
    });
    await gorseliBagla(s);
    durum.demo = s;

    // Örneğin referansı ArUco: bulup, ipucu noktalarını koyup doğrudan ölçüyoruz.
    $("aruco-kenar").value = s.referans.kenar_mm;
    document.querySelector('[data-ref="aruco"]').click();
    await arucoBul();

    // Ölçüm türünü sunucu bildiriyor: JSON anahtar sırasına güvenmek, sıralamayı
    // değiştiren bir sunucu ayarında paneli sessizce başka bir ölçüme çevirir.
    const tur = s.varsayilan_olcum || Object.keys(s.ipucu)[0];
    document.querySelector(`[data-olcum="${tur}"]`).click();
    if (s.ayak_izi_mm != null) $("ayak-izi").value = s.ayak_izi_mm;
    guncelle();
    bilgi(s.aciklama);
    await calistir();
  } catch (h) {
    hataGoster(h.message);
    bilgi("Örnek yüklenemedi.");
  }
}

function istekGovdesi() {
  const ref = { tur: durum.ref.tur };
  if (durum.ref.tur === "aruco") {
    ref.kenar_mm = Number($("aruco-kenar").value);
    ref.sozluk = $("aruco-sozluk").value;
    ref.koseler = durum.ref.koseler;
    ref.etiket = durum.ref.etiket;
  } else if (durum.ref.tur === "kare") {
    ref.kenar_mm = Number($("kare-kenar").value);
    ref.koseler = durum.ref.koseler;
  } else if (durum.ref.tur === "dikdortgen") {
    ref.genislik_mm = Number($("dik-gen").value);
    ref.yukseklik_mm = Number($("dik-yuk").value);
    ref.koseler = durum.ref.koseler;
  } else {
    ref.nesne = $("nesne-ad").value;
    ref.koseler = durum.ref.koseler;
  }

  const ol = { tur: durum.olcum.tur, noktalar: durum.olcum.noktalar };
  if (durum.olcum.tur === "gecit") {
    ol.ayak_izi_mm = Number($("ayak-izi").value);
    ol.pay_mm = Number($("gecit-pay").value);
    ol.adim_mm = Number($("adim").value);
    ol.ornek_mm = Number($("ornek-mm").value);
  }

  return {
    gorsel_id: durum.gorsel.kimlik,
    referans: ref,
    olcum: ol,
    sigma_px: Number($("sigma").value),
    guven: Number($("guven").value),
    yontem: $("yontem").value,
    mc_n: Number($("mc-n").value),
  };
}

async function calistir() {
  if (durum.calisiyor) return;
  const eksik = eksikNe();
  if (eksik) return hataGoster(eksik);

  durum.calisiyor = true;
  $("calistir").disabled = true;
  $("calistir").textContent = "Ölçülüyor…";
  bilgi("Ölçüm çalışıyor…");
  try {
    const s = await istek("/api/olc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(istekGovdesi()),
    });
    durum.sonuc = s;
    if (s.referans.koseler) durum.ref.koseler = s.referans.koseler;
    sonucuBas(s);
    bilgi(`${s.olcum.metin}  ·  ${s.sure_ms.toFixed(0)} ms`);
  } catch (h) {
    durum.sonuc = null;
    hataGoster(h.message);
    bilgi("Ölçüm başarısız.");
  } finally {
    durum.calisiyor = false;
    $("calistir").textContent = "Çalıştır";
    guncelle();
  }
}

/* ================================================================ sonuç */
function hataGoster(metin) {
  $("sonuc-govde").innerHTML = "";
  const kutu = document.createElement("div");
  kutu.className = "hata-kutu";
  kutu.textContent = metin;
  $("sonuc-govde").append(kutu);
  $("sure-rozeti").hidden = true;
}

const sayi = (v, basamak = 1) => v.toLocaleString("tr-TR", {
  minimumFractionDigits: basamak, maximumFractionDigits: basamak });
const yuzde = (oran, basamak = 2) => "%" + sayi(oran * 100, basamak);

function sonucuBas(s) {
  const o = s.olcum;
  const govde = $("sonuc-govde");
  govde.innerHTML = "";
  $("sure-rozeti").hidden = false;
  $("sure-rozeti").textContent = `${s.sure_ms.toFixed(0)} ms`;

  govde.append(el("div", "ana-deger", [
    el("span", "", sayi(o.deger)),
    el("span", "pay", "± " + sayi(o.std)),
    el("span", "birim", o.birim),
  ]));
  govde.append(el("div", "aralik-metni",
    `%${Math.round(o.guven * 100)} güven aralığı: ${sayi(o.alt)} – ${sayi(o.ust)} ${o.birim}` +
    `   ·   bağıl belirsizlik ${yuzde(o.bagil_hata)}`));

  govde.append(aralikCubugu(o, gercekDeger(s.tur)));

  const gercek = gercekDeger(s.tur);
  if (gercek != null) {
    const icinde = gercek >= o.alt && gercek <= o.ust;
    const sapma = Math.abs(o.deger - gercek) / Math.abs(gercek) * 100;
    govde.append(el("div", "gercek-kutu", [
      el("b", "", `Sentetik sahnenin doğru cevabı: ${sayi(gercek)} ${o.birim}. `),
      document.createTextNode(
        `Ölçüm ${yuzde(sapma / 100)} sapmayla ${icinde ? "aralığın içinde" : "ARALIĞIN DIŞINDA"}. ` +
        (icinde ? "Güven aralığı bu ölçümde tuttu."
                : "Tek ölçümde bu %5 olasılıkla beklenen bir sonuç; kapsama testi için tekrarla.")),
    ]));
  }

  if (s.gecit && s.gecit.karar) govde.append(kararKutusu(s.gecit.karar));
  if (s.gecit) govde.append(profilGrafigi(s.gecit));

  if (s.parcalar && s.parcalar.length > 1) {
    const liste = el("ul", "parca-listesi");
    s.parcalar.forEach((mm, i) => liste.append(el("li", "", [
      el("span", "", `${i + 1}→${i + 2} `),
      document.createTextNode(sayi(mm) + " mm"),
    ])));
    govde.append(liste);
  }

  const kunye = el("dl", "kunye");
  const olcek = s.homografi.olcek_mm_px;
  [["yöntem", o.yontem],
   ["referans", s.referans.etiket],
   ["σ köşe", `${sayi(s.referans.sigma_px, 2)} px`],
   ["yeniden izdüşüm", `${sayi(s.homografi.rms_px, 3)} px`],
   ["yerel ölçek", olcek.length ? `${sayi(olcek[0], 3)} mm/px · 1. nokta` : "—"],
  ].forEach(([k, v]) => { kunye.append(el("dt", "", k)); kunye.append(el("dd", "", String(v))); });
  govde.append(kunye);

  if (s.uyarilar && s.uyarilar.length) {
    const liste = el("ul", "uyari-listesi");
    s.uyarilar.forEach((u) => liste.append(el("li", u.seviye, u.metin)));
    govde.append(liste);
  }
}

function gercekDeger(tur) {
  const g = durum.demo && durum.demo.gercek && durum.demo.gercek[tur];
  return g ? g.deger : null;
}

function el(etiket, sinif, icerik) {
  const d = document.createElement(etiket);
  if (sinif) d.className = sinif;
  if (typeof icerik === "string") d.textContent = icerik;
  else if (Array.isArray(icerik)) d.append(...icerik);
  return d;
}

function aralikCubugu(o, gercek) {
  const kutu = el("div", "aralik-cubugu");
  const degerler = [o.alt, o.ust, o.deger].concat(gercek != null ? [gercek] : []);
  let dusuk = Math.min(...degerler), yuksek = Math.max(...degerler);
  const pay = (yuksek - dusuk) * 0.35 || Math.abs(o.deger) * 0.02 || 1;
  dusuk -= pay; yuksek += pay;
  const yuzde = (v) => ((v - dusuk) / (yuksek - dusuk)) * 100;

  kutu.append(el("div", "zemin"));
  const dolgu = el("div", "dolgu");
  dolgu.style.left = yuzde(o.alt) + "%";
  dolgu.style.width = (yuzde(o.ust) - yuzde(o.alt)) + "%";
  kutu.append(dolgu);

  const isaret = el("div", "isaret");
  isaret.style.left = yuzde(o.deger) + "%";
  kutu.append(isaret);

  const solEtiket = el("div", "etiket sol", sayi(o.alt));
  solEtiket.style.left = yuzde(o.alt) + "%";
  const sagEtiket = el("div", "etiket sag", sayi(o.ust));
  sagEtiket.style.left = yuzde(o.ust) + "%";
  kutu.append(solEtiket, sagEtiket);

  if (gercek != null) {
    const g = el("div", "gercek");
    g.style.left = yuzde(gercek) + "%";
    g.title = "sahnenin doğru cevabı";
    kutu.append(g);
  }
  return kutu;
}

function kararKutusu(karar) {
  const gecer = karar.gecer;
  const kutu = el("div", "karar " + (gecer ? "gecer" : "gecmez"));
  kutu.append(el("span", "etiket", gecer ? "GEÇER" : "GEÇMEZ"));
  const marj = karar.marj_mm;
  let metin = `${sayi(karar.gerekli_mm, 0)} mm gereken genişliğe göre aralığın alt ucu ` +
              `${marj >= 0 ? "+" : ""}${sayi(marj)} mm.`;
  if (!gecer && karar.nokta_tahmini_gecer) {
    metin += " Nokta tahmini geçiyor görünüyor ama alt uç geçmiyor — karar alt uca " +
             "göre veriliyor, çünkü geçemeyeceği yola girmek geçebileceği yolu " +
             "kaçırmaktan pahalı.";
  }
  kutu.append(el("small", "", metin));
  return kutu;
}

function profilGrafigi(g) {
  const NS = "http://www.w3.org/2000/svg";
  const G = 340, Y = 86, KENAR = 6;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "profil");
  svg.setAttribute("viewBox", `0 0 ${G} ${Y}`);
  svg.setAttribute("preserveAspectRatio", "none");

  const p = g.profil_mm, x = g.istasyonlar_mm;
  const enB = Math.max(...p) * 1.1 || 1;
  const xk = (i) => KENAR + (x[i] / (x[x.length - 1] || 1)) * (G - 2 * KENAR);
  const yk = (v) => Y - KENAR - (v / enB) * (Y - 2 * KENAR);

  const yol = document.createElementNS(NS, "path");
  yol.setAttribute("d",
    `M ${xk(0)} ${Y - KENAR} ` +
    p.map((v, i) => `L ${xk(i).toFixed(1)} ${yk(v).toFixed(1)}`).join(" ") +
    ` L ${xk(p.length - 1)} ${Y - KENAR} Z`);
  yol.setAttribute("fill", renk("--mavi", "#2a78d6"));
  yol.setAttribute("fill-opacity", "0.16");
  svg.append(yol);

  const cizgi = document.createElementNS(NS, "polyline");
  cizgi.setAttribute("points", p.map((v, i) => `${xk(i).toFixed(1)},${yk(v).toFixed(1)}`).join(" "));
  cizgi.setAttribute("fill", "none");
  cizgi.setAttribute("stroke", renk("--mavi", "#2a78d6"));
  cizgi.setAttribute("stroke-width", "1.8");
  svg.append(cizgi);

  const i = g.en_dar_indeks;
  const im = document.createElementNS(NS, "circle");
  im.setAttribute("cx", xk(i)); im.setAttribute("cy", yk(p[i])); im.setAttribute("r", "4");
  im.setAttribute("fill", renk("--kirmizi", "#cc3b2f"));
  svg.append(im);

  const yazi = document.createElementNS(NS, "text");
  yazi.setAttribute("x", Math.min(xk(i) + 8, G - 60));
  yazi.setAttribute("y", Math.max(yk(p[i]) - 6, 12));
  yazi.setAttribute("font-size", "11");
  yazi.setAttribute("font-family", "ui-monospace, Menlo, monospace");
  yazi.setAttribute("fill", renk("--kirmizi", "#cc3b2f"));
  yazi.textContent = `${p[i].toFixed(0)} mm`;
  svg.append(yazi);

  const sarmalDiv = el("div");
  sarmalDiv.append(svg);
  sarmalDiv.append(el("p", "ipucu",
    `Serbest genişlik profili: koridor boyunca ${p.length} kesit. ` +
    `Uçlardan ${g.kenar_payi_mm.toFixed(0)} mm atlandı — maskenin bittiği yer ` +
    `geçidin darlığı değildir.`));
  return sarmalDiv;
}

/* ================================================================ başlangıç */
new ResizeObserver(tuvaliOlcekle).observe(sarmal);
tuvaliOlcekle();
guncelle();

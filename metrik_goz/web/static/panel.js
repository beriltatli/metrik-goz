/*
  metrik-göz paneli — tarayıcı tarafı.

  Akış üç adım: fotoğraf → referans → nesne. Mod düğmesi yok; hangi adımdaysan
  tuvaldeki sürükleme onu çizer. Referans tamamlanmamışsa referansı, tamamsa
  nesne kutusunu.

  Burada hiçbir şey ölçülmüyor. Tuval piksel koordinatı topluyor, sunucu
  ölçüyor. Aynı hesabın iki yerde yazılması, bir gün ikisinin ayrışması demek.
*/
"use strict";

const VERI = JSON.parse(document.getElementById("sunucu-verisi").textContent);
const $ = (k) => document.getElementById(k);
const RENK = getComputedStyle(document.documentElement);
const renk = (ad, yedek) => (RENK.getPropertyValue(ad).trim() || yedek);

const REF_KURALI = {
  olcek: { nokta: 2, cizim: "cizgi",
    serit: "Referansın iki ucuna sürükle — yuvarlaksa en geniş yerinden.",
    not: "Madeni para en pratiği: yuvarlak olduğu için hangi açıyla dursa çapı çaptır. " +
         "Ölçtüğün nesnenin yanına, mümkünse üstüne koy." },
  dikdortgen: { nokta: 4, cizim: "dortgen",
    serit: "Dikdörtgenin dört köşesini sırayla tıkla: sol üst → sağ üst → sağ alt → sol alt.",
    not: "Dört köşe perspektifi DÜZELTİR. Eğik çekilmiş fotoğrafta tek doğru yol bu." },
  aruco: { nokta: 4, cizim: "dortgen", otomatik: true,
    serit: "ArUco işareti otomatik aranıyor…",
    not: "En doğru yol: köşeler alt piksel doğrulukta okunuyor ve perspektif düzeltiliyor." },
};

const durum = {
  gorsel: null, img: null,
  ref: { tur: "olcek", noktalar: [], etiket: null },
  nesne: { koseler: [] },
  zorlaReferans: false,
  gorunum: { o: 1, x: 0, y: 0 },
  sonuc: null, demo: null,
  ciziliyor: null, suruklenen: null, kaydirma: null,
  bosluk: false, calisiyor: false,
};

const tuval = $("tuval");
const ctx = tuval.getContext("2d");
const buyutec = $("buyutec");
const bctx = buyutec.getContext("2d");
const sarmal = $("tuval-sarmal");

const refKural = () => REF_KURALI[durum.ref.tur];
const refTamam = () => durum.ref.noktalar.length === refKural().nokta;
const nesneTamam = () => durum.nesne.koseler.length === 4;
const hedef = () => (!refTamam() || durum.zorlaReferans ? "referans" : "nesne");

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
  const o = Math.min(g / durum.gorsel.genislik, y / durum.gorsel.yukseklik) * 0.94;
  durum.gorunum = { o, x: (g - durum.gorsel.genislik * o) / 2,
                    y: (y - durum.gorsel.yukseklik * o) / 2 };
  olcekRozeti();
  ciz();
}

function yakinlastir(carpan, merkezCss) {
  if (!durum.gorsel) return;
  const gor = durum.gorunum;
  const yeni = Math.min(40, Math.max(0.02, gor.o * carpan));
  const m = merkezCss || { x: sarmal.clientWidth / 2, y: sarmal.clientHeight / 2 };
  gor.x = m.x - (m.x - gor.x) * (yeni / gor.o);   // imlecin altındaki nokta sabit kalsın
  gor.y = m.y - (m.y - gor.y) * (yeni / gor.o);
  gor.o = yeni;
  olcekRozeti();
  ciz();
}

const olcekRozeti = () => { $("olcek-rozeti").textContent = "%" + Math.round(durum.gorunum.o * 100); };
const ekrana = (p) => ({ x: p[0] * durum.gorunum.o + durum.gorunum.x,
                         y: p[1] * durum.gorunum.o + durum.gorunum.y });
const cssKonum = (o) => {
  const k = tuval.getBoundingClientRect();
  return { x: o.clientX - k.left, y: o.clientY - k.top };
};
const goruntuye = (c) => [(c.x - durum.gorunum.x) / durum.gorunum.o,
                          (c.y - durum.gorunum.y) / durum.gorunum.o];

/* ================================================================ çizim */
function ciz() {
  const g = sarmal.clientWidth, y = sarmal.clientHeight;
  ctx.clearRect(0, 0, g, y);
  if (!durum.img) return;

  const gor = durum.gorunum;
  ctx.save();
  ctx.translate(gor.x, gor.y);
  ctx.scale(gor.o, gor.o);
  ctx.imageSmoothingEnabled = gor.o < 3;
  ctx.drawImage(durum.img, 0, 0);
  ctx.restore();

  const mavi = renk("--mavi", "#2a78d6");
  const turuncu = renk("--turuncu", "#eb6834");
  cizSekil(durum.ref.noktalar, mavi, refKural().cizim, refEtiketi());
  cizSekil(durum.nesne.koseler, turuncu, "dortgen", null);
  if (durum.ciziliyor) cizSekil(durum.ciziliyor.noktalar, durum.ciziliyor.renk,
                                durum.ciziliyor.cizim, null, true);
  cizKenarOlculeri();
}

function refEtiketi() {
  if (durum.ref.tur !== "olcek" || durum.ref.noktalar.length < 2) return null;
  const mm = Number($("uzunluk-mm").value);
  return isFinite(mm) ? sayi(mm, 2) + " mm" : null;
}

function cizSekil(noktalar, cizgi, bicim, etiket, gecici) {
  if (!noktalar.length) return;
  const p = noktalar.map(ekrana);
  ctx.save();
  ctx.strokeStyle = cizgi;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  if (gecici) ctx.setLineDash([6, 4]);

  if (p.length > 1) {
    ctx.beginPath();
    ctx.moveTo(p[0].x, p[0].y);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
    if (bicim === "dortgen" && p.length > 2) {
      ctx.closePath();
      ctx.fillStyle = cizgi + "1f";
      ctx.fill();
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  if (bicim === "cizgi" && p.length === 2) {
    // Uçlarda dik çentik: nereye kadar ölçtüğün belli olsun.
    const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
    const n = Math.hypot(dx, dy) || 1;
    const ux = (-dy / n) * 7, uy = (dx / n) * 7;
    [p[0], p[1]].forEach((q) => {
      ctx.beginPath();
      ctx.moveTo(q.x - ux, q.y - uy);
      ctx.lineTo(q.x + ux, q.y + uy);
      ctx.stroke();
    });
    if (etiket) yaziKutusu(etiket, (p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2 - 16, cizgi);
  }
  if (!gecici) p.forEach((q) => cizTutamak(q, cizgi));
  ctx.restore();
}

function cizTutamak(p, dolgu) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = dolgu;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = renk("--yuzey", "#fff");
  ctx.stroke();
}

function yaziKutusu(metin, x, y, dolgu) {
  ctx.save();
  ctx.font = "600 12px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const g = ctx.measureText(metin).width;
  ctx.fillStyle = renk("--yuzey", "#fff");
  ctx.globalAlpha = 0.92;
  ctx.fillRect(x - g / 2 - 5, y - 9, g + 10, 18);
  ctx.globalAlpha = 1;
  ctx.fillStyle = dolgu;
  ctx.fillText(metin, x, y);
  ctx.restore();
}

function cizKenarOlculeri() {
  const s = durum.sonuc;
  if (!s || s.tur !== "kutu" || !nesneTamam() || !s.kutu) return;
  const p = durum.nesne.koseler.map(ekrana);
  const turuncu = renk("--turuncu", "#eb6834");
  s.kutu.kenarlar_mm.forEach((mm, i) => {
    const a = p[i], b = p[(i + 1) % 4];
    yaziKutusu(sayi(mm) + " mm", (a.x + b.x) / 2, (a.y + b.y) / 2, turuncu);
  });
}

function buyuteciCiz(css) {
  if (!durum.img || durum.kaydirma) { buyutec.style.display = "none"; return; }
  const KAT = 6, B = 132, ip = goruntuye(css), yari = B / (2 * KAT);
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
  buyutec.style.left = (css.x > sarmal.clientWidth - 170 ? css.x - 150 : css.x + 18) + "px";
  buyutec.style.top = Math.min(Math.max(css.y - 66, 6), sarmal.clientHeight - 138) + "px";
}

/* ================================================================ tuval olayları */
function yakinTutamak(css) {
  const listeler = [durum.nesne.koseler, durum.ref.noktalar];
  for (const liste of listeler) {
    if (liste === durum.ref.noktalar && refKural().otomatik) continue;
    for (let i = liste.length - 1; i >= 0; i--) {
      const e = ekrana(liste[i]);
      if (Math.hypot(e.x - css.x, e.y - css.y) <= 11) return { liste, i };
    }
  }
  return null;
}

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

  const tut = yakinTutamak(css);
  if (tut) {
    durum.suruklenen = { ...tut, tasindi: false };
    tuval.setPointerCapture(o.pointerId);
    return;
  }

  const nokta = goruntuye(css);
  if (hedef() === "referans") {
    if (refKural().otomatik) { bilgi("ArUco köşeleri otomatik bulunuyor."); return; }
    if (durum.ref.tur === "olcek") {
      durum.ciziliyor = { tur: "ref-cizgi", cizim: "cizgi", bas: nokta,
                          noktalar: [nokta, nokta], renk: renk("--mavi", "#2a78d6") };
      tuval.setPointerCapture(o.pointerId);
    } else {
      // Dikdörtgen köşeleri tek tek tıklanır: dörtgen serbest, kutu değil.
      if (durum.ref.noktalar.length >= 4) durum.ref.noktalar.length = 0;
      durum.ref.noktalar.push(nokta);
      durum.zorlaReferans = durum.ref.noktalar.length < 4;
      degisti();
    }
  } else {
    durum.ciziliyor = { tur: "nesne", cizim: "dortgen", bas: nokta,
                        noktalar: kutuKoseleri(nokta, nokta),
                        renk: renk("--turuncu", "#eb6834") };
    tuval.setPointerCapture(o.pointerId);
  }
});

const kutuKoseleri = (a, b) => [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];

tuval.addEventListener("pointermove", (o) => {
  const css = cssKonum(o);
  if (durum.kaydirma) {
    durum.gorunum.x = durum.kaydirma.gor.x + (css.x - durum.kaydirma.css.x);
    durum.gorunum.y = durum.kaydirma.gor.y + (css.y - durum.kaydirma.css.y);
    ciz();
  } else if (durum.suruklenen) {
    durum.suruklenen.liste[durum.suruklenen.i] = goruntuye(css);
    durum.suruklenen.tasindi = true;
    ciz();
  } else if (durum.ciziliyor) {
    const p = goruntuye(css);
    durum.ciziliyor.noktalar = durum.ciziliyor.cizim === "cizgi"
      ? [durum.ciziliyor.bas, p] : kutuKoseleri(durum.ciziliyor.bas, p);
    ciz();
  }
  buyuteciCiz(css);
});

function suruklemeyiBitir(o) {
  if (durum.kaydirma) { durum.kaydirma = null; tuval.classList.remove("kaydiriyor"); }

  if (durum.ciziliyor) {
    const c = durum.ciziliyor;
    durum.ciziliyor = null;
    const a = ekrana(c.noktalar[0]), b = ekrana(c.noktalar[c.cizim === "cizgi" ? 1 : 2]);
    if (Math.hypot(b.x - a.x, b.y - a.y) < 8) {
      bilgi("Çok küçük — basılı tutup sürükle.");
      ciz();
    } else if (c.tur === "ref-cizgi") {
      durum.ref.noktalar = c.noktalar;
      durum.zorlaReferans = false;
      degisti();
    } else {
      durum.nesne.koseler = c.noktalar;
      degisti();
    }
  }

  if (durum.suruklenen) {
    const tasindi = durum.suruklenen.tasindi;
    durum.suruklenen = null;
    if (tasindi) degisti();
  }
  if (o && tuval.hasPointerCapture(o.pointerId)) tuval.releasePointerCapture(o.pointerId);
}
tuval.addEventListener("pointerup", suruklemeyiBitir);
tuval.addEventListener("pointercancel", suruklemeyiBitir);
tuval.addEventListener("contextmenu", (o) => o.preventDefault());
tuval.addEventListener("pointerleave", () => { buyutec.style.display = "none"; });
tuval.addEventListener("wheel", (o) => {
  if (!durum.img) return;
  o.preventDefault();
  yakinlastir(Math.exp(-o.deltaY * 0.0022), cssKonum(o));
}, { passive: false });

window.addEventListener("keydown", (o) => {
  const yaziyor = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (o.code === "Space" && !yaziyor) {
    durum.bosluk = true; tuval.classList.add("kaydir"); o.preventDefault(); return;
  }
  if (yaziyor) return;
  if (o.key.toLowerCase() === "f") sigdir();
  else if (o.key === "Escape") { durum.ciziliyor = null; ciz(); }
});
window.addEventListener("keyup", (o) => {
  if (o.code === "Space") { durum.bosluk = false; tuval.classList.remove("kaydir"); }
});
window.addEventListener("resize", tuvaliOlcekle);

/* ---------------------------------------------------------------- dosya */
["dragenter", "dragover"].forEach((ad) => window.addEventListener(ad, (o) => {
  if (!o.dataTransfer || ![...o.dataTransfer.types].includes("Files")) return;
  o.preventDefault();
  $("birak").classList.remove("gizli");
  $("birak").classList.add("uzerinde");
}));
["dragleave", "drop"].forEach((ad) => window.addEventListener(ad, (o) => {
  if (ad === "dragleave" && o.relatedTarget) return;
  $("birak").classList.remove("uzerinde");
  if (durum.gorsel) $("birak").classList.add("gizli");
}));
window.addEventListener("drop", (o) => {
  const d = o.dataTransfer && o.dataTransfer.files[0];
  if (d) { o.preventDefault(); gorselYukle(d); }
});
window.addEventListener("paste", (o) => {
  const oge = [...(o.clipboardData ? o.clipboardData.items : [])]
    .find((x) => x.type.startsWith("image/"));
  if (oge) gorselYukle(oge.getAsFile());
});

/* ================================================================ arayüz bağları */
$("dosya-sec").onclick = $("dosya-sec-buyuk").onclick =
  $("yeni-olcum").onclick = () => $("dosya").click();
$("dosya").onchange = (o) => o.target.files[0] && gorselYukle(o.target.files[0]);
$("sigdir").onclick = sigdir;
$("yakinlas").onclick = () => yakinlastir(1.3);
$("uzaklas").onclick = () => yakinlastir(1 / 1.3);
$("aruco-bul").onclick = arucoBul;
$("ref-yeniden").onclick = () => {
  durum.ref.noktalar = [];
  durum.ref.etiket = null;
  durum.zorlaReferans = true;
  degisti();
};
$("nesne-yeniden").onclick = () => { durum.nesne.koseler = []; degisti(); };

// Kenar çubuğundaki belirsizlik kartı "gelişmiş" ayarlarının aynası.
function belirsizlikKarti() {
  const guven = Number($("guven").value);
  $("kenar-guven").textContent = yuzde(guven, 0);
  $("kenar-cubuk").style.width = `${(guven * 100).toFixed(0)}%`;
  $("kenar-not").textContent =
    `σ ${sayi(Number($("sigma").value), 1)} px · ${$("mc-n").value} MC örneği`;
}
["sigma", "guven", "mc-n"].forEach((k) => $(k).addEventListener("input", belirsizlikKarti));

document.querySelectorAll("[data-ornek]").forEach((d) =>
  d.onclick = () => ornekYukle(d.dataset.ornek));

$("ref-turleri").onclick = (o) => {
  const d = o.target.closest("[data-ref]");
  if (!d || d.dataset.ref === durum.ref.tur) return;
  durum.ref = { tur: d.dataset.ref, noktalar: [], etiket: null };
  durum.zorlaReferans = true;
  $("sigma").value = d.dataset.ref === "aruco" ? VERI.sigmalar.aruco : VERI.sigmalar.elle;
  degisti();
  if (d.dataset.ref === "aruco" && durum.gorsel) arucoBul();
};

$("uzunluk-ad").onchange = () => {
  const s = $("uzunluk-ad").selectedOptions[0];
  if (s.dataset.mm) $("uzunluk-mm").value = s.dataset.mm;
  else $("uzunluk-mm").focus();
  degisti();
};
$("uzunluk-mm").oninput = () => {
  // Elle bir sayı yazıldığında hazır seçim artık geçerli değil.
  const s = $("uzunluk-ad").selectedOptions[0];
  if (s && s.dataset.mm && Number(s.dataset.mm) !== Number($("uzunluk-mm").value)) {
    $("uzunluk-ad").value = "";
  }
  ciz();
};
$("uzunluk-mm").onchange = degisti;
$("nesne-ad").onchange = () => {
  const ad = $("nesne-ad").value;
  const ozel = !ad;
  $("dikdortgen-olculeri").hidden = !ozel;
  if (!ozel && VERI.nesneler[ad]) {
    $("dik-gen").value = VERI.nesneler[ad][0];
    $("dik-yuk").value = VERI.nesneler[ad][1];
  }
  degisti();
};
["dik-gen", "dik-yuk", "aruco-kenar", "sigma", "guven", "mc-n"].forEach(
  (k) => { $(k).onchange = degisti; });

/* ================================================================ durum akışı */
let zamanlayici = null;

function degisti() {
  arayuzuTazele();
  clearTimeout(zamanlayici);
  if (durum.gorsel && refTamam() && nesneTamam()) {
    zamanlayici = setTimeout(olc, 60);      // sürükleme bitince tek istek
  } else {
    durum.sonuc = null;
    sonucuTemizle();
    ciz();
  }
}

function arayuzuTazele() {
  document.querySelectorAll("[data-ref]").forEach((d) =>
    d.classList.toggle("secili", d.dataset.ref === durum.ref.tur));
  ["olcek", "dikdortgen", "aruco"].forEach((a) =>
    $("ref-" + a).hidden = durum.ref.tur !== a);

  const kural = refKural();
  const foto = !!durum.gorsel;
  const aktif = !foto ? "foto" : hedef() === "referans" ? "ref" : "nesne";

  [["adim-foto", "foto", foto],
   ["adim-ref", "ref", refTamam()],
   ["adim-nesne", "nesne", nesneTamam()]].forEach(([kimlik, kisa, tamam]) => {
    const el = $(kimlik);
    el.classList.toggle("tamam", tamam);
    el.classList.toggle("etkin", aktif === kisa && !tamam);
    $(kisa + "-onay").hidden = !tamam;
    // Kenar çubuğundaki akış listesi aynı durumu gösteriyor.
    const nav = document.querySelector(`[data-akis="${kisa}"]`);
    if (nav) {
      nav.classList.toggle("tamam", tamam);
      nav.classList.toggle("etkin", aktif === kisa && !tamam);
    }
  });

  $("foto-not").textContent = foto
    ? `${durum.gorsel.ad} · ${durum.gorsel.genislik}×${durum.gorsel.yukseklik} px`
    : "Sürükle-bırak ve ⌘V ile yapıştırma da çalışır.";
  $("dosya-sec").textContent = foto ? "Başka fotoğraf seç…" : "Fotoğraf seç…";

  const refNot = durum.ref.etiket && refTamam()
    ? `${durum.ref.etiket}${durum.sonuc && durum.sonuc.referans.piksel_boyu
        ? ` · görüntüde ${Math.round(durum.sonuc.referans.piksel_boyu)} px` : ""}`
    : kural.not;
  $("ref-not").textContent = refNot;
  $("ref-not").classList.toggle("vurgu", refTamam());
  $("ref-yeniden").hidden = !durum.ref.noktalar.length || kural.otomatik;

  $("nesne-not").textContent = nesneTamam()
    ? "Köşeleri sürükleyerek düzeltebilirsin; ölçü anında yenilenir."
    : "Ölçmek istediğin nesnenin üstüne bir kutu çiz (basılı tutup sürükle).";
  $("nesne-yeniden").hidden = !nesneTamam();

  const serit = $("serit");
  if (!foto) serit.hidden = true;
  else if (hedef() === "referans" && !kural.otomatik) {
    serit.hidden = false;
    serit.textContent = durum.ref.tur === "dikdortgen"
      ? `${kural.serit}  (${durum.ref.noktalar.length}/4)` : kural.serit;
  } else if (!nesneTamam()) {
    serit.hidden = false;
    serit.textContent = "Nesnenin üstüne bir kutu çiz.";
  } else serit.hidden = true;

  olcekRozeti();
  ciz();
}

const bilgi = (m) => { $("durum-metni").textContent = m; };

/* ================================================================ sunucu */
async function istek(yol, secenek) {
  const yanit = await fetch(yol, secenek);
  const govde = await yanit.json().catch(() => ({}));
  if (!yanit.ok) throw new Error(govde.hata || `Sunucu hatası (${yanit.status}).`);
  return govde;
}

async function gorselYukle(dosya) {
  bilgi("Fotoğraf yükleniyor…");
  try {
    const form = new FormData();
    form.append("dosya", dosya);
    await gorseliBagla(await istek("/api/gorsel", { method: "POST", body: form }));
    durum.demo = null;
    bilgi("Hazır.");
    if (refKural().otomatik) arucoBul();
  } catch (h) { hataGoster(h.message); bilgi("Yükleme başarısız."); }
}

function gorseliBagla(g) {
  return new Promise((coz, red) => {
    const img = new Image();
    img.onload = () => {
      durum.gorsel = g;
      durum.img = img;
      durum.ref.noktalar = [];
      durum.ref.etiket = null;
      durum.nesne.koseler = [];
      durum.sonuc = null;
      durum.zorlaReferans = false;
      $("birak").classList.add("gizli");
      sonucuTemizle();
      sigdir();
      arayuzuTazele();
      coz();
    };
    img.onerror = () => red(new Error("Fotoğraf tarayıcıda açılamadı."));
    img.src = g.url;
  });
}

async function arucoBul() {
  if (!durum.gorsel) return;
  bilgi("ArUco işareti aranıyor…");
  try {
    const y = await istek("/api/aruco", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gorsel_id: durum.gorsel.kimlik,
                             kenar_mm: Number($("aruco-kenar").value) }),
    });
    durum.ref = { tur: "aruco", noktalar: y.koseler, etiket: y.etiket };
    durum.zorlaReferans = false;
    $("sigma").value = y.sigma_px;
    bilgi(y.etiket + " bulundu.");
    degisti();
  } catch (h) {
    hataGoster(h.message);
    bilgi("İşaret bulunamadı.");
  }
}

async function ornekYukle(ad) {
  bilgi("Örnek sahne üretiliyor…");
  try {
    const s = await istek("/api/ornek", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ad }),
    });
    await gorseliBagla(s);
    durum.demo = s;
    document.querySelector(`[data-ref="${s.referans.tur}"]`).classList.add("secili");
    durum.ref.tur = s.referans.tur;
    if (s.referans.ad) $("uzunluk-ad").value = s.referans.ad;
    if (s.referans.uzunluk_mm) $("uzunluk-mm").value = s.referans.uzunluk_mm;
    $("sigma").value = VERI.sigmalar.elle;
    durum.ref.noktalar = (s.ipucu.referans || []).map((p) => [p[0], p[1]]);
    durum.nesne.koseler = (s.ipucu.kutu || []).map((p) => [p[0], p[1]]);
    bilgi(s.aciklama);
    degisti();
  } catch (h) { hataGoster(h.message); bilgi("Örnek yüklenemedi."); }
}

function istekGovdesi() {
  const ref = { tur: durum.ref.tur };
  if (durum.ref.tur === "olcek") {
    ref.uzunluk_mm = Number($("uzunluk-mm").value);
    ref.ad = $("uzunluk-ad").value || null;
    ref.noktalar = durum.ref.noktalar;
  } else if (durum.ref.tur === "dikdortgen") {
    const ad = $("nesne-ad").value;
    if (ad) { ref.tur = "nesne"; ref.nesne = ad; }
    else {
      ref.genislik_mm = Number($("dik-gen").value);
      ref.yukseklik_mm = Number($("dik-yuk").value);
    }
    ref.koseler = durum.ref.noktalar;
  } else {
    ref.kenar_mm = Number($("aruco-kenar").value);
    ref.koseler = durum.ref.noktalar;
    ref.etiket = durum.ref.etiket;
  }
  return {
    gorsel_id: durum.gorsel.kimlik,
    referans: ref,
    olcum: { tur: "kutu", noktalar: durum.nesne.koseler },
    sigma_px: Number($("sigma").value),
    guven: Number($("guven").value),
    mc_n: Number($("mc-n").value),
  };
}

async function olc() {
  if (durum.calisiyor) { zamanlayici = setTimeout(olc, 80); return; }
  durum.calisiyor = true;
  $("sonuc").classList.add("calisiyor");
  try {
    const s = await istek("/api/olc", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(istekGovdesi()),
    });
    durum.sonuc = s;
    durum.ref.etiket = s.referans.etiket;
    if (durum.ref.tur === "aruco" && s.referans.noktalar)
      durum.ref.noktalar = s.referans.noktalar;
    sonucuBas(s);
    bilgi(`${s.olculer[0].metin}  ·  ${s.sure_ms.toFixed(0)} ms`);
    $("ref-not").textContent = `${s.referans.etiket}` +
      (s.referans.piksel_boyu ? ` · görüntüde ${Math.round(s.referans.piksel_boyu)} px` : "");
    ciz();
  } catch (h) {
    durum.sonuc = null;
    hataGoster(h.message);
    bilgi("Ölçüm başarısız.");
  } finally {
    durum.calisiyor = false;
    $("sonuc").classList.remove("calisiyor");
  }
}

/* ================================================================ sonuç */
const sayi = (v, b = 1) => v.toLocaleString("tr-TR",
  { minimumFractionDigits: b, maximumFractionDigits: b });
const yuzde = (o, b = 1) => "%" + sayi(o * 100, b);

function el(etiket, sinif, icerik) {
  const d = document.createElement(etiket);
  if (sinif) d.className = sinif;
  if (typeof icerik === "string") d.textContent = icerik;
  else if (Array.isArray(icerik)) d.append(...icerik);
  return d;
}

function kartlariBosalt() {
  $("olcut-ad").textContent = "ÖLÇÜ";
  $("olcut-deger").textContent = "—";
  $("olcut-not").textContent = "ölçüm bekleniyor";
  $("pay-deger").textContent = "—";
  $("pay-not").textContent = "standart sapma";
  $("aralik-deger").textContent = "—";
  $("aralik-not").textContent = "kararı bu aralığın ucu verir";
  $("aralik-iz").hidden = true;
  $("ref-deger").textContent = "—";
  $("ref-not-kart").textContent = "henüz işaretlenmedi";
  $("ref-cipler").innerHTML = "";
  $("tuval-cipler").hidden = true;
}

function kartlariDoldur(s) {
  const o = s.olcum;
  $("olcut-ad").textContent = o.ad.toLocaleUpperCase("tr-TR");
  $("olcut-deger").innerHTML = "";
  $("olcut-deger").append(document.createTextNode(sayi(o.deger)),
                          el("small", "", o.birim));
  $("olcut-not").textContent = `${s.tur} · ${o.yontem}`;

  $("pay-deger").innerHTML = "";
  $("pay-deger").append(document.createTextNode(`± ${sayi(o.std)}`),
                        el("small", "", o.birim));
  $("pay-not").textContent = `bağıl ${yuzde(o.bagil_hata, 2)}`;

  $("aralik-deger").textContent = `${sayi(o.alt)} – ${sayi(o.ust)}`;
  $("aralik-not").textContent = `${yuzde(o.guven, 0)} güven · ${o.birim}`;
  // İşaretin yeri: nokta tahmininin aralık içindeki gerçek konumu.
  const genislik = o.ust - o.alt;
  $("aralik-iz").hidden = !(genislik > 0);
  if (genislik > 0) {
    const oran = Math.min(1, Math.max(0, (o.deger - o.alt) / genislik));
    $("aralik-imleci").style.left = `${(oran * 100).toFixed(1)}%`;
  }

  $("ref-deger").innerHTML = "";
  $("ref-deger").append(document.createTextNode(sayi(s.homografi.olcek_mm_px, 3)),
                        el("small", "", "mm/px"));
  $("ref-not-kart").textContent = s.referans.etiket;
  const cipler = $("ref-cipler");
  cipler.innerHTML = "";
  [s.homografi.model, `RMS ${sayi(s.homografi.rms_px, 2)} px`,
   `σ ${sayi(s.referans.sigma_px, 1)} px`]
    .forEach((metin) => cipler.append(el("span", "cip", metin)));

  $("tuval-cipler").hidden = false;
  $("cip-sigma").textContent = `σ ${sayi(s.referans.sigma_px, 1)} px`;
  $("cip-rms").textContent = `RMS ${sayi(s.homografi.rms_px, 2)} px`;
  $("cip-olcek").textContent = `${sayi(s.homografi.olcek_mm_px, 3)} mm/px`;
}

function sonucuTemizle() {
  kartlariBosalt();
  $("sonuc-govde").innerHTML = "";
  $("sonuc-govde").append(el("p", "sonuc-bos",
    !durum.gorsel ? "Önce bir fotoğraf yükle."
    : !refTamam() ? "Referansı işaretle."
    : "Nesnenin üstüne bir kutu çiz."));
}

function hataGoster(metin) {
  kartlariBosalt();
  $("sonuc-govde").innerHTML = "";
  $("sonuc-govde").append(el("div", "hata-kutu", metin));
}

function olcuBlogu(o, gercek) {
  const kutu = el("div", "olcu");
  kutu.append(el("div", "ad", o.ad));
  kutu.append(el("div", "deger", [document.createTextNode(sayi(o.deger)),
                                  el("small", "", o.birim)]));
  kutu.append(el("div", "pay", `± ${sayi(o.std)}`));
  kutu.append(el("div", "aralik", `${sayi(o.alt)} – ${sayi(o.ust)}`));
  if (gercek != null) {
    const icinde = gercek >= o.alt && gercek <= o.ust;
    kutu.append(el("div", "gercek" + (icinde ? "" : " disarida"),
      `gerçek ${sayi(gercek)} ${icinde ? "✓" : "✗"}`));
  }
  return kutu;
}

function sonucuBas(s) {
  kartlariDoldur(s);
  const govde = $("sonuc-govde");
  govde.innerHTML = "";
  const gercek = (ad) => (durum.demo && durum.demo.gercek && durum.demo.gercek[ad]
    ? durum.demo.gercek[ad].deger : null);

  const uzunluklar = s.olculer.filter((o) => o.birim === "mm");
  const cift = el("div", "olcu-cifti");
  uzunluklar.slice(0, 2).forEach((o) => cift.append(olcuBlogu(o, gercek(o.ad))));
  govde.append(cift);

  const alan = s.olculer.find((o) => o.ad === "alan");
  const ek = el("div", "ek-olcu");
  if (alan) {
    const g = gercek("alan");
    ek.append(el("div", "", [el("b", "", "alan "),
      document.createTextNode(`${sayi(alan.deger)} ± ${sayi(alan.std)} ${alan.birim}` +
        (g != null ? `  (gerçek ${sayi(g)})` : ""))]));
  }
  if (s.kutu) {
    const k = s.kutu.kenarlar_mm;
    ek.append(el("div", "", [el("b", "", "kenarlar "),
      document.createTextNode(k.map((v) => sayi(v)).join(" · ") + " mm")]));
    if (s.kutu.dikdortgenlik > 0.01) {
      ek.append(el("div", "", [el("b", "", "karşılıklı kenar farkı "),
        document.createTextNode(yuzde(s.kutu.dikdortgenlik))]));
    }
  }
  ek.append(el("div", "", [el("b", "", "referans "),
    document.createTextNode(`${s.referans.etiket} · ${s.homografi.model}` +
      ` · ${sayi(s.homografi.olcek_mm_px, 3)} mm/px`)]));
  govde.append(ek);

  if (s.uyarilar && s.uyarilar.length) {
    const liste = el("ul", "uyari-listesi kapali");
    s.uyarilar.forEach((u) => liste.append(el("li", u.seviye, u.metin)));
    govde.append(liste);
    if (s.uyarilar.length > 2) {
      const dugme = el("button", "uyari-ac", `${s.uyarilar.length - 2} not daha`);
      dugme.onclick = () => {
        const kapali = liste.classList.toggle("kapali");
        dugme.textContent = kapali ? `${s.uyarilar.length - 2} not daha` : "daha az göster";
      };
      govde.append(dugme);
    }
  }
}

/* ================================================================ başlangıç */
new ResizeObserver(tuvaliOlcekle).observe(sarmal);
tuvaliOlcekle();
arayuzuTazele();
belirsizlikKarti();
sonucuTemizle();

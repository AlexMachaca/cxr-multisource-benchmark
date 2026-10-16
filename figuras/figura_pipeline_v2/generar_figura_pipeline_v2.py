"""Figura 1 (v2): pipeline del estudio en 3 fases, en inglés y en español.

Versión 2 de figuras/figura_pipeline/generar_figura_pipeline.py (la v1 no se modifica). Cambios:
  - Paleta neutra (grises pizarra + tonos apagados aptos para daltonismo), fondo blanco, sin degradados.
  - Sin título dentro de la figura (va en la leyenda del artículo).
  - Radiografías reales de Kermany et al. (Mendeley Data, CC BY 4.0) en lugar de iconos dibujados;
    ver figuras/_recursos_cxr/ATRIBUCION.md. No se usan imágenes de SilicoData, VinDr-CXR ni Shenzhen/Montgomery.
  - Textos alineados con Materials and Methods: deduplicación por pHash, partición a nivel de imagen
    (solo SilicoData agrupada por paciente), leyenda para la variante principal.

Salidas (en esta carpeta): figura_pipeline_v2_{EN,ES}.{svg,pdf,png}; PNG a 600 dpi para 17.5 cm de ancho.
Uso: .venv/bin/python figuras/figura_pipeline_v2/generar_figura_pipeline_v2.py
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CXR = ROOT / "figuras" / "_recursos_cxr"
W, H = 1800, 1990

FONT = "'Liberation Sans', Arial, Helvetica, sans-serif"
# ---- paleta neutra ------------------------------------------------------------------------------
INK, INK2, MUTED, LINE, SOFT = "#1F2933", "#4B5563", "#8A94A3", "#D8DDE3", "#F4F6F8"
ACC, ACC_L, NEG = "#2E5A87", "#E6EDF4", "#9E3D32"
FAM = {"CNN": "#4A6A93", "Transformer": "#5E8F72", "Hybrid": "#B8873F", "Custom": "#B596B8"}
CLS = {"COVID-19": "#A65B4E", "Pneumonia": "#C29A4B", "Normal": "#6E9C7F", "Tuberculosis": "#3D5A80",
       "Silicosis": "#B3A6CC", "STB": "#4E868E"}
SPLITC = {"train": "#3D5A80", "val": "#8DA9C4", "test": "#6B7785"}

# ---------------------------------------------------------------------------------------------
# datos (mismas fuentes y verificaciones que la v1)
# ---------------------------------------------------------------------------------------------
MAN = ROOT / "dataset_final_224" / "manifiestos"
man = pd.read_csv(MAN / "manifiesto_6clases_con_exclusion.csv")
N6 = len(man)
SPLIT = man.split.value_counts().to_dict()
SRC_ORDER = ["COVID-QU-Ex", "Chest_x_ray", "RSNA_Pneumonia", "SilicoData", "TBX11K-simplified", "Shenzhen_Montgomery", "VinDr_CXR"]
SRC_NAME = {"COVID-QU-Ex": "COVID-QU-Ex", "Chest_x_ray": "Chest X-Ray", "RSNA_Pneumonia": "RSNA", "SilicoData": "SilicoData",
            "TBX11K-simplified": "TBX11K", "Shenzhen_Montgomery": "Shenzhen/Montgomery", "VinDr_CXR": "VinDr-CXR"}
CLS_ORDER = ["COVID-19", "Pneumonia", "Normal", "Tuberculosis", "Silicosis", "STB"]
FLOW = pd.crosstab(man.source_dataset, man.target_class).reindex(index=SRC_ORDER, columns=CLS_ORDER, fill_value=0)
CLS_N = FLOW.sum(0)
VARIANTS = {k: len(pd.read_csv(MAN / f"manifiesto_{k}.csv"))
            for k in ("6clases_sin_excluir", "6clases_con_exclusion", "5clases_sin_excluir", "5clases_con_exclusion")}
n5_test = int((pd.read_csv(MAN / "manifiesto_5clases_con_exclusion.csv").split == "test").sum())
assert VARIANTS == {"6clases_sin_excluir": 23105, "6clases_con_exclusion": 23027, "5clases_sin_excluir": 22349, "5clases_con_exclusion": 22271}
assert SPLIT == {"train": 16107, "val": 3463, "test": 3457} and n5_test == 3346
FUNNEL_N = [65007, 46856, 43971, 23105, 23027]
FUNNEL_D = [None, -18151, -2885, -20866, -78]
assert all(FUNNEL_N[i - 1] + FUNNEL_D[i] == FUNNEL_N[i] for i in range(1, 5))

fmt = lambda n: f"{int(n):,}"

# ---------------------------------------------------------------------------------------------
# textos en inglés y en español
# ---------------------------------------------------------------------------------------------
TXT = {
    "EN": dict(
        ph1="Data acquisition and curation", ph2="Model training and comparison", ph3="Evaluation and statistical analysis",
        a1="Multi-source acquisition", a1s="7 public sources → 6 diagnostic classes", a1n="n = {n} images after curation",
        cls={"COVID-19": "COVID-19", "Pneumonia": "Pneumonia", "Normal": "Normal", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        b1="Data curation", b1s="Images remaining after each step",
        fun=["Raw pool from 7 sources", "Cross-source duplicates removed (pHash)", "COVID-QU-Ex Non-COVID category excluded",
             "Cap of 6,000 per abundant class", "Manual quality exclusion"],
        c1="Partitioning and dataset variants", c1s="One held-out test set per variant",
        spl={"train": "Train", "val": "Val", "test": "Test"},
        split_note=["Image-level split, stratified by source;", "SilicoData images grouped by patient"],
        var="Four dataset variants", vcols=["6 classes", "5 classes (no STB)"], vrows=[["Without", "exclusion"], ["With quality", "exclusion"]],
        star="★ main variant", rs1="Resized to 224 × 224 px", rs2="(bilinear, three channels)",
        a2="Ten architectures, four families", a2s="ImageNet-pretrained, same training protocol",
        fam={"CNN": "CNN", "Transformer": "Transformer", "Hybrid": "Hybrid", "Custom": "Custom"},
        cust=["Custom: pretrained ResNet-18 stem +", "Transformer encoder trained from scratch"],
        b2="Two-stage transfer learning", b2s="Identical protocol for every architecture",
        st1="Stage 1 · head only", st1s="up to 8 epochs · lr 1e-3", st2="Stage 2 · partial fine-tuning", st2s="up to 12 epochs · lr 1e-5",
        bb="Backbone", frozen="frozen", first70="first 70% frozen", last=("last", "30%"), head="Head", trained="trained",
        bul=[["Class-balanced weighted sampling"], ["Augmentation: flip, ±15° rotation,", "0.9–1.1 scaling, ±10% contrast"],
             ["Early stopping · LR reduced on plateau;", "best checkpoint by validation loss"]],
        c2="Systematic comparison", c2s="10 architectures × 4 variants = 40 training runs",
        gh=["6 classes", "5 classes"], gsub=["all", "−78", "all", "−78"], c2n="Same test set within each variant",
        a3="Multi-metric evaluation", a3s="Held-out test set", cm=["Confusion matrix", "(schematic)"],
        mets=["Accuracy", "Macro-F1", "Per-class F1", "Per-class AUC"],
        tst=["Test images: {a} (6 classes)", "{b} (5 classes)"],
        b3="Statistical inference", b3s="Are the top models really different?",
        boot=("Paired bootstrap", ["2,000 resamples:", "95% CI of macro-F1"]),
        mcn=("McNemar test", ["Pairwise, top 5 models;", "Holm correction"]),
        wil=("Wilcoxon signed-rank test", ["Paired ΔF1 of quality", "exclusion, 10 models"]),
        c3="Targeted analyses", c3s="What drives the differences?",
        stb=("Effect of the STB class", ["6 vs. 5 classes: composition", "vs. residual effect"]),
        qua=("Quality exclusion", ["78 flagged images:", "with vs. without, 10 models"]),
        rnk=("Ranking stability", ["Model order compared", "across the 4 variants"]),
        foot="STB: silicotuberculosis · −78: images excluded for quality",
    ),
    "ES": dict(
        ph1="Adquisición y curación de datos", ph2="Entrenamiento y comparación de modelos", ph3="Evaluación y análisis estadístico",
        a1="Adquisición multifuente", a1s="7 fuentes públicas → 6 clases diagnósticas", a1n="n = {n} imágenes tras la curación",
        cls={"COVID-19": "COVID-19", "Pneumonia": "Neumonía", "Normal": "Normal", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        b1="Curación de datos", b1s="Imágenes restantes tras cada paso",
        fun=["Pool bruto de 7 fuentes", "Duplicados entre fuentes eliminados (pHash)", "Categoría Non-COVID de COVID-QU-Ex excluida",
             "Tope de 6,000 por clase abundante", "Exclusión manual por calidad"],
        c1="Partición y variantes del dataset", c1s="Un conjunto de prueba por variante",
        spl={"train": "Entren.", "val": "Val.", "test": "Prueba"},
        split_note=["Partición a nivel de imagen, estratificada", "por fuente; SilicoData agrupada por paciente"],
        var="Cuatro variantes del dataset", vcols=["6 clases", "5 clases (sin STB)"], vrows=[["Sin", "exclusión"], ["Con exclusión", "por calidad"]],
        star="★ variante principal", rs1="Redimensionadas a 224 × 224 px", rs2="(bilineal, tres canales)",
        a2="Diez arquitecturas, cuatro familias", a2s="Preentrenadas en ImageNet, mismo protocolo",
        fam={"CNN": "CNN", "Transformer": "Transformer", "Hybrid": "Híbrido", "Custom": "Propio"},
        cust=["Propio: base ResNet-18 preentrenada +", "encoder Transformer entrenado desde cero"],
        b2="Aprendizaje por transferencia", b2s="Dos fases, mismo protocolo para todas",
        st1="Fase 1 · solo la cabeza", st1s="hasta 8 épocas · lr 1e-3", st2="Fase 2 · ajuste fino parcial", st2s="hasta 12 épocas · lr 1e-5",
        bb="Backbone", frozen="congelado", first70="70% congelado", last=("30%", "final"), head="Cabeza", trained="entrenada",
        bul=[["Muestreo ponderado balanceado por clase"], ["Aumento: volteo, rotación ±15°,", "escala 0.9–1.1, contraste ±10%"],
             ["Parada temprana · LR reducida ante estancamiento;", "mejor checkpoint por pérdida de validación"]],
        c2="Comparación sistemática", c2s="10 arquitecturas × 4 variantes = 40 entrenamientos",
        gh=["6 clases", "5 clases"], gsub=["todas", "−78", "todas", "−78"], c2n="Mismo conjunto de prueba en cada variante",
        a3="Evaluación con varias métricas", a3s="Conjunto de prueba reservado", cm=["Matriz de confusión", "(esquemática)"],
        mets=["Exactitud", "Macro-F1", "F1 por clase", "AUC por clase"],
        tst=["Imágenes de prueba: {a} (6 clases)", "{b} (5 clases)"],
        b3="Inferencia estadística", b3s="¿Los mejores modelos difieren realmente?",
        boot=("Bootstrap pareado", ["2,000 remuestreos:", "IC 95% de la macro-F1"]),
        mcn=("Prueba de McNemar", ["Por pares, top 5 de modelos;", "corrección de Holm"]),
        wil=("Prueba de Wilcoxon", ["ΔF1 pareado de la exclusión", "por calidad, 10 modelos"]),
        c3="Análisis específicos", c3s="¿Qué explica las diferencias?",
        stb=("Efecto de la clase STB", ["6 vs. 5 clases: efecto de", "composición vs. residual"]),
        qua=("Exclusión por calidad", ["78 imágenes marcadas:", "con vs. sin, 10 modelos"]),
        rnk=("Estabilidad del ranking", ["Orden de los modelos", "en las 4 variantes"]),
        foot="STB: silicotuberculosis · −78: imágenes excluidas por calidad",
    ),
}


# ---------------------------------------------------------------------------------------------
# primitivas SVG
# ---------------------------------------------------------------------------------------------
def T(x, y, s, size=22, weight="normal", fill=INK, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def TL(x, y, lines, size=22, lh=1.25, **kw):
    return "".join(T(x, y + i * size * lh, ln, size=size, **kw) for i, ln in enumerate(lines))


def R(x, y, w, h, r=8, fill="#fff", stroke="none", sw=2, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def C(x, y, r, fill, stroke="none", sw=2):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def L(x1, y1, x2, y2, stroke=LINE, sw=2, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" {d}/>'


def P(d, fill="none", stroke="none", sw=2, extra=""):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'


def arrow(x1, y, x2, color=MUTED, sw=3, head=9):
    return L(x1, y, x2 - head, y, color, sw) + P(f"M{x2 - head - 1},{y - head * .8} L{x2},{y} L{x2 - head - 1},{y + head * .8} Z", fill=color)


def lerp(c1, c2, t):
    a, b = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)]), np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)])
    return "#%02x%02x%02x" % tuple((a + (b - a) * t).round().astype(int))


_IMG = {}


def IMG(x, y, s, name, r=6):
    """Radiografía real incrustada (base64) con esquinas redondeadas y borde fino."""
    if name not in _IMG:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.open(CXR / name).resize((256, 256), Image.Resampling.LANCZOS).save(buf, "PNG", optimize=True)
        _IMG[name] = base64.b64encode(buf.getvalue()).decode()
    cid = f"clip_{name.split('.')[0]}_{int(x)}_{int(y)}"
    return (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{s}" height="{s}" rx="{r}"/></clipPath>'
            f'<image x="{x}" y="{y}" width="{s}" height="{s}" clip-path="url(#{cid})" preserveAspectRatio="xMidYMid slice" '
            f'href="data:image/png;base64,{_IMG[name]}"/>' + R(x, y, s, s, r, "none", "#9AA3AE", 1.5))


def heading(x0, y0, title, sub):
    return T(x0 + 280, y0 + 30, title, 28, "bold", INK, "middle") + T(x0 + 280, y0 + 59, sub, 21, "normal", INK2, "middle")


# ---------------------------------------------------------------------------------------------
# iconos (neutros)
# ---------------------------------------------------------------------------------------------
def icon_lock(x, y, s=30, color=ACC):
    body = R(x, y + s * .42, s, s * .58, s * .12, color)
    sh = P(f"M{x + s * .22},{y + s * .44} V{y + s * .28} A{s * .28},{s * .28} 0 0 1 {x + s * .78},{y + s * .28} V{y + s * .44}", stroke=color, sw=s * .13)
    return sh + body + C(x + s / 2, y + s * .68, s * .09, "#fff")


def icon_forest(x, y, w=170, h=112):
    o = [L(x + w * .52, y, x + w * .52, y + h, MUTED, 2, "5 5")]
    rows = [(.40, .72, .56), (.34, .70, .52), (.42, .66, .54), (.25, .55, .40), (.60, .86, .73)]
    for i, (a, b, m) in enumerate(rows):
        yy = y + 12 + i * (h - 24) / 4
        col = ACC if i < 4 else MUTED
        o += [L(x + w * a, yy, x + w * b, yy, col, 4), C(x + w * m, yy, 7, col)]
    return "".join(o)


def icon_mcnemar(x, y, s=112):
    c, o = s / 2, []
    for i in range(2):
        for j in range(2):
            disc = (i != j)
            o.append(R(x + j * c, y + i * c, c - 4, c - 4, 6, ACC_L if disc else SOFT, ACC if disc else "#B9C2CC", 2.5))
            o.append(T(x + j * c + c / 2 - 2, y + i * c + c / 2 + 9, [["a", "b"], ["c", "d"]][i][j], 25, "bold", INK2, "middle"))
    return "".join(o)


def icon_paired(x, y, w=170, h=100):
    o = []
    for a, b in zip([.10, .26, .40, .54, .68, .82], [.28, .52, .46, .72, .80, .96]):
        o.append(L(x + 24, y + h * (1 - a), x + w - 24, y + h * (1 - b), "#B9C2CC", 3))
        o += [C(x + 24, y + h * (1 - a), 7, MUTED), C(x + w - 24, y + h * (1 - b), 7, ACC)]
    return "".join(o)


def icon_heat(x, y, n=6, cell=37):
    rng, o = np.random.default_rng(3), []
    for i in range(n):
        for j in range(n):
            v = rng.uniform(.80, .97) if i == j else (rng.uniform(.18, .38) if rng.random() < .22 else rng.uniform(.0, .08))
            o.append(R(x + j * cell, y + i * cell, cell - 3, cell - 3, 3, lerp("#F1F4F7", "#2E4A6B", v)))
    return "".join(o)


def icon_stack_bars(x, y):
    o, tot = [], CLS_N.sum()
    for lab, yy in (("6", y), ("5", y + 50)):
        xx = x
        for c in CLS_ORDER:
            if lab == "5" and c == "STB":
                continue
            wseg = 150 * CLS_N[c] / tot
            o.append(R(xx, yy, max(wseg - 2, 1), 32, 3, CLS[c]))
            xx += wseg
    return "".join(o) + T(x - 10, y + 24, "6", 21, "bold", INK2, "end") + T(x - 10, y + 74, "5", 21, "bold", INK2, "end")


def icon_quality(x, y):
    o = []
    for i in range(3):
        for j in range(5):
            bad = (i, j) in ((0, 3), (2, 1))
            o.append(R(x + j * 36, y + i * 32, 30, 26, 4, "#F3E3E0" if bad else SOFT, NEG if bad else "#C9D0D8", 2 if bad else 1.5))
            if bad:
                o.append(P(f"M{x + j * 36 + 8},{y + i * 32 + 6} l14,14 M{x + j * 36 + 22},{y + i * 32 + 6} l-14,14", stroke=NEG, sw=3))
    return "".join(o)


def icon_rank(x, y, w=170, h=80):
    o, orders = [], [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [0, 1, 3, 2, 4], [1, 0, 2, 3, 4]]
    cols = [FAM["CNN"], FAM["Transformer"], FAM["Hybrid"], FAM["Custom"], "#B9C2CC"]
    for m in range(5):
        pts = [(x + k * w / 3, y + orders[k].index(m) * h / 4) for k in range(4)]
        o.append(P("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts), stroke=cols[m], sw=3.5))
        o += [C(a, b, 5.5, cols[m]) for a, b in pts]
    return "".join(o)


# ---------------------------------------------------------------------------------------------
# paneles
# ---------------------------------------------------------------------------------------------
def p1a(x0, y0, t):
    o = [heading(x0, y0, t["a1"], t["a1s"])]
    top, K, GAP_L, GAP_R, MINH = y0 + 92, 0.0135, 9, 12, 20
    xl, xr = x0 + 208, x0 + 336
    hl = [max(MINH, FLOW.loc[s].sum() * K) for s in SRC_ORDER]
    hr = [max(MINH, CLS_N[c] * K) for c in CLS_ORDER]
    tot_l, tot_r = sum(hl) + GAP_L * 6, sum(hr) + GAP_R * 5
    yl, yr, cur = [], [], top
    for h in hl:
        yl.append(cur); cur += h + GAP_L
    cur = top + (tot_l - tot_r) / 2
    for h in hr:
        yr.append(cur); cur += h + GAP_R
    off_l = {s: yl[i] + (hl[i] - FLOW.loc[s].sum() * K) / 2 for i, s in enumerate(SRC_ORDER)}
    off_r = {c: yr[j] + (hr[j] - CLS_N[c] * K) / 2 for j, c in enumerate(CLS_ORDER)}
    xm = (xl + 14 + xr) / 2
    for s in SRC_ORDER:
        for c in CLS_ORDER:
            n = FLOW.loc[s, c]
            if n == 0:
                continue
            w = max(n * K, 1.6)
            a0, b0 = off_l[s], off_r[c]
            o.append(P(f"M{xl + 14},{a0:.1f} C{xm},{a0:.1f} {xm},{b0:.1f} {xr},{b0:.1f} L{xr},{b0 + w:.1f} C{xm},{b0 + w:.1f} {xm},{a0 + w:.1f} {xl + 14},{a0 + w:.1f} Z",
                       fill=CLS[c], extra='fill-opacity=".42"'))
            off_l[s] += n * K; off_r[c] += n * K
    for i, s in enumerate(SRC_ORDER):
        o.append(R(xl, yl[i], 14, hl[i], 3, "#5B6573"))
        o.append(T(xl - 10, yl[i] + hl[i] / 2 + 7, SRC_NAME[s], 20, "normal", INK, "end"))
    for j, c in enumerate(CLS_ORDER):
        o.append(R(xr, yr[j], 14, hr[j], 3, CLS[c]))
        o.append(f'<text x="{xr + 24}" y="{yr[j] + hr[j] / 2 + 7:.1f}" font-size="20" fill="{INK}"><tspan font-weight="bold">{escape(t["cls"][c])}</tspan>'
                 f'<tspan fill="{INK2}" dx="8">{fmt(CLS_N[c])}</tspan></text>')
    o.append(T(x0 + 280, y0 + 526, t["a1n"].format(n=fmt(N6)), 21, "normal", INK2, "middle"))
    return "".join(o)


def p1b(x0, y0, t):
    o = [heading(x0, y0, t["b1"], t["b1s"])]
    cx, top, rowh, wmax = x0 + 106, y0 + 92, 88, 184
    ws = [wmax * n / FUNNEL_N[0] for n in FUNNEL_N]
    for i, (n, label, delta) in enumerate(zip(FUNNEL_N, t["fun"], FUNNEL_D)):
        y = top + i * rowh
        w_top, w_bot = ws[i], (ws[i + 1] if i + 1 < len(ws) else ws[i])
        col = lerp("#A9B9CA", "#2E4A6B", i / 4)
        o.append(P(f"M{cx - w_top / 2},{y} L{cx + w_top / 2},{y} L{cx + w_bot / 2},{y + rowh - 6} L{cx - w_bot / 2},{y + rowh - 6} Z", fill=col))
        o.append(T(x0 + 222, y + 28, fmt(n), 27, "bold", INK))
        if delta:
            o.append(T(x0 + 548, y + 28, fmt(delta).replace("-", "−"), 23, "bold", NEG, "end"))
        words, lines, curl = label.split(" "), [], ""
        for wd in words:
            if len(curl) + len(wd) + 1 > 30:
                lines.append(curl); curl = wd
            else:
                curl = (curl + " " + wd).strip()
        lines.append(curl)
        o.append(TL(x0 + 222, y + 52, lines, 19, 1.18, fill=INK2))
    return "".join(o)


def p1c(x0, y0, t):
    o = [heading(x0, y0, t["c1"], t["c1s"])]
    bx, by, bw = x0 + 20, y0 + 122, 520
    cur = bx
    for key in ("train", "val", "test"):
        n = SPLIT[key]
        w = bw * n / N6
        light = key == "val"
        o.append(R(cur, by, w - 3, 52, 6, SPLITC[key]))
        o.append(T(cur + (w - 3) / 2, by + 33, fmt(n), 21 if w > 150 else 18, "bold", INK if light else "#fff", "middle"))
        o.append(T(cur + (w - 3) / 2, by - 32, t["spl"][key], 20, "bold", INK2, "middle") + T(cur + (w - 3) / 2, by - 10, f"{n / N6:.0%}", 20, "normal", INK2, "middle"))
        cur += w
    o.append(TL(x0 + 280, by + 86, t["split_note"], 19, 1.2, fill=INK2, anchor="middle"))
    gy = y0 + 290
    o.append(T(x0 + 280, gy - 14, t["var"], 23, "bold", INK, "middle"))
    gx, cw, ch = x0 + 168, 182, 56
    for j, h in enumerate(t["vcols"]):
        o.append(T(gx + j * (cw + 6) + cw / 2, gy + 14, h, 19, "bold", INK2, "middle"))
    rows = [("6clases_sin_excluir", "5clases_sin_excluir"), ("6clases_con_exclusion", "5clases_con_exclusion")]
    for i, keys in enumerate(rows):
        yy = gy + 28 + i * (ch + 6)
        o.append(TL(x0 + 158, yy + 26, t["vrows"][i], 18, 1.15, fill=INK2, anchor="end"))
        for j, k in enumerate(keys):
            primary = (k == "6clases_con_exclusion")
            xx = gx + j * (cw + 6)
            o.append(R(xx, yy, cw, ch, 8, ACC_L if primary else SOFT, ACC if primary else "#C9D0D8", 3 if primary else 1.5))
            o.append(T(xx + cw / 2, yy + 37, fmt(VARIANTS[k]), 26, "bold", INK, "middle"))
            if primary:
                o.append(T(xx + cw - 9, yy + 20, "★", 18, "bold", ACC, "end"))
    o.append(T(gx + 2 * cw + 6, gy + 28 + 2 * (ch + 6) + 12, t["star"], 17, "normal", ACC, "end"))
    ry = y0 + 480
    o.append(IMG(x0 + 40, ry - 6, 56, "cxr_normal_2.png", 5) + arrow(x0 + 102, ry + 22, x0 + 132, MUTED, 3, 8) + IMG(x0 + 138, ry + 8, 28, "cxr_normal_2.png", 3))
    o.append(T(x0 + 186, ry + 18, t["rs1"], 20, "bold", INK) + T(x0 + 186, ry + 43, t["rs2"], 19, "normal", INK2))
    return "".join(o)


def chip(x, y, w, h, text, col):
    return R(x, y, w, h, 8, lerp(col, "#FFFFFF", .86), col, 2) + T(x + w / 2, y + h / 2 + 8, text, 21, "bold", INK, "middle")


def p2a(x0, y0, t):
    o = [heading(x0, y0, t["a2"], t["a2s"])]
    left = [("CNN", ["VGG16-BN", "ResNet-50", "EfficientNet-B0", "MobileNetV3-Small"]), ("Custom", ["CNN–Transformer"])]
    right = [("Transformer", ["ViT-Base/16", "DeiT-Base", "Swin-Tiny"]), ("Hybrid", ["CoAtNet-0", "MaxViT-Tiny"])]
    for cx, groups in zip([x0 + 20, x0 + 290], (left, right)):
        y = y0 + 100
        for fam, names in groups:
            o.append(R(cx, y - 4, 16, 16, 3, FAM[fam]) + T(cx + 26, y + 11, f"{t['fam'][fam]} ({len(names)})", 22, "bold", INK))
            y += 30
            for nm in names:
                o.append(chip(cx, y, 250, 44, nm, FAM[fam]))
                y += 52
            y += 14
    o.append(TL(x0 + 280, y0 + 478, t["cust"], 19, 1.2, fill=INK2, anchor="middle"))
    return "".join(o)


def p2b(x0, y0, t):
    o = [heading(x0, y0, t["b2"], t["b2s"])]

    def row(y, title, sub, phase):
        r = [T(x0 + 20, y + 20, title, 22, "bold", INK) + T(x0 + 540, y + 20, sub, 19, "normal", INK2, "end")]
        by, bh = y + 34, 64
        r.append(IMG(x0 + 20, by, bh, "cxr_normal_1.png", 5))
        r.append(arrow(x0 + 86, by + bh / 2, x0 + 106))
        bx, bw = x0 + 108, 290
        if phase == 1:
            r.append(R(bx, by, bw, bh, 8, SOFT, ACC, 2.5) + icon_lock(bx + 14, by + 16, 30) + T(bx + 58, by + 28, t["bb"], 21, "bold", INK) + T(bx + 58, by + 50, t["frozen"], 19, "normal", INK2))
        else:
            wf = bw * .70
            r.append(R(bx, by, wf, bh, 8, SOFT, ACC, 2.5) + icon_lock(bx + 12, by + 16, 28) + T(bx + 50, by + 28, t["bb"], 20, "bold", INK) + T(bx + 50, by + 50, t["first70"], 17, "normal", INK2))
            xs, ws = bx + wf + 3, bw - wf - 3
            r.append(R(xs, by, ws, bh, 8, ACC_L, ACC, 2.5) + T(xs + ws / 2, by + 28, t["last"][0], 18, "bold", INK, "middle") + T(xs + ws / 2, by + 50, t["last"][1], 18, "bold", INK, "middle"))
        r.append(arrow(bx + bw + 2, by + bh / 2, bx + bw + 24))
        r.append(R(x0 + 436, by, 104, bh, 8, ACC_L, ACC, 2.5))
        r.append(T(x0 + 488, by + 28, t["head"], 20, "bold", INK, "middle") + T(x0 + 488, by + 50, t["trained"], 18, "normal", INK2, "middle"))
        return "".join(r)

    o.append(row(y0 + 92, t["st1"], t["st1s"], 1))
    o.append(row(y0 + 214, t["st2"], t["st2s"], 2))
    y = y0 + 352
    for lines in t["bul"]:
        o.append(C(x0 + 30, y - 6, 5, ACC) + TL(x0 + 46, y, list(lines), 20, 1.2, fill=INK2))
        y += 25 * len(lines) + 22
    return "".join(o)


MODELS = [("VGG16-BN", "CNN"), ("ResNet-50", "CNN"), ("EfficientNet-B0", "CNN"), ("MobileNetV3-S", "CNN"), ("ViT-Base/16", "Transformer"),
          ("DeiT-Base", "Transformer"), ("Swin-Tiny", "Transformer"), ("CoAtNet-0", "Hybrid"), ("MaxViT-Tiny", "Hybrid"), ("CNN–Transformer", "Custom")]


def p2c(x0, y0, t):
    o = [heading(x0, y0, t["c2"], t["c2s"])]
    gx, cw, rh = x0 + 196, 82, 34
    for h, j in zip(t["gh"], (0, 2)):
        o.append(T(gx + (j + 1) * cw - 4, y0 + 106, h, 20, "bold", INK2, "middle"))
        o.append(L(gx + j * cw + 8, y0 + 116, gx + (j + 2) * cw - 12, y0 + 116, "#B9C2CC", 2))
    for j, s in enumerate(t["gsub"]):
        o.append(T(gx + j * cw + cw / 2 - 4, y0 + 142, s, 19, "normal", INK2, "middle"))
    for i, (nm, fam) in enumerate(MODELS):
        y = y0 + 154 + i * rh
        o.append(T(gx - 14, y + 22, nm, 19, "normal", INK, "end"))
        for j in range(4):
            o.append(R(gx + j * cw, y + 3, cw - 8, rh - 8, 4, FAM[fam], extra='fill-opacity=".85"'))
    o.append(T(x0 + 280, y0 + 524, t["c2n"], 20, "normal", INK2, "middle"))
    return "".join(o)


def p3a(x0, y0, t):
    o = [heading(x0, y0, t["a3"], t["a3s"]), icon_heat(x0 + 26, y0 + 104)]
    o.append(T(x0 + 26 + 110, y0 + 358, t["cm"][0], 19, "bold", INK2, "middle") + T(x0 + 26 + 110, y0 + 380, t["cm"][1], 17, "normal", MUTED, "middle"))
    for i, m in enumerate(t["mets"]):
        o.append(R(x0 + 292, y0 + 110 + i * 60, 244, 48, 8, SOFT, "#B9C2CC", 2) + T(x0 + 414, y0 + 142 + i * 60, m, 21, "bold", INK, "middle"))
    o.append(R(x0 + 24, y0 + 420, 512, 92, 8, SOFT, LINE, 1.5))
    o.append(TL(x0 + 280, y0 + 452, [t["tst"][0].format(a=fmt(SPLIT["test"])), t["tst"][1].format(b=fmt(n5_test))], 21, 1.3, fill=INK2, anchor="middle", weight="bold"))
    return "".join(o)


def item(x0, y, icon, title, lines):
    return icon + T(x0 + 218, y + 34, title, 23, "bold", INK) + TL(x0 + 218, y + 64, lines, 20, 1.22, fill=INK2)


def p3b(x0, y0, t):
    o = [heading(x0, y0, t["b3"], t["b3s"])]
    o.append(item(x0, y0 + 100, icon_forest(x0 + 24, y0 + 104), t["boot"][0], t["boot"][1]))
    o.append(item(x0, y0 + 244, icon_mcnemar(x0 + 54, y0 + 250), t["mcn"][0], t["mcn"][1]))
    o.append(item(x0, y0 + 388, icon_paired(x0 + 24, y0 + 394), t["wil"][0], t["wil"][1]))
    return "".join(o)


def p3c(x0, y0, t):
    o = [heading(x0, y0, t["c3"], t["c3s"])]
    o.append(item(x0, y0 + 100, icon_stack_bars(x0 + 44, y0 + 124), t["stb"][0], t["stb"][1]))
    o.append(item(x0, y0 + 244, icon_quality(x0 + 22, y0 + 262), t["qua"][0], t["qua"][1]))
    o.append(item(x0, y0 + 388, icon_rank(x0 + 24, y0 + 400), t["rnk"][0], t["rnk"][1]))
    return "".join(o)


# ---------------------------------------------------------------------------------------------
# estructura global
# ---------------------------------------------------------------------------------------------
def phase_block(y, num, title, panels, t):
    o = [R(40, y + 54, 1720, 566, 12, "#FFFFFF", LINE, 2)]
    o.append(C(62, y + 22, 20, INK) + T(62, y + 31, str(num), 24, "bold", "#FFFFFF", "middle"))
    o.append(T(96, y + 32, title, 30, "bold", INK) + L(96 + len(title) * 16.2 + 28, y + 23, 1760, y + 23, LINE, 2))
    for i, fn in enumerate(panels):
        x0, y0 = 60 + i * 560, y + 72
        o.append(fn(x0, y0, t))
        if i:
            o.append(L(x0, y0 + 6, x0, y0 + 528, LINE, 2))
    return "".join(o)


def build(lang):
    t = TXT[lang]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']
    s.append(phase_block(20, 1, t["ph1"], [p1a, p1b, p1c], t))
    s.append(phase_block(660, 2, t["ph2"], [p2a, p2b, p2c], t))
    s.append(phase_block(1300, 3, t["ph3"], [p3a, p3b, p3c], t))
    fx = 60
    for fam in FAM:
        s.append(R(fx, H - 42, 20, 20, 3, FAM[fam]) + T(fx + 30, H - 25, t["fam"][fam], 20, "normal", INK2))
        fx += 50 + len(t["fam"][fam]) * 11 + 40
    s.append(T(1760, H - 25, t["foot"], 19, "normal", MUTED, "end"))
    s.append("</svg>")
    return "\n".join(s)


def render(lang):
    base = OUT / f"figura_pipeline_v2_{lang}"
    svg = base.with_suffix(".svg")
    svg.write_text(build(lang), encoding="utf-8")
    dpi = W / (17.5 / 2.54)                      # 1800 unidades = 17.5 cm (ancho de página de MDPI)
    subprocess.run(["rsvg-convert", "-f", "pdf", "--dpi-x", f"{dpi:.3f}", "--dpi-y", f"{dpi:.3f}", "-o", str(base.with_suffix(".pdf")), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-w", "4134", "-o", str(base.with_suffix(".png")), str(svg)], check=True)   # 17.5 cm a 600 dpi
    subprocess.run(["magick", "mogrify", "-units", "PixelsPerInch", "-density", "600", str(base.with_suffix(".png"))], check=True)


if __name__ == "__main__":
    for lang in ("EN", "ES"):
        render(lang)
    print("ok", W, H)

"""Figura 2 (v2): arquitectura del modelo propio CNN–Transformer, en inglés y en español.

Versión 2 de figuras/figura_arquitectura/ (la v1 no se modifica). Cambios:
  - Paleta neutra (la misma de la Figura 1 v2), fondo blanco, sin título interno.
  - Radiografía real de Kermany et al. (CC BY 4.0), ver figuras/_recursos_cxr/ATRIBUCION.md.
  - Se distingue qué partes vienen preentrenadas y cuáles se entrenan desde cero, y qué capas del
    backbone se ajustan en la fase 2 (layer2 y layer3: el 30% final de los tensores de parámetros).
  - Cifras calculadas reconstruyendo el modelo de notebooks_entrenamiento/entrenamiento_custom_variantes.ipynb.

Salidas: figura_arquitectura_v2_{EN,ES}.{svg,pdf,png}; PNG a 600 dpi para 17.5 cm de ancho.
Uso: .venv/bin/python figuras/figura_arquitectura_v2/generar_figura_arquitectura_v2.py
"""
from __future__ import annotations

import base64
import io
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import torch
import torch.nn as nn
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CXR = ROOT / "figuras" / "_recursos_cxr"
W, H = 1800, 500

FONT = "'Liberation Sans', Arial, Helvetica, sans-serif"
INK, INK2, MUTED, LINE, SOFT = "#1F2933", "#4B5563", "#8A94A3", "#D8DDE3", "#F4F6F8"
ACC, ACC_L = "#2E5A87", "#E6EDF4"
FROZEN, TUNED, NEW = "#C9D3DF", "#7F98B5", "#80709A"      # preentrenado congelado / ajustado en fase 2 / desde cero
NEW_L = "#ECE8F1"


# ---------------------------------------------------------------------------------------------
# cifras del modelo (misma definición que el notebook de entrenamiento)
# ---------------------------------------------------------------------------------------------
class _PosEmbed(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n, d))


class _Model(nn.Module):
    def __init__(self, num_classes, d=256):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)
        self.proj = nn.Conv2d(256, d, 1)
        self.embed = _PosEmbed(197, d)
        layer = nn.TransformerEncoderLayer(d, 8, d * 4, 0.1, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)


_m = _Model(6)
N_TOTAL = sum(p.numel() for p in _m.parameters())
N_STEM = sum(p.numel() for p in _m.backbone.parameters())
N_NEW = N_TOTAL - N_STEM
_named = list(_m.named_parameters())
_last = {".".join(n.split(".")[:2]) for n, _ in _named[-max(1, int(len(_named) * 0.3)):]}
assert N_TOTAL == 6_060_358 and _last == {"backbone.5", "backbone.6"}   # fase 2: layer2 y layer3
assert tuple(_m.backbone(torch.zeros(1, 3, 224, 224)).shape) == (1, 256, 14, 14)

TXT = {
    "EN": dict(
        inp="Input", inp2="224 × 224 × 3",
        stem="ResNet-18 stem", stem_n="Pretrained on ImageNet;", stem_n2="layer4 and original head removed",
        out="14 × 14 × 256",
        tok="Tokenization", proj="1 × 1 conv → 256", flat="196 patch tokens", cls="+ [CLS] token", pos="+ learned position", pos2="embedding",
        tokout="197 × 256",
        enc="Transformer encoder × 4", enc2="trained from scratch",
        mhsa="Multi-head", mhsa2="self-attention", mhsa3="8 heads", mlp="MLP", mlp2="256 → 1024 → 256", mlp3="GELU · dropout 0.1",
        ln="LayerNorm", res="residual",
        head1="LayerNorm", head1b="[CLS] token", head2="Linear", head2b="256 → C",
        classes=["COVID-19", "Normal", "Pneumonia", "Silicosis", "Tuberculosis", "STB*"],
        cnote="*Only in the 6-class variant (C = 6);", cnote2="C = 5 when STB is excluded",
        leg=["Pretrained, frozen in both stages", "Pretrained, fine-tuned in stage 2", "Trained from scratch (from stage 1)"],
        params="Parameters: {t} M (stem {s} M; new layers {n} M)",
    ),
    "ES": dict(
        inp="Entrada", inp2="224 × 224 × 3",
        stem="Base ResNet-18", stem_n="Preentrenada en ImageNet;", stem_n2="sin layer4 ni cabeza original",
        out="14 × 14 × 256",
        tok="Tokenización", proj="conv 1 × 1 → 256", flat="196 tokens de parches", cls="+ token [CLS]", pos="+ embedding posicional", pos2="aprendido",
        tokout="197 × 256",
        enc="Encoder Transformer × 4", enc2="entrenado desde cero",
        mhsa="Autoatención", mhsa2="multicabeza", mhsa3="8 cabezas", mlp="MLP", mlp2="256 → 1024 → 256", mlp3="GELU · dropout 0.1",
        ln="LayerNorm", res="residual",
        head1="LayerNorm", head1b="token [CLS]", head2="Lineal", head2b="256 → C",
        classes=["COVID-19", "Normal", "Neumonía", "Silicosis", "Tuberculosis", "STB*"],
        cnote="*Solo en la variante de 6 clases (C = 6);", cnote2="C = 5 cuando se excluye la STB",
        leg=["Preentrenado, congelado en ambas fases", "Preentrenado, ajustado en la fase 2", "Entrenado desde cero (desde la fase 1)"],
        params="Parámetros: {t} M (base {s} M; capas nuevas {n} M)",
    ),
}


def T(x, y, s, size=20, weight="normal", fill=INK, anchor="middle", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{escape(str(s))}</text>')


def R(x, y, w, h, r=8, fill="#fff", stroke="none", sw=2, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def L(x1, y1, x2, y2, stroke=MUTED, sw=2, dash=""):
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" {d}/>'


def P(d, fill="none", stroke="none", sw=2, extra=""):
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" {extra}/>'


def arrow(x1, y, x2, color=MUTED, sw=3, head=10):
    return L(x1, y, x2 - head, y, color, sw) + P(f"M{x2 - head - 1},{y - head * .75} L{x2},{y} L{x2 - head - 1},{y + head * .75} Z", fill=color)


def plus(x, y, r=14):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{INK2}" stroke-width="2"/>'
            + L(x - r * .55, y, x + r * .55, y, INK2, 2.5) + L(x, y - r * .55, x, y + r * .55, INK2, 2.5))


def IMG(x, y, s, name, r=6):
    buf = io.BytesIO()
    Image.open(CXR / name).resize((320, 320), Image.Resampling.LANCZOS).save(buf, "PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<clipPath id="ci"><rect x="{x}" y="{y}" width="{s}" height="{s}" rx="{r}"/></clipPath>'
            f'<image x="{x}" y="{y}" width="{s}" height="{s}" clip-path="url(#ci)" href="data:image/png;base64,{b64}"/>'
            + R(x, y, s, s, r, "none", "#9AA3AE", 1.5))


def build(lang):
    t = TXT[lang]
    cy = 232
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']

    # 1) entrada (20–170) -----------------------------------------------------------------------
    o.append(IMG(20, cy - 75, 150, "cxr_normal_1.png"))
    o.append(T(95, cy + 104, t["inp"], 20, "bold") + T(95, cy + 128, t["inp2"], 18, "normal", INK2))
    o.append(arrow(176, cy, 206))

    # 2) base ResNet-18 (210–510) ---------------------------------------------------------------
    x0, w0 = 210, 300
    o.append(R(x0, cy - 150, w0, 330, 10, "#FFFFFF", LINE, 2))
    o.append(T(x0 + w0 / 2, cy - 118, t["stem"], 21, "bold"))
    stages = [("conv1 + max-pool", "56 × 56 × 64", 1.00, FROZEN), ("layer1", "56 × 56 × 64", .80, FROZEN),
              ("layer2", "28 × 28 × 128", .62, TUNED), ("layer3", "14 × 14 × 256", .46, TUNED)]
    for i, (name, shape, frac, col) in enumerate(stages):
        bw = 168 * frac
        yy = cy - 94 + i * 50
        o.append(R(x0 + 16, yy, bw, 38, 5, col))
        o.append(T(x0 + 16 + bw / 2, yy + 25, name, 16, "bold", INK if col == FROZEN else "#FFFFFF"))
        o.append(T(x0 + w0 - 12, yy + 25, shape, 15, "normal", INK2, "end"))
    o.append(T(x0 + w0 / 2, cy + 132, t["stem_n"], 16, "normal", INK2) + T(x0 + w0 / 2, cy + 152, t["stem_n2"], 16, "normal", INK2))
    o.append(arrow(x0 + w0 + 6, cy, x0 + w0 + 36))

    # 3) tokenización (550–800) -----------------------------------------------------------------
    x1, w1 = 550, 250
    o.append(R(x1, cy - 150, w1, 330, 10, NEW_L, NEW, 2))
    o.append(T(x1 + w1 / 2, cy - 118, t["tok"], 21, "bold"))
    o.append(T(x1 + w1 / 2, cy - 92, t["proj"], 16, "normal", INK2))
    g, gs = 8, 11                                    # rejilla 14×14 representada con 8×8 celdas
    gx, gy = x1 + 24, cy - 72
    for i in range(g):
        for j in range(g):
            o.append(R(gx + j * gs, gy + i * gs, gs - 2, gs - 2, 1.5, "#B4A9C6"))
    o.append(T(gx + g * gs / 2, gy + g * gs + 20, "14 × 14", 15, "normal", INK2))
    o.append(arrow(gx + g * gs + 8, gy + g * gs / 2, gx + g * gs + 34, MUTED, 2.5, 8))
    tx = gx + g * gs + 40
    for k in range(8):                                # tira de tokens: [CLS] + parches
        o.append(R(tx, gy + k * 11, 32, 9, 2, NEW if k == 0 else "#B4A9C6"))
    o.append(T(tx + 16, gy + 104, "…", 18, "bold", INK2))
    o.append(T(tx + 38, gy + 9, "[CLS]", 13, "bold", NEW, "start"))
    o.append(T(x1 + w1 / 2, cy + 70, t["flat"], 15, "normal", INK2) + T(x1 + w1 / 2, cy + 89, t["cls"], 15, "normal", INK2))
    o.append(T(x1 + w1 / 2, cy + 108, t["pos"], 15, "normal", INK2) + T(x1 + w1 / 2, cy + 127, t["pos2"], 15, "normal", INK2))
    o.append(T(x1 + w1 / 2, cy + 158, t["tokout"], 17, "bold", INK))
    o.append(arrow(x1 + w1 + 6, cy, x1 + w1 + 36))

    # 4) encoder Transformer × 4 (840–1400) -----------------------------------------------------
    x2, w2 = 840, 560
    o.append(R(x2, cy - 150, w2, 330, 12, "none", NEW, 2.5, 'stroke-dasharray="9 7"'))
    o.append(R(x2 + w2 / 2 - 165, cy - 168, 330, 36, 18, "#FFFFFF", NEW, 2))
    o.append(T(x2 + w2 / 2, cy - 143, t["enc"], 19, "bold", INK))
    o.append(T(x2 + w2 / 2, cy + 160, t["enc2"], 16, "normal", INK2))

    def sub(x, title_lines, extra):
        bw = 132
        s = [R(x, cy - 30, 40, 60, 6, "#FFFFFF", NEW, 2), T(x + 20, cy + 6, "LN", 15, "bold", INK)]
        s.append(arrow(x + 42, cy, x + 56, MUTED, 2.5, 7))
        s.append(R(x + 58, cy - 60, bw, 120, 8, NEW_L, NEW, 2))
        for i, ln in enumerate(title_lines):
            s.append(T(x + 58 + bw / 2, cy - 26 + i * 21, ln, 16, "bold", INK))
        for i, ln in enumerate(extra):
            s.append(T(x + 58 + bw / 2, cy + 24 + i * 19, ln, 14, "normal", INK2))
        return "".join(s), x + 58 + bw

    xa = x2 + 24
    s1, e1 = sub(xa, [t["mhsa"], t["mhsa2"]], [t["mhsa3"]])
    p1 = e1 + 26
    o.append(s1 + arrow(e1 + 2, cy, p1 - 14, MUTED, 2.5, 7) + plus(p1, cy))
    o.append(P(f"M{xa - 10},{cy} V{cy - 100} H{p1} V{cy - 14}", stroke=MUTED, sw=2, extra='stroke-dasharray="5 5"'))
    xb = p1 + 30
    s2, e2 = sub(xb, [t["mlp"]], [t["mlp2"], t["mlp3"]])
    p2 = e2 + 26
    o.append(arrow(p1 + 14, cy, xb, MUTED, 2.5, 7))
    o.append(s2 + arrow(e2 + 2, cy, p2 - 14, MUTED, 2.5, 7) + plus(p2, cy))
    o.append(P(f"M{xb - 8},{cy} V{cy + 100} H{p2} V{cy + 14}", stroke=MUTED, sw=2, extra='stroke-dasharray="5 5"'))
    o.append(arrow(p2 + 14, cy, x2 + w2 + 36))

    # 5) cabeza de clasificación (1440–1790) --------------------------------------------------
    x3 = x2 + w2 + 40
    o.append(R(x3, cy - 46, 96, 92, 8, NEW_L, NEW, 2) + T(x3 + 48, cy - 6, t["head1"], 16, "bold") + T(x3 + 48, cy + 17, t["head1b"], 14, "normal", INK2))
    o.append(arrow(x3 + 100, cy, x3 + 124))
    o.append(R(x3 + 128, cy - 46, 84, 92, 8, NEW_L, NEW, 2) + T(x3 + 170, cy - 6, t["head2"], 16, "bold") + T(x3 + 170, cy + 17, t["head2b"], 14, "normal", INK2))
    o.append(arrow(x3 + 216, cy, x3 + 238))
    cx = x3 + 244
    for i, c in enumerate(t["classes"]):
        o.append(T(cx, cy - 72 + i * 30, c, 16, "normal", INK, "start"))
    o.append(T(x3, cy + 128, t["cnote"], 14, "normal", INK2, "start") + T(x3, cy + 146, t["cnote2"], 14, "normal", INK2, "start"))

    # leyenda --------------------------------------------------------------------------------------
    ly = H - 34
    o.append(L(20, ly - 30, W - 20, ly - 30, LINE, 1.5))
    lx = 24
    for col, txt in zip((FROZEN, TUNED, NEW_L), t["leg"]):
        o.append(R(lx, ly - 13, 26, 20, 4, col, NEW if col == NEW_L else "none", 2) + T(lx + 34, ly + 3, txt, 16, "normal", INK2, "start"))
        lx += 34 + len(txt) * 8.2 + 34
    fm = lambda n: f"{n / 1e6:.2f}"
    o.append(T(W - 24, ly + 3, t["params"].format(t=fm(N_TOTAL), s=fm(N_STEM), n=fm(N_NEW)), 16, "normal", INK2, "end"))
    o.append("</svg>")
    return "\n".join(o)


def render(lang):
    base = OUT / f"figura_arquitectura_v2_{lang}"
    svg = base.with_suffix(".svg")
    svg.write_text(build(lang), encoding="utf-8")
    dpi = W / (17.5 / 2.54)
    subprocess.run(["rsvg-convert", "-f", "pdf", "--dpi-x", f"{dpi:.3f}", "--dpi-y", f"{dpi:.3f}", "-o", str(base.with_suffix(".pdf")), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-w", "4134", "-o", str(base.with_suffix(".png")), str(svg)], check=True)
    subprocess.run(["magick", "mogrify", "-units", "PixelsPerInch", "-density", "600", str(base.with_suffix(".png"))], check=True)


if __name__ == "__main__":
    for lang in ("EN", "ES"):
        render(lang)
    print("ok", N_TOTAL, N_STEM, N_NEW)

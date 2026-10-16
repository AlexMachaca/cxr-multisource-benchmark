"""Figuras de resultados del artículo (Figuras 3–10), en inglés y en español, con la paleta neutra de la Figura 1 v2.

Lee únicamente lo que ya está calculado (no entrena ni evalúa nada):
  - resultados/comparacion_variantes/*.csv  (salida de notebooks_entrenamiento/analisis_4_variantes.ipynb)
  - resultados/resultados_entrenamiento/<variante>/<modelo>/*_predicciones.npz  (matrices de confusión)

La variante 6clases_con_exclusion se lee de la corrida B (resultados_entrenamiento/6_clases_con_exclusion_224). Si cambian los
resultados, primero re-ejecutar analisis_4_variantes.ipynb (regenera los CSV) y después este script.
Figuras 9 y 10: resultados/analisis_por_fuente/ (analisis_por_fuente.ipynb) y resultados/gradcam_clasificadores/ (notebooks_app/gradcam_clasificadores.py).

Salidas: figuras/figuras_resultados/figN_<nombre>_{EN,ES}.{pdf,png} (PNG a 600 dpi, 17.5 cm de ancho).
Uso: .venv/bin/python figuras/figuras_resultados/generar_figuras_resultados.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CV = ROOT / "resultados" / "comparacion_variantes"
RE = ROOT / "resultados" / "resultados_entrenamiento"

# ---- paleta neutra (la misma de las Figuras 1 y 2 v2) ------------------------------------------
INK, INK2, MUTED, LINE = "#1F2933", "#4B5563", "#8A94A3", "#D8DDE3"
ACC, NEG = "#2E5A87", "#9E3D32"
FAM_COL = {"CNN": "#4A6A93", "Transformer": "#5E8F72", "Hybrid": "#B8873F", "Custom": "#B596B8"}
CLS_COL = {"COVID-19": "#A65B4E", "Pneumonia": "#C29A4B", "Normal": "#6E9C7F", "Tuberculosis": "#3D5A80",
           "Silicosis": "#B3A6CC", "STB": "#4E868E"}
CMAP = LinearSegmentedColormap.from_list("neutro", ["#FFFFFF", "#C9D3DF", "#2E4A6B"])

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "grid.color": LINE, "grid.linewidth": 0.6, "savefig.dpi": 600, "pdf.fonttype": 42,
})
WIDTH = 17.5 / 2.54

# ---- nombres de modelos, familias y clases --------------------------------------------------------
NAME = {"VGG16": "VGG16-BN", "ResNet50": "ResNet-50", "EfficientNetB0": "EfficientNet-B0", "MobileNetV3": "MobileNetV3-Small",
        "ViT-Base": "ViT-Base/16", "DeiT-Base": "DeiT-Base", "Swin-Tiny": "Swin-Tiny", "CoAtNet": "CoAtNet-0",
        "MaxViT": "MaxViT-Tiny", "Custom": "CNN–Transformer"}
FAMILY = {"VGG16": "CNN", "ResNet50": "CNN", "EfficientNetB0": "CNN", "MobileNetV3": "CNN", "ViT-Base": "Transformer",
          "DeiT-Base": "Transformer", "Swin-Tiny": "Transformer", "CoAtNet": "Hybrid", "MaxViT": "Hybrid", "Custom": "Custom"}
FILE_PREFIX = {"vgg16_bn": "VGG16", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNetB0", "mobilenetv3_small_100": "MobileNetV3",
               "vit_base_patch16_224": "ViT-Base", "deit_base_patch16_224": "DeiT-Base", "swin_tiny_patch4_window7_224": "Swin-Tiny",
               "coatnet_0_rw_224": "CoAtNet", "maxvit_tiny_tf_224": "MaxViT", "custom_cnn_transformer": "Custom"}
VAR_DIR = {"6clases_con_exclusion": "6_clases_con_exclusion_224", "5clases_con_exclusion": "5_clases_con_exclusion",
           "6clases_sin_excluir": "6_clases_sin_exclusion", "5clases_sin_excluir": "5_clases_sin_exclusion"}
TRAIN_ORDER = {6: ["COVID-19", "Normal", "Pneumonia", "Silicosis", "STB", "Tuberculosis"],
               5: ["COVID-19", "Normal", "Pneumonia", "Silicosis", "Tuberculosis"]}
SHOW_ORDER = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Silicosis", "STB"]

TXT = {
    "EN": dict(
        fam={"CNN": "CNN", "Transformer": "Transformer", "Hybrid": "Hybrid", "Custom": "Custom"},
        cls={"Normal": "Normal", "Pneumonia": "Pneumonia", "COVID-19": "COVID-19", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        var={"6clases_con_exclusion": "Six classes, with exclusion", "5clases_con_exclusion": "Five classes, with exclusion",
             "6clases_sin_excluir": "Six classes, without exclusion", "5clases_sin_excluir": "Five classes, without exclusion"},
        varshort={"6clases_con_exclusion": "6 cl.\nwith excl.", "5clases_con_exclusion": "5 cl.\nwith excl.",
                  "6clases_sin_excluir": "6 cl.\nwithout excl.", "5clases_sin_excluir": "5 cl.\nwithout excl."},
        macro="Macro-F1", ci="Macro-F1 (95% bootstrap CI)", pbest="P(best)", mean="Family mean",
        true="True class", pred="Predicted class", rowpct="Row-normalized (%)",
        six="Six classes", five="Five classes (without STB)", perclass="Mean F1 across the 10 models",
        delta_q="ΔMacro-F1 (with − without exclusion)", params="Parameters (millions, log scale)",
        six_cls="6 classes", five_cls="5 classes",
        bacc="Balanced accuracy", glob="All\nsrc.", normal_rec="Recall of Normal", sil_normal="SilicoData Normal", oth_normal="Other sources' Normal",
        n_lbl="n = {n}", cl_lbl="{k} cl.", only="1 cl.",
        occ_acc="Accuracy (500-image subset)", occ_cond=["None", "Upper\ncorners\nhidden\n12.5%", "Centre\nhidden\n25%", "Frame\nhidden\n38%",
                                                               "Lung\nregion\nhidden\n43%", "Frame\nonly\nvisible\n62%"],
        chance="Chance (20%)", onlyframe="Recall by class with only the frame visible",
    ),
    "ES": dict(
        fam={"CNN": "CNN", "Transformer": "Transformer", "Hybrid": "Híbrido", "Custom": "Propio"},
        cls={"Normal": "Normal", "Pneumonia": "Neumonía", "COVID-19": "COVID-19", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        var={"6clases_con_exclusion": "Seis clases, con exclusión", "5clases_con_exclusion": "Cinco clases, con exclusión",
             "6clases_sin_excluir": "Seis clases, sin exclusión", "5clases_sin_excluir": "Cinco clases, sin exclusión"},
        varshort={"6clases_con_exclusion": "6 cl.\ncon excl.", "5clases_con_exclusion": "5 cl.\ncon excl.",
                  "6clases_sin_excluir": "6 cl.\nsin excl.", "5clases_sin_excluir": "5 cl.\nsin excl."},
        macro="Macro-F1", ci="Macro-F1 (IC 95% por bootstrap)", pbest="P(mejor)", mean="Media de la familia",
        true="Clase real", pred="Clase predicha", rowpct="Normalizada por fila (%)",
        six="Seis clases", five="Cinco clases (sin STB)", perclass="F1 medio de los 10 modelos",
        delta_q="ΔMacro-F1 (con − sin exclusión)", params="Parámetros (millones, escala log)",
        six_cls="6 clases", five_cls="5 clases",
        bacc="Exactitud balanceada", glob="Todas\nfuentes", normal_rec="Sensibilidad de Normal", sil_normal="Normal de SilicoData", oth_normal="Normal de otras fuentes",
        n_lbl="n = {n}", cl_lbl="{k} cl.", only="1 cl.",
        occ_acc="Exactitud (500 imágenes)", occ_cond=["Ninguna", "Esquinas\nsup. tapadas\n12.5%", "Centro\ntapado\n25%", "Marco\ntapado\n38%",
                                                                              "Región\npulmonar\ntapada 43%", "Solo\nmarco\nvisible 62%"],
        chance="Azar (20%)", onlyframe="Sensibilidad por clase viendo solo el marco",
    ),
}
VARS = ["6clases_con_exclusion", "5clases_con_exclusion", "6clases_sin_excluir", "5clases_sin_excluir"]

cons = pd.read_csv(CV / "consolidado_40_resultados.csv")
assert len(cons) == 40 and cons.groupby("variant_name").size().eq(10).all()


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")


def save(fig, name, lang):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}_{lang}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def family_legend(ax, t, **kw):
    hs = [plt.Line2D([], [], marker="o", ls="", color=FAM_COL[f], ms=5, label=t["fam"][f]) for f in FAM_COL]
    return ax.legend(handles=hs, frameon=False, **kw)


# ---------------------------------------------------------------------------------------------
# Figura 3: macro-F1 con IC 95% bootstrap (variantes principales: con exclusión)
# ---------------------------------------------------------------------------------------------
def fig3(lang):
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=False)
    for ax, v, lab in zip(axes, ["6clases_con_exclusion", "5clases_con_exclusion"], "ab"):
        b = pd.read_csv(CV / f"bootstrap_{v}.csv", index_col=0).sort_values("F1_punto")
        y = np.arange(len(b))
        for yi, (m, r) in zip(y, b.iterrows()):
            c = FAM_COL[FAMILY[m]]
            ax.plot([r.IC95_inf, r.IC95_sup], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(r.F1_punto, yi, "o", color=c, ms=5, zorder=3)
            ax.text(r.IC95_sup + 0.004, yi, f"{r.P_ser_el_mejor * 100:.1f}%" if r.P_ser_el_mejor >= 0.0005 else "", va="center", fontsize=7, color=INK2)
        ax.set_yticks(y, [NAME[m] for m in b.index])
        ax.set_xlabel(t["ci"]); ax.set_title(t["var"][v], loc="left", fontsize=9)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        lo, hi = b.IC95_inf.min(), b.IC95_sup.max()
        ax.set_xlim(lo - 0.01, hi + 0.045)
        ax.text(1.0, 1.01, t["pbest"], transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
        panel_label(ax, lab, x=-0.30)
    family_legend(axes[1], t, loc="lower right", ncol=1, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_bootstrap_macro_f1", lang)


# ---------------------------------------------------------------------------------------------
# Figura 4: macro-F1 por familia (cada modelo como punto + media de la familia), 4 variantes
# ---------------------------------------------------------------------------------------------
def fig4(lang):
    t = TXT[lang]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    fams = list(FAM_COL)
    rng = np.random.default_rng(0)
    for vi, v in enumerate(VARS):
        d = cons[cons.variant_name == v]
        for fi, f in enumerate(fams):
            x = vi + (fi - 1.5) * 0.19
            vals = d[d.label.map(FAMILY) == f].f1_macro.values
            ax.scatter(x + rng.uniform(-0.03, 0.03, len(vals)), vals, s=14, color=FAM_COL[f], alpha=0.85, lw=0, zorder=3)
            ax.plot([x - 0.07, x + 0.07], [vals.mean()] * 2, color=FAM_COL[f], lw=2.2, zorder=4)
    ax.set_xticks(range(4), [t["varshort"][v] for v in VARS])
    ax.set_ylabel(t["macro"]); ax.grid(axis="y"); ax.set_axisbelow(True)
    for vi in range(1, 4):
        ax.axvline(vi - 0.5, color=LINE, lw=0.8)
    hs = [plt.Line2D([], [], marker="o", ls="", color=FAM_COL[f], ms=4.5, label=t["fam"][f]) for f in fams]
    hs.append(plt.Line2D([], [], color=MUTED, lw=2.2, label=t["mean"]))
    ax.legend(handles=hs, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, "fig4_f1_por_familia", lang)


# ---------------------------------------------------------------------------------------------
# Figura 5: matrices de confusión normalizadas del mejor modelo de cada variante principal
# ---------------------------------------------------------------------------------------------
def best_model(v):
    return pd.read_csv(CV / f"bootstrap_{v}.csv", index_col=0).F1_punto.idxmax()


def confusion(v, model):
    pref = [k for k, m in FILE_PREFIX.items() if m == model][0]
    k = 6 if v.startswith("6") else 5
    f = next((RE / VAR_DIR[v]).glob(f"*/{pref}__{v}_predicciones.npz"), None)
    if f is not None:
        d = np.load(f)
        cm = np.zeros((k, k), int)
        np.add.at(cm, (d["y_true"], d["y_pred"]), 1)
    else:   # repositorio público: no se distribuyen las predicciones por imagen, solo las matrices de confusión (conteos)
        cm = pd.read_csv(ROOT / "resultados" / "matrices_confusion" / v / f"{pref}.csv", index_col=0).to_numpy()
    order = [TRAIN_ORDER[k].index(c) for c in SHOW_ORDER if c in TRAIN_ORDER[k]]
    cm = cm[np.ix_(order, order)]
    return cm, [c for c in SHOW_ORDER if c in TRAIN_ORDER[k]]


def fig5(lang):
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), gridspec_kw=dict(width_ratios=[6, 5]))
    for ax, v, lab in zip(axes, ["6clases_con_exclusion", "5clases_con_exclusion"], "ab"):
        m = best_model(v)
        cm, cls = confusion(v, m)
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", fontsize=7,
                        color="#FFFFFF" if pct[i, j] > 55 else INK)
        names = [t["cls"][c] for c in cls]
        ax.set_xticks(range(len(cls)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(cls)), names)
        ax.set_xlabel(t["pred"]); ax.set_ylabel(t["true"] if lab == "a" else "")
        ax.set_title(f"{t['var'][v]} — {NAME[m]}", loc="left", fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel_label(ax, lab, x=-0.08)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_matrices_confusion", lang)


# ---------------------------------------------------------------------------------------------
# Figura 6: efecto de la clase STB — (a) macro-F1 6 vs 5 clases por modelo; (b) F1 medio por clase
# ---------------------------------------------------------------------------------------------
def fig6(lang):
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    p = cons[cons.variant_name.isin(["6clases_con_exclusion", "5clases_con_exclusion"])].pivot(index="label", columns="variant_name", values="f1_macro")
    p = p.sort_values("5clases_con_exclusion")
    y = np.arange(len(p))
    for yi, (m, r) in zip(y, p.iterrows()):
        ax.plot([r["6clases_con_exclusion"], r["5clases_con_exclusion"]], [yi, yi], color=LINE, lw=2.2, zorder=1)
    ax.scatter(p["6clases_con_exclusion"], y, color=MUTED, s=20, zorder=3, label=t["six_cls"])
    ax.scatter(p["5clases_con_exclusion"], y, color=ACC, s=20, zorder=3, label=t["five_cls"])
    ax.set_yticks(y, [NAME[m] for m in p.index]); ax.set_xlabel(t["macro"])
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "a", x=-0.32)

    ax = axes[1]
    pc = pd.read_csv(CV / "f1_medio_por_clase_y_variante.csv", index_col=0).reindex(SHOW_ORDER)
    x = np.arange(len(pc))
    w = 0.38
    ax.bar(x - w / 2, pc["6clases_con_exclusion"], w, color=MUTED, label=t["six_cls"])
    ax.bar(x + w / 2, pc["5clases_con_exclusion"].fillna(0), w, color=ACC, label=t["five_cls"])
    ax.set_xticks(x, [t["cls"][c] for c in pc.index], rotation=35, ha="right")
    ax.set_ylabel(t["perclass"]); ax.set_ylim(0, 1)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.12))
    panel_label(ax, "b", x=-0.12)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig6_efecto_stb", lang)


# ---------------------------------------------------------------------------------------------
# Figura 7: efecto de la exclusión por calidad (Δ macro-F1 por modelo)
# ---------------------------------------------------------------------------------------------
def fig7(lang):
    t = TXT[lang]
    q = pd.read_csv(CV / "efecto_exclusion_calidad.csv", index_col=0)
    q = q.loc[q.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.9))
    y = np.arange(len(q))
    h = 0.36
    ax.barh(y + h / 2, q["6_clases"], h, color=MUTED, label=t["six_cls"])
    ax.barh(y - h / 2, q["5_clases"], h, color=ACC, label=t["five_cls"])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [NAME[m] for m in q.index]); ax.set_xlabel(t["delta_q"])
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig7_efecto_exclusion_calidad", lang)


# ---------------------------------------------------------------------------------------------
# Figura 8: macro-F1 frente al número de parámetros
# ---------------------------------------------------------------------------------------------
def fig8(lang):
    t = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=False)
    # Etiquetas: ("off", dx, dy, ha) desplazamiento en puntos; ("at", x, y, ha) posición en datos con línea guía
    # (para el grupo de modelos con macro-F1 y número de parámetros muy parecidos; ViT y DeiT tienen ambos 85.8 M)
    LAB = {
        "6clases_con_exclusion": {"CoAtNet": ("at", 10, .777, "right"), "ViT-Base": ("at", 10, .769, "right"),
                                  "Swin-Tiny": ("at", 10, .761, "right"), "DeiT-Base": ("at", 10, .753, "right"),
                                  "VGG16": ("at", 150, .744, "left"), "MaxViT": ("off", 7, -3, "left"),
                                  "MobileNetV3": ("off", 7, 2, "left"), "EfficientNetB0": ("off", 7, -9, "left"),
                                  "Custom": ("off", 7, 2, "left"), "ResNet50": ("off", 7, 2, "left")},
        "5clases_con_exclusion": {"ViT-Base": ("at", 12, .881, "right"), "DeiT-Base": ("at", 12, .873, "right"),
                                  "CoAtNet": ("at", 12, .865, "right"), "Swin-Tiny": ("at", 12, .857, "right"),
                                  "VGG16": ("at", 150, .841, "left"), "MaxViT": ("off", 7, -3, "left"),
                                  "MobileNetV3": ("off", 7, 2, "left"), "EfficientNetB0": ("off", 7, -9, "left"),
                                  "Custom": ("off", 7, 2, "left"), "ResNet50": ("off", 7, 2, "left")},
    }
    for ax, v, lab in zip(axes, ["6clases_con_exclusion", "5clases_con_exclusion"], "ab"):
        d = cons[cons.variant_name == v]
        for _, r in d.iterrows():
            c = FAM_COL[FAMILY[r.label]]
            x, y = r.total_params / 1e6, r.f1_macro
            ax.scatter(x, y, color=c, s=28, zorder=3, edgecolor="white", lw=0.5)
            kind, a, b, ha = LAB[v][r.label]
            if kind == "off":
                ax.annotate(NAME[r.label], (x, y), xytext=(a, b), textcoords="offset points", fontsize=6.8, color=INK2, ha=ha, va="center")
            else:
                ax.annotate(NAME[r.label], (x, y), xytext=(a, b), textcoords="data", fontsize=6.8, color=INK2, ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", color="#B9C2CC", lw=0.6, shrinkA=2, shrinkB=3))
        ax.set_xscale("log"); ax.set_xlabel(t["params"]); ax.set_ylabel(t["macro"])
        ax.set_title(t["var"][v], loc="left", fontsize=9)
        ax.grid(True, which="major"); ax.set_axisbelow(True)
        ax.set_xlim(0.9, 400)
        lo, hi = d.f1_macro.min(), d.f1_macro.max()
        ax.set_ylim(lo - 0.012, hi + 0.012)
        panel_label(ax, lab, x=-0.12)
    hs = [plt.Line2D([], [], marker="o", ls="", color=FAM_COL[f], ms=5, label=t["fam"][f]) for f in FAM_COL]
    fig.legend(handles=hs, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig8_f1_vs_parametros", lang)


# ---------------------------------------------------------------------------------------------
# Figura 9: evaluación por fuente (5 clases con exclusión): (a) exactitud balanceada por fuente; (b) recall de Normal en SilicoData vs otras fuentes
# ---------------------------------------------------------------------------------------------
SRC_ORDER = ["COVID-QU-Ex", "Chest_x_ray", "RSNA_Pneumonia", "TBX11K-simplified", "Shenzhen_Montgomery", "SilicoData", "VinDr_CXR"]
SRC_NICE = {"COVID-QU-Ex": "COVID-\nQU-Ex", "Chest_x_ray": "Chest\nX-Ray", "RSNA_Pneumonia": "RSNA", "TBX11K-simplified": "TBX11K",
            "Shenzhen_Montgomery": "Shenzhen/\nMontg.", "SilicoData": "Silico-\nData", "VinDr_CXR": "VinDr-\nCXR"}
PF = ROOT / "resultados" / "analisis_por_fuente"
OC = ROOT / "resultados" / "gradcam_clasificadores"


def fig9(lang):
    t = TXT[lang]
    pf = pd.read_csv(PF / "tabla_por_fuente.csv")
    pf = pf[(pf.variante == "5clases_con_exclusion") & (pf.corrida == "A")]
    M = pf.pivot(index="modelo", columns="fuente", values="bal_acc")[["GLOBAL"] + SRC_ORDER]
    M = M.loc[M.GLOBAL.sort_values(ascending=False).index]
    info = pf[pf.fuente != "GLOBAL"].groupby("fuente")[["n", "n_clases"]].first().reindex(SRC_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), gridspec_kw=dict(width_ratios=[2.0, 1]))
    ax = axes[0]
    ax.imshow(M.values, cmap=CMAP, vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color="#FFFFFF" if M.values[i, j] > 0.75 else INK)
    xl = [t["glob"]] + [f"{SRC_NICE[s]}\n" + (t["n_lbl"].format(n=int(info.loc[s, "n"])) + "\n" + (t["only"] if info.loc[s, "n_clases"] == 1 else t["cl_lbl"].format(k=int(info.loc[s, "n_clases"]))))
                        for s in SRC_ORDER]
    ax.set_xticks(range(M.shape[1]), xl, fontsize=6.2)
    ax.set_yticks(range(M.shape[0]), [NAME[m] for m in M.index])
    ax.tick_params(length=0, labeltop=True, labelbottom=False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(0.5, color="white", lw=2.5)
    ax.set_title(t["bacc"], loc="left", fontsize=8.5, pad=40)
    panel_label(ax, "a", x=-0.27, y=1.22)

    ax = axes[1]
    ns = pd.read_csv(PF / "normales_silicodata.csv", index_col=0).loc[M.index]
    y = np.arange(len(ns))[::-1]
    for yi, (m, r) in zip(y, ns.iterrows()):
        ax.plot([r.recall_Normal_SilicoData, r.recall_Normal_otras], [yi, yi], color=LINE, lw=2.2, zorder=1)
    ax.scatter(ns.recall_Normal_otras, y, color=ACC, s=22, zorder=3, label=t["oth_normal"])
    ax.scatter(ns.recall_Normal_SilicoData, y, color=NEG, s=22, zorder=3, label=t["sil_normal"])
    ax.set_yticks(y, [NAME[m] for m in ns.index]); ax.set_xlim(0, 1.0); ax.set_xlabel(t["normal_rec"])
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.20), ncol=1, fontsize=7)
    panel_label(ax, "b", x=-0.50, y=1.22)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig9_evaluacion_por_fuente", lang)


# ---------------------------------------------------------------------------------------------
# Figura 10: oclusión (5 modelos, 5 clases con exclusión): (a) exactitud por condición; (b) recall por clase viendo solo el marco
# ---------------------------------------------------------------------------------------------
def fig10(lang):
    t = TXT[lang]
    oc = pd.read_csv(OC / "tabla_oclusion.csv")
    cond = ["ninguna", "sin esquinas sup.", "sin centro 112²", "sin marco (24 px)", "sin región pulmonar", "SOLO marco (24 px)"]
    models = ["ViT-Base", "CoAtNet", "Swin-Tiny", "VGG16", "MobileNetV3"]
    mk = {"ViT-Base": "o", "CoAtNet": "s", "Swin-Tiny": "^", "VGG16": "D", "MobileNetV3": "v"}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.2), gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax = axes[0]
    for m in models:
        d = oc[oc.modelo == m].set_index("condicion").loc[cond]
        ax.plot(range(len(cond)), d.exactitud, marker=mk[m], color=FAM_COL[FAMILY[m]], lw=1.3, ms=4.5, label=NAME[m],
                ls="-" if m != "VGG16" else "--")
    ax.axhline(0.2, color=MUTED, lw=0.8, ls=":")
    ax.text(len(cond) - 0.7, 0.215, t["chance"], fontsize=6.8, color=INK2, ha="right")
    ax.set_xticks(range(len(cond)), t["occ_cond"], fontsize=6.4)
    ax.set_xlim(-0.35, len(cond) - 0.65)
    ax.set_ylabel(t["occ_acc"]); ax.set_ylim(0.1, 0.95)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6.6, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.1)
    panel_label(ax, "a", x=-0.10)

    ax = axes[1]
    pc = pd.read_csv(OC / "tabla_oclusion_por_clase.csv")
    pc = pc[pc.condicion == "SOLO marco (24 px)"].pivot(index="modelo", columns="clase", values="recall").loc[models]
    classes = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Silicosis"]
    pc = pc[classes]
    ax.imshow(pc.values, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(pc.shape[0]):
        for j in range(pc.shape[1]):
            ax.text(j, i, f"{pc.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#FFFFFF" if pc.values[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)), [t["cls"][c] for c in classes], rotation=35, ha="right")
    ax.set_yticks(range(len(models)), [NAME[m] for m in models])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(t["onlyframe"], loc="left", fontsize=7.8)
    panel_label(ax, "b", x=-0.40)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig10_oclusion", lang)


if __name__ == "__main__":
    for lang in ("EN", "ES"):
        for f in (fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10):
            f(lang)
    print("ok:", sorted(p.name for p in OUT.glob("*.png")))

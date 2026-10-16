"""Genera las Tablas 8-13 de la sección de Resultados (LaTeX), en inglés y en español, directamente de los CSV de análisis.

Así ninguna cifra de las tablas se transcribe a mano. Para cambiar un resultado: re-ejecutar el análisis (cuadernos de
`notebooks_entrenamiento/`) y luego este script.

Entradas (todas ya calculadas; no entrena ni evalúa nada):
  resultados/comparacion_variantes/{consolidado_40_resultados,bootstrap_*,mcnemar_top5_4_variantes,f1_y_puestos_por_variante,
                                    descomposicion_quitar_stb_con_exclusion}.csv
  resultados/resultados_entrenamiento/<variante>/<modelo>/*_metricas_por_clase.csv
  resultados/comparacion_replicas/{replica_A_vs_B,ruido_entrenamiento}.csv
  resultados/costo_computacional/latencia.csv   (notebooks_entrenamiento/benchmark_latencia.py)
Salidas: Redaccion/manuscrito/secciones/tablas/tabN.tex y secciones_ES/tablas/tabN.tex
Uso: .venv/bin/python Redaccion/manuscrito/generar_tablas_resultados.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CV = ROOT / "resultados" / "comparacion_variantes"
RE = ROOT / "resultados" / "resultados_entrenamiento"
REP = ROOT / "resultados" / "comparacion_replicas"
COST = ROOT / "resultados" / "costo_computacional"

V6, V5, V6S, V5S = "6clases_con_exclusion", "5clases_con_exclusion", "6clases_sin_excluir", "5clases_sin_excluir"
VDIR = {V6: "6_clases_con_exclusion_224", V5: "5_clases_con_exclusion", V6S: "6_clases_sin_exclusion", V5S: "5_clases_sin_exclusion"}
NAME = {"VGG16": "VGG16-BN", "ResNet50": "ResNet50", "EfficientNetB0": "EfficientNet-B0", "MobileNetV3": "MobileNetV3-Small",
        "ViT-Base": "ViT-Base/16", "DeiT-Base": "DeiT-Base", "Swin-Tiny": "Swin-Tiny", "CoAtNet": "CoAtNet-0",
        "MaxViT": "MaxViT-Tiny", "Custom": "CNN--Transformer propio"}
NAME_EN = dict(NAME, Custom="Custom CNN--Transformer")
PREFIX = {"vgg16_bn": "VGG16", "resnet50": "ResNet50", "efficientnet_b0": "EfficientNetB0", "mobilenetv3_small_100": "MobileNetV3",
          "vit_base_patch16_224": "ViT-Base", "deit_base_patch16_224": "DeiT-Base", "swin_tiny_patch4_window7_224": "Swin-Tiny",
          "coatnet_0_rw_224": "CoAtNet", "maxvit_tiny_tf_224": "MaxViT", "custom_cnn_transformer": "Custom"}
CLASS_ORDER = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Silicosis", "STB"]
FAM = {"VGG16": "CNN", "ResNet50": "CNN", "EfficientNetB0": "CNN", "MobileNetV3": "CNN", "ViT-Base": "T", "DeiT-Base": "T",
       "Swin-Tiny": "T", "CoAtNet": "H", "MaxViT": "H", "Custom": "C"}

L = {
    "EN": dict(
        name=NAME_EN, cls={"Normal": "Normal", "Pneumonia": "Pneumonia", "COVID-19": "COVID-19", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        model="Model", mean="Mean", rng="range", pbest="P(best) (\\%)",
        six="Six classes, with exclusion", five="Five classes, with exclusion",
        six_s="Six classes, without exclusion", five_s="Five classes, without exclusion",
    ),
    "ES": dict(
        name=NAME, cls={"Normal": "Normal", "Pneumonia": "Neumonía", "COVID-19": "COVID-19", "Tuberculosis": "Tuberculosis", "Silicosis": "Silicosis", "STB": "STB"},
        model="Modelo", mean="Media", rng="rango", pbest="P(mejor) (\\%)",
        six="Seis clases, con exclusión", five="Cinco clases, con exclusión",
        six_s="Seis clases, sin exclusión", five_s="Cinco clases, sin exclusión",
    ),
}


def f3(x):
    return f"{x:.3f}"


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def write(lang, n, body):
    d = HERE / ("secciones" if lang == "EN" else "secciones_ES") / "tablas"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"tab{n}.tex").write_text(body, encoding="utf-8")


def wrap(caption, label, colspec, header, rows, note=None, wide=True, size="\\small"):
    out = ["\\begin{table}[H]", f"\\caption{{{caption}\\label{{{label}}}}}"]
    if wide:
        out.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    out += [size, "\\newcolumntype{C}{>{\\centering\\arraybackslash}X}", "\\newcolumntype{L}{>{\\raggedright\\arraybackslash}X}", f"\\begin{{tabularx}}{{{'\\fulllength' if wide else '\\textwidth'}}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    out += rows
    out += ["\\bottomrule", "\\end{tabularx}"]
    if note:
        out.append(f"\\noindent{{\\footnotesize{{{note}}}}}")
    if wide:
        out.append("\\end{adjustwidth}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


cons = pd.read_csv(CV / "consolidado_40_resultados.csv")


def boot(v):
    return pd.read_csv(CV / f"bootstrap_{v}.csv", index_col=0)


def per_class_all(v):
    frames = []
    for f in (RE / VDIR[v]).glob("*/*_metricas_por_clase.csv"):
        d = pd.read_csv(f)
        d["label"] = PREFIX[f.name.split("__")[0]]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------------------------ Tabla 8
def tab8(lang):
    t = L[lang]
    panels = [(V6, t["six"], "a"), (V5, t["five"], "b")]
    rows = []
    for v, title, lab in panels:
        c = cons[cons.variant_name == v].set_index("label")
        b = boot(v)
        n = int(c.n_test_images.iloc[0])
        order = b.sort_values("F1_punto", ascending=False).index
        cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_roc_macro"]
        best = {k: c[k].max() for k in cols}
        n_txt = f"$n = {n:,}$".replace(",", "{,}")
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{({lab}) {title} ({n_txt})}}}}\\\\")
        for m in order:
            r = c.loc[m]
            ci = f"{f3(r.f1_macro)} ({f3(b.loc[m, 'IC95_inf'])}--{f3(b.loc[m, 'IC95_sup'])})"
            ci = bold(ci, r.f1_macro == best["f1_macro"])
            cells = [t["name"][m]] + [bold(f3(r[k]), r[k] == best[k]) for k in ("accuracy", "precision_macro", "recall_macro")] + [ci] + \
                    [bold(f3(r["auc_roc_macro"]), r["auc_roc_macro"] == best["auc_roc_macro"])] + [f"{b.loc[m, 'P_ser_el_mejor'] * 100:.1f}"]
            rows.append(" & ".join(cells) + "\\\\")
        if lab == "a":
            rows.append("\\midrule")
    if lang == "EN":
        cap = ("Overall performance of the ten architectures on the held-out test set of the two main dataset variants (with quality-based exclusion). "
               "Macro-averaged precision, recall and one-vs-rest AUC; macro-F1 is shown with its 95\\% paired-bootstrap confidence interval (2000 resamples), "
               "and P(best) is the proportion of resamples in which the model ranked first. Rows are sorted by macro-F1; the best value in each column is in bold. "
               "Each value comes from a single training run.")
        hdr = "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro-precision} & \\textbf{Macro-recall} & \\textbf{Macro-F1 (95\\% CI)} & \\textbf{Macro-AUC} & \\textbf{P(best) (\\%)}\\\\"
    else:
        cap = ("Desempeño global de las diez arquitecturas en el conjunto de prueba reservado de las dos variantes principales del dataset (con exclusión por calidad). "
               "Precisión, sensibilidad y AUC uno-contra-todos promedio macro; la macro-F1 se muestra con su intervalo de confianza al 95\\% por bootstrap pareado "
               "(2000 remuestreos) y P(mejor) es la proporción de remuestreos en que el modelo quedó primero. Las filas están ordenadas por macro-F1; el mejor valor de "
               "cada columna está en negrita. Cada valor proviene de un solo entrenamiento.")
        hdr = "\\textbf{Modelo} & \\textbf{Exactitud} & \\textbf{Precisión macro} & \\textbf{Sensibilidad macro} & \\textbf{Macro-F1 (IC 95\\%)} & \\textbf{AUC macro} & \\textbf{P(mejor) (\\%)}\\\\"
    write(lang, 8, wrap(cap, "tab:overall", "lCCCCCC", hdr, rows))


# ------------------------------------------------------------------------------------------------ Tabla 9
def tab9(lang):
    m = pd.read_csv(CV / "mcnemar_top5_4_variantes.csv")
    t = L[lang]
    rows = []
    for v, title in [(V6, t["six"]), (V5, t["five"]), (V6S, t["six_s"]), (V5S, t["five_s"])]:
        d = m[m.variante == v]
        top = list(dict.fromkeys(list(d.modelo_A) + list(d.modelo_B)))
        sig = d[d.significativo_Holm]
        sig_txt = "--" if sig.empty else "; ".join(f"{t['name'][a].split('/')[0]} vs. {t['name'][b].split('/')[0]}" for a, b in zip(sig.modelo_A, sig.modelo_B))
        pmin = d.p_Holm.min()
        pmin_txt = "$<0.001$" if pmin < 0.001 else f"{pmin:.3f}"
        rows.append(" & ".join([title, ", ".join(t["name"][x].split("/")[0].replace("-Base", "").replace("-Tiny", "") for x in top), str(len(d)),
                                str(int(d.significativo_sin_corregir.sum())), str(len(sig)), pmin_txt, sig_txt]) + "\\\\")
    if lang == "EN":
        cap = ("Pairwise comparisons among the five best models of each variant (McNemar's test with continuity correction, 10 pairs per variant, Holm correction). "
               "``Uncorrected'' counts the pairs with $p < 0.05$ before the correction.")
        hdr = "\\textbf{Variant} & \\textbf{Five best models} & \\textbf{Pairs} & \\textbf{Uncorrected $p<0.05$} & \\textbf{Significant after Holm} & \\textbf{Smallest adjusted $p$} & \\textbf{Significant pairs}\\\\"
    else:
        cap = ("Comparaciones por pares entre los cinco mejores modelos de cada variante (prueba de McNemar con corrección de continuidad, 10 pares por variante, "
               "corrección de Holm). ``Sin corregir'' cuenta los pares con $p < 0.05$ antes de la corrección.")
        hdr = "\\textbf{Variante} & \\textbf{Cinco mejores modelos} & \\textbf{Pares} & \\textbf{$p<0.05$ sin corregir} & \\textbf{Significativos tras Holm} & \\textbf{Menor $p$ ajustado} & \\textbf{Pares significativos}\\\\"
    write(lang, 9, wrap(cap, "tab:mcnemar", "lLCCCCL", hdr, rows, size="\\footnotesize"))


# ------------------------------------------------------------------------------------------------ Tabla 10
def tab10(lang):
    t = L[lang]
    rows = []
    blocks = []
    for v in (V6, V5):
        pc = per_class_all(v)
        best = boot(v).F1_punto.idxmax()
        blocks.append((v, pc, best))
    classes = [c for c in CLASS_ORDER]
    for c in classes:
        cells = [t["cls"][c]]
        for v, pc, best in blocks:
            d = pc[pc["class"] == c]
            if d.empty:
                cells += ["--"] * 6
                continue
            b = d[d.label == best].iloc[0]
            cells += [str(int(b.support)), f3(b.precision), f3(b.recall), f3(b.f1), f3(d.f1.mean()), f"\\mbox{{{d.f1.min():.2f}--{d.f1.max():.2f}}}"]
        rows.append(" & ".join(cells) + "\\\\")
    b6, b5 = blocks[0][2], blocks[1][2]
    if lang == "EN":
        cap = (f"Per-class performance on the test set. For each main variant, precision, recall and F1 of the best model ({L[lang]['name'][b6]} for six classes, "
               f"{L[lang]['name'][b5]} for five classes) and the mean F1 (minimum--maximum) across the ten models. $n$: test images of the class. Dashes: class not present in the variant.")
        top = "\\multirow{2}{*}{\\textbf{Class}} & \\multicolumn{6}{c}{\\textbf{Six classes, with exclusion}} & \\multicolumn{6}{c}{\\textbf{Five classes, with exclusion}}\\\\"
        sub = (f"\\cmidrule(lr){{2-7}}\\cmidrule(lr){{8-13}} & $n$ & Precision & Recall & F1 & Mean F1 & Range & $n$ & Precision & Recall & F1 & Mean F1 & Range\\\\")
        sub = sub.replace("& $n$", "& \\textbf{$n$}", 1)
    else:
        cap = (f"Desempeño por clase en el conjunto de prueba. Para cada variante principal se muestran la precisión, la sensibilidad y el F1 del mejor modelo "
               f"({L[lang]['name'][b6]} con seis clases, {L[lang]['name'][b5]} con cinco) y el F1 medio (mínimo--máximo) de los diez modelos. $n$: imágenes de prueba de la clase. "
               f"Guiones: clase ausente en la variante.")
        top = "\\multirow{2}{*}{\\textbf{Clase}} & \\multicolumn{6}{c}{\\textbf{Seis clases, con exclusión}} & \\multicolumn{6}{c}{\\textbf{Cinco clases, con exclusión}}\\\\"
        sub = ("\\cmidrule(lr){2-7}\\cmidrule(lr){8-13} & $n$ & Precisión & Sensib. & F1 & F1 medio & Rango & $n$ & Precisión & Sensib. & F1 & F1 medio & Rango\\\\")
    hdr = top + "\n" + sub
    write(lang, 10, wrap(cap, "tab:perclass", "lCCCCCCCCCCCC", hdr, rows, size="\\scriptsize"))


# ------------------------------------------------------------------------------------------------ Tabla 11
def tab11(lang):
    t = L[lang]
    d = pd.read_csv(CV / "descomposicion_quitar_stb_con_exclusion.csv")
    d["total"] = d.F1_macro_5 - d.F1_macro_6
    p6, p5 = per_class_all(V6), per_class_all(V5)
    sil6 = p6[p6["class"] == "Silicosis"].set_index("label").f1
    sil5 = p5[p5["class"] == "Silicosis"].set_index("label").f1
    d = d.sort_values("F1_macro_5", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([t["name"][r.label], f3(r.F1_macro_6), f3(r.F1_macro_5), f"+{r.total:.3f}", f"+{r.efecto_composicion:.3f}", f"+{r.efecto_real:.3f}",
                                f3(sil6[r.label]), f3(sil5[r.label])]) + "\\\\")
    rows.append("\\midrule")
    rows.append(" & ".join([f"\\textbf{{{t['mean']}}}", f3(d.F1_macro_6.mean()), f3(d.F1_macro_5.mean()), f"+{d.total.mean():.3f}", f"+{d.efecto_composicion.mean():.3f}",
                            f"+{d.efecto_real.mean():.3f}", f3(sil6.mean()), f3(sil5.mean())]) + "\\\\")
    if lang == "EN":
        cap = ("Effect of modelling silicotuberculosis (STB) as a separate class (variants with quality-based exclusion). The change in macro-F1 from six to five classes "
               "is split into a composition effect (removing the worst-performing class from the average) and a residual effect (change in the five shared classes). "
               "The last two columns give the F1 of the Silicosis class.")
        hdr = ("\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{Macro-F1}} & \\multicolumn{3}{c}{\\textbf{Change when STB is removed}} & \\multicolumn{2}{c}{\\textbf{F1 of Silicosis}}\\\\\n"
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8} & 6 classes & 5 classes & Total & Composition & Residual & 6 classes & 5 classes\\\\")
    else:
        cap = ("Efecto de modelar la silicotuberculosis (STB) como clase aparte (variantes con exclusión por calidad). El cambio de macro-F1 de seis a cinco clases se "
               "descompone en un efecto de composición (quitar del promedio la clase de peor desempeño) y un efecto residual (cambio en las cinco clases compartidas). "
               "Las dos últimas columnas dan el F1 de la clase Silicosis.")
        hdr = ("\\multirow{2}{*}{\\textbf{Modelo}} & \\multicolumn{2}{c}{\\textbf{Macro-F1}} & \\multicolumn{3}{c}{\\textbf{Cambio al quitar la STB}} & \\multicolumn{2}{c}{\\textbf{F1 de Silicosis}}\\\\\n"
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8} & 6 clases & 5 clases & Total & Composición & Residual & 6 clases & 5 clases\\\\")
    write(lang, 11, wrap(cap, "tab:stb", "lCCCCCCC", hdr, rows, size="\\small"))


# ------------------------------------------------------------------------------------------------ Tabla 12
def tab12(lang):
    t = L[lang]
    f = pd.read_csv(CV / "f1_y_puestos_por_variante.csv").set_index("label")
    r = pd.read_csv(REP / "replica_A_vs_B.csv").set_index("modelo")
    f = f.sort_values("puesto_medio")
    rows = []
    for m, x in f.iterrows():
        pu = [int(x["puesto_" + v]) for v in (V6, V5, V6S, V5S)]
        rows.append(" & ".join([t["name"][m]] + [str(p) for p in pu] + [f"{x.puesto_medio:.2f}", str(max(pu) - min(pu)),
                                f3(r.loc[m, "F1_A"]), f3(r.loc[m, "F1_B"]), f"{r.loc[m, 'F1_B'] - r.loc[m, 'F1_A']:+.3f}"]) + "\\\\")
    if lang == "EN":
        cap = ("Ranking stability across the four dataset variants (rank by macro-F1; 1 = best) and run-to-run variability. The last three columns compare two independent training runs "
               "of the six-class variant with exclusion: run A used full-resolution images resized during training and run B (the one reported in the rest of the paper) used the pre-resized images.")
        hdr = ("\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c}{\\textbf{Rank in variant}} & \\multirow{2}{*}{\\textbf{Mean rank}} & \\multirow{2}{*}{\\textbf{Range}} & \\multicolumn{3}{c}{\\textbf{Macro-F1, six classes with exclusion}}\\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(lr){8-10} & 6 cl., excl. & 5 cl., excl. & 6 cl., no excl. & 5 cl., no excl. & & & Run A & Run B & B $-$ A\\\\")
    else:
        cap = ("Estabilidad del ranking entre las cuatro variantes del dataset (puesto por macro-F1; 1 = mejor) y variabilidad entre corridas. Las tres últimas columnas comparan dos "
               "entrenamientos independientes de la variante de seis clases con exclusión: la corrida A usó imágenes de resolución completa redimensionadas durante el entrenamiento y la "
               "corrida B (la que se reporta en el resto del artículo) usó las imágenes ya redimensionadas.")
        hdr = ("\\multirow{2}{*}{\\textbf{Modelo}} & \\multicolumn{4}{c}{\\textbf{Puesto en la variante}} & \\multirow{2}{*}{\\textbf{Puesto medio}} & \\multirow{2}{*}{\\textbf{Rango}} & \\multicolumn{3}{c}{\\textbf{Macro-F1, seis clases con exclusión}}\\\\\n"
               "\\cmidrule(lr){2-5}\\cmidrule(lr){8-10} & 6 cl., excl. & 5 cl., excl. & 6 cl., sin excl. & 5 cl., sin excl. & & & Corrida A & Corrida B & B $-$ A\\\\")
    write(lang, 12, wrap(cap, "tab:stability", "lCCCCCCCCC", hdr, rows, size="\\footnotesize"))


# ------------------------------------------------------------------------------------------------ Tabla 13
def tab13(lang):
    fcsv = COST / "latencia.csv"
    if not fcsv.exists():
        return
    t = L[lang]
    lat = pd.read_csv(fcsv)
    lat["label"] = lat.modelo.map(PREFIX)
    lat = lat.set_index("label")
    c5 = cons[cons.variant_name == V5].set_index("label")
    order = c5.sort_values("f1_macro", ascending=False).index
    rows = []
    for m in order:
        x = lat.loc[m]
        thr_g = 32 / (x.gpu_b32_ms / 1000)
        thr_c = 32 / (x.cpu_b32_ms / 1000)
        rows.append(" & ".join([t["name"][m], f"{x.size_mb:.1f}", (f"{x.GMACs:.2f}" if x.GMACs < 1 else f"{x.GMACs:.1f}"), f"{x.gpu_b1_ms:.1f}", f"{x.cpu_b1_ms:.0f}", f"{thr_g:.0f}", f"{thr_c:.0f}",
                                f"{c5.loc[m, 'total_training_time_s'] / 3600:.2f}", f3(c5.loc[m, "f1_macro"])]) + "\\\\")
    if lang == "EN":
        cap = ("Computational cost of the ten architectures (five-class configuration). Latency is the median time to process one image (batch size 1) and throughput is images per second "
               "with batches of 32, measured with the input tensor already in memory, in FP32 and after warm-up, on an NVIDIA GeForce RTX 3050 Laptop GPU (4~GB) and an 8-thread AMD Ryzen 7 5800H CPU. "
               "GMACs: multiply--accumulate operations per image (convolutions and matrix products). Training time is the total over both phases on a Tesla T4 (Google Colab) for the five-class variant with exclusion.")
        hdr = ("\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{Weights (MB)}} & \\multirow{2}{*}{\\textbf{GMACs}} & \\multicolumn{2}{c}{\\textbf{Latency, batch 1 (ms)}} & \\multicolumn{2}{c}{\\textbf{Throughput, batch 32 (img/s)}} & "
               "\\multirow{2}{*}{\\textbf{Training (h)}} & \\multirow{2}{*}{\\textbf{Macro-F1}}\\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7} & & & GPU & CPU & GPU & CPU & &\\\\")
    else:
        cap = ("Costo computacional de las diez arquitecturas (configuración de cinco clases). La latencia es la mediana del tiempo para procesar una imagen (lote de 1) y el rendimiento son imágenes por segundo "
               "con lotes de 32, medidos con el tensor de entrada ya en memoria, en FP32 y tras un calentamiento, en una GPU NVIDIA GeForce RTX 3050 Laptop (4~GB) y una CPU AMD Ryzen 7 5800H con 8 hilos. "
               "GMACs: operaciones de multiplicar y acumular por imagen (convoluciones y productos matriciales). El tiempo de entrenamiento es el total de las dos fases en una Tesla T4 (Google Colab) para la variante de cinco clases con exclusión.")
        hdr = ("\\multirow{2}{*}{\\textbf{Modelo}} & \\multirow{2}{*}{\\textbf{Pesos (MB)}} & \\multirow{2}{*}{\\textbf{GMACs}} & \\multicolumn{2}{c}{\\textbf{Latencia, lote 1 (ms)}} & \\multicolumn{2}{c}{\\textbf{Rendimiento, lote 32 (img/s)}} & "
               "\\multirow{2}{*}{\\textbf{Entren. (h)}} & \\multirow{2}{*}{\\textbf{Macro-F1}}\\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7} & & & GPU & CPU & GPU & CPU & &\\\\")
    write(lang, 13, wrap(cap, "tab:cost", "lCCCCCCCC", hdr, rows, size="\\footnotesize"))


if __name__ == "__main__":
    for lang in ("EN", "ES"):
        for f in (tab8, tab9, tab10, tab11, tab12, tab13):
            f(lang)
    print("ok")

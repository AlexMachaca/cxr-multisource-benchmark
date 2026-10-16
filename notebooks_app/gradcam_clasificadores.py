"""Grad-CAM de los 5 modelos de la app (5 clases con exclusión) para revisar EN QUÉ SE FIJAN antes de mostrarlo a un usuario.

Tres análisis, todos sobre el test:
 A) ATENCIÓN MEDIA por clase y por fuente (Grad-CAM de la clase predicha, promediado): ¿mira pulmones o bordes/esquinas/rótulos?
    Con la fracción de "energía" del mapa en el centro, en el marco exterior y en las esquinas superiores (donde suelen ir rótulos),
    dividida por el área de esa región (1 = atención uniforme; >1 = atención de más).
 B) OCLUSIÓN (500 imágenes, 100 por clase): se tapa con gris el marco, las esquinas superiores, el centro o la región pulmonar, y se
    mide cuánto cae el desempeño; y al revés, "solo marco": si el modelo sigue acertando viendo solo los bordes, hay información de la
    clase fuera de los pulmones (atajo).
 C) EJEMPLOS: figura con las mismas imágenes en los 5 modelos y figuras por modelo. Las imágenes de SilicoData (DUA) NUNCA entran en las
    figuras de ejemplos "externas": solo en `INTERNO_*` (no publicar ni subir) y en promedios agregados.

Salida: resultados/gradcam_clasificadores/ (+ LEEME.txt). Uso: .venv-gpu/bin/python notebooks_app/gradcam_clasificadores.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks_entrenamiento"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import entrenamiento_utils as tu  # noqa: E402
from explicabilidad import TARGETS, GradCAM, load_model  # noqa: E402

VARIANT = "5clases_con_exclusion"
CLASSES = tu.CLASSES_5
DATA_ROOT = ROOT / "dataset_final_224"
RESULTS = ROOT / "resultados" / "resultados_entrenamiento"
OUT = ROOT / "resultados" / "gradcam_clasificadores"
MODELS = list(TARGETS)
LABEL = {"vit_base_patch16_224": "ViT-Base", "coatnet_0_rw_224": "CoAtNet", "swin_tiny_patch4_window7_224": "Swin-Tiny",
         "vgg16_bn": "VGG16", "mobilenetv3_small_100": "MobileNetV3"}
SRC_NICE = {"COVID-QU-Ex": "COVID-QU-Ex", "Chest_x_ray": "Chest X-ray", "RSNA_Pneumonia": "RSNA", "Shenzhen_Montgomery": "Shenzhen/Montg.",
            "SilicoData": "SilicoData", "TBX11K-simplified": "TBX11K", "VinDr_CXR": "VinDr-CXR"}
SEED, BATCH = 0, 32
device = torch.device("cuda")
SILICO_NORMAL = "SilicoData · Normal"


def regions():
    r = {}
    c = torch.zeros(224, 224, dtype=torch.bool); c[int(224 * .15):int(224 * .85), int(224 * .20):int(224 * .80)] = True
    f = torch.ones(224, 224, dtype=torch.bool); f[int(224 * .10):-int(224 * .10), int(224 * .10):-int(224 * .10)] = False
    k = torch.zeros(224, 224, dtype=torch.bool); k[:int(224 * .25), :int(224 * .25)] = True; k[:int(224 * .25), -int(224 * .25):] = True
    return {"centro": c.to(device), "marco": f.to(device), "esquinas_sup": k.to(device)}


def find_unique(name: str) -> Path:
    found = sorted(RESULTS.rglob(name))
    assert len(found) == 1, (name, found)
    return found[0]


def occlusion_masks():
    """True = región que se tapa con gris (0 en el espacio normalizado = color medio)."""
    def box(rows, cols):
        m = torch.zeros(224, 224, dtype=torch.bool)
        m[rows, cols] = True
        return m
    frame = torch.zeros(224, 224, dtype=torch.bool)
    frame[:24], frame[-24:], frame[:, :24], frame[:, -24:] = True, True, True, True
    corners = box(slice(0, 56), slice(0, 56)) | box(slice(0, 56), slice(168, 224))
    return {"ninguna": torch.zeros(224, 224, dtype=torch.bool), "sin marco (24 px)": frame, "sin esquinas sup.": corners,
            "sin centro 112²": box(slice(56, 168), slice(56, 168)), "sin región pulmonar": box(slice(40, 190), slice(40, 184)),
            "SOLO marco (24 px)": ~frame}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LEEME.txt").write_text(
        "Mapas de calor de los clasificadores (uso interno hasta verificar licencias).\n"
        "Los archivos INTERNO_* contienen imágenes derivadas de SilicoData (DUA): NO publicar, NO subir a servicios externos.\n"
        "Las figuras fig_ejemplos_* excluyen SilicoData; aun así, antes de publicar hay que verificar la licencia de la fuente de cada imagen.\n",
        encoding="utf-8")
    test = tu.load_manifest(DATA_ROOT / "manifiestos" / f"manifiesto_{VARIANT}.csv", CLASSES)["test"].reset_index(drop=True)
    y_true = test.target_class.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    src = test.source_dataset.to_numpy()
    N = len(test)
    print(f"test: {N} imágenes | fuentes: {dict(pd.Series(src).value_counts())}")
    reg = regions()
    area = {k: float(v.float().mean()) for k, v in reg.items()}
    groups = {f"clase:{c}": (y_true == i) for i, c in enumerate(CLASSES)}
    groups |= {f"fuente:{s}": (src == s) for s in np.unique(src)}
    groups |= {f"clase_fuente:{c}|{s}": (y_true == i) & (src == s) for i, c in enumerate(CLASSES) for s in np.unique(src) if ((y_true == i) & (src == s)).any()}
    # imágenes de ejemplo comunes (no SilicoData): 3 por clase (4 clases; Silicosis es 100 % SilicoData)
    rng = np.random.default_rng(SEED)
    ex_ids = {c: rng.choice(np.where((y_true == CLASSES.index(c)) & (src != "SilicoData"))[0], 3, replace=False) for c in CLASSES if c != "Silicosis"}
    sample_idx = np.concatenate([rng.choice(np.where(y_true == i)[0], 100, replace=False) for i in range(len(CLASSES))])
    cam_store, ex_store, occ_rows, occ_cls, occ_sil, en_tables = {}, {}, [], [], [], []
    sil_idx = np.where(src == "SilicoData")[0]
    masks = occlusion_masks()

    for mname in MODELS:
        t0 = time.time()
        model, dc = load_model(mname, len(CLASSES), find_unique(f"{mname}__{VARIANT}_best.pt"), device)
        tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(dc["mean"], dc["std"])])
        ds = tu.CXRDataset(test, DATA_ROOT, CLASSES, transform=tfm)
        gc = GradCAM(model, mname)
        sums, cnts = {}, {}
        pred_all, prob_all, en = np.zeros(N, int), np.zeros(N), np.zeros((N, 3))
        raw_cam = {}                                         # CAM completos solo de las imágenes de ejemplo
        want = {int(i) for v in ex_ids.values() for i in v}
        for b, (xb, _) in enumerate(DataLoader(ds, batch_size=BATCH, num_workers=6)):
            lo, hi = b * BATCH, b * BATCH + len(xb)
            logits, cls, cam = gc(xb.to(device))
            pred_all[lo:hi] = cls.cpu().numpy(); prob_all[lo:hi] = F.softmax(logits, 1).max(1)[0].cpu().numpy()
            tot = cam.flatten(1).sum(1).clamp_min(1e-8)
            for j, k in enumerate(("centro", "marco", "esquinas_sup")):
                en[lo:hi, j] = ((cam * reg[k]).flatten(1).sum(1) / tot).cpu().numpy()
            small = F.avg_pool2d(cam[:, None], 4).squeeze(1).cpu()           # 56x56
            pred_b = cls.cpu().numpy(); yb = y_true[lo:hi]; sb = src[lo:hi]
            sel = dict((k, v[lo:hi]) for k, v in groups.items())
            sel[SILICO_NORMAL + " (acierta)"] = (sb == "SilicoData") & (yb == 1) & (pred_b == 1)
            sel[SILICO_NORMAL + " (falla)"] = (sb == "SilicoData") & (yb == 1) & (pred_b != 1)
            for k, msk in sel.items():
                if msk.any():
                    sums[k] = sums.get(k, 0) + small[torch.from_numpy(msk)].sum(0); cnts[k] = cnts.get(k, 0) + int(msk.sum())
            for j in range(len(xb)):
                if lo + j in want:
                    raw_cam[lo + j] = cam[j].cpu().numpy()
        gc.remove()
        cam_store[mname] = {k: (sums[k] / cnts[k]).numpy() for k in sums}
        np.savez(OUT / f"cam_medio_{mname}.npz", **{k.replace("|", "_").replace(" ", "_").replace(":", "_"): v for k, v in cam_store[mname].items()},
                 **{"n_" + k.replace("|", "_").replace(" ", "_").replace(":", "_"): c for k, c in cnts.items()})
        ex_store[mname] = {"cam": raw_cam, "pred_map": {i: (pred_all[i], prob_all[i]) for i in want}}

        df = pd.DataFrame({"modelo": LABEL[mname], "path": test.path_final, "clase": test.target_class, "fuente": src, "pred": [CLASSES[p] for p in pred_all],
                           "prob": prob_all, "acierta": pred_all == y_true})
        for j, k in enumerate(("centro", "marco", "esquinas_sup")):
            df[f"razon_{k}"] = en[:, j] / area[k]                 # energía / área de la región (1 = uniforme)
        df.to_csv(OUT / f"energias_{mname}.csv", index=False)
        en_tables.append(df)

        # ---- oclusión (500 imágenes estratificadas)
        xs = torch.stack([ds[int(i)][0] for i in sample_idx]).to(device); ys = y_true[sample_idx]
        base = None
        for cname, mk in masks.items():
            mkd, ps = mk.to(device), []
            with torch.inference_mode():
                for i in range(0, len(xs), 20):                      # lotes chicos: VGG16 no cabe en 4 GB con 100 imágenes
                    x = xs[i:i + 20].clone(); x[:, :, mkd] = 0.0
                    ps.append(model(x).argmax(1))
            p = torch.cat(ps).cpu().numpy()
            if base is None:
                base = p
            bal = float(np.mean([(p[ys == c] == c).mean() for c in range(len(CLASSES))]))
            for c_i, c_nm in enumerate(CLASSES):
                occ_cls.append({"modelo": LABEL[mname], "condicion": cname, "clase": c_nm, "recall": float((p[ys == c_i] == c_i).mean())})
            occ_rows.append({"modelo": LABEL[mname], "condicion": cname, "area_tapada": float(mk.float().mean()), "exactitud": float((p == ys).mean()),
                             "exactitud_balanceada": bal, "pred_sin_cambio": float((p == base).mean())})
        # ---- las 348 imágenes de SilicoData: ¿la señal está en las esquinas / fuera de los pulmones?
        xs_sil = torch.stack([ds[int(i)][0] for i in sil_idx]); ys_sil = y_true[sil_idx]
        for cname in ("ninguna", "sin esquinas sup.", "sin marco (24 px)", "sin región pulmonar", "SOLO marco (24 px)"):
            mkd, ps = masks[cname].to(device), []
            with torch.inference_mode():
                for i in range(0, len(xs_sil), 20):
                    x = xs_sil[i:i + 20].to(device).clone(); x[:, :, mkd] = 0.0
                    ps.append(model(x).argmax(1))
            p = torch.cat(ps).cpu().numpy()
            r = {"modelo": LABEL[mname], "condicion": cname, "exactitud": float((p == ys_sil).mean())}
            for c_i, c_nm in enumerate(CLASSES):
                if (ys_sil == c_i).any():
                    r[f"recall_{c_nm}"] = float((p[ys_sil == c_i] == c_i).mean())
            for c_i, c_nm in enumerate(CLASSES):
                r[f"pred_{c_nm}"] = float((p == c_i).mean())          # qué se predice, en general, para imágenes de SilicoData
            occ_sil.append(r)
        del model, xs, xs_sil
        torch.cuda.empty_cache()
        print(f"[ok] {LABEL[mname]} ({time.time() - t0:.0f} s)")

    # ---------------- tablas ----------------
    E = pd.concat(en_tables)
    E_sum = E.groupby(["modelo", "fuente"])[["razon_centro", "razon_marco", "razon_esquinas_sup"]].median().round(3)
    E_sum.to_csv(OUT / "tabla_energia_por_modelo_fuente.csv")
    E_cls = E.groupby(["modelo", "clase"])[["razon_centro", "razon_marco", "razon_esquinas_sup"]].median().round(3)
    E_cls.to_csv(OUT / "tabla_energia_por_modelo_clase.csv")
    print("\nrazón energía/área, mediana por modelo (todas las imágenes):")
    print(E.groupby("modelo")[["razon_centro", "razon_marco", "razon_esquinas_sup"]].median().round(2).to_string())
    print("\nrazón del MARCO por modelo y fuente (1 = uniforme):")
    print(E_sum.razon_marco.unstack().round(2).to_string())
    sn = E[(E.fuente == "SilicoData") & (E.clase == "Normal")]
    print("\nSilicoData·Normal: razón del marco, aciertos vs fallos:")
    print(sn.groupby(["modelo", "acierta"]).razon_marco.median().unstack().round(2).to_string())
    O = pd.DataFrame(occ_rows); O.to_csv(OUT / "tabla_oclusion.csv", index=False)
    print("\nexactitud balanceada (500 imágenes) por condición de oclusión:")
    print(O.pivot(index="modelo", columns="condicion", values="exactitud_balanceada")[list(masks)].round(2).to_string())
    print("(azar = 0.20)")
    OC = pd.DataFrame(occ_cls); OC.to_csv(OUT / "tabla_oclusion_por_clase.csv", index=False)
    OS = pd.DataFrame(occ_sil); OS.to_csv(OUT / "tabla_oclusion_silicodata.csv", index=False)
    for cond in ("sin región pulmonar", "SOLO marco (24 px)"):
        print(f"\nrecall por clase con '{cond}' (500 imágenes):")
        print(OC[OC.condicion == cond].pivot(index="modelo", columns="clase", values="recall").round(2).to_string())
    print("\nSilicoData (348 imágenes): exactitud y recall por clase según la condición")
    for cond in ("ninguna", "sin esquinas sup.", "sin región pulmonar", "SOLO marco (24 px)"):
        print(f"  -- {cond}"); print(OS[OS.condicion == cond].set_index("modelo")[["exactitud", "recall_Normal", "recall_Silicosis", "recall_Tuberculosis"]].round(2).to_string())

    # ---------------- figuras ----------------
    order = [LABEL[m] for m in MODELS]
    rows_f = [f"clase:{c}" for c in CLASSES] + [SILICO_NORMAL + " (acierta)", SILICO_NORMAL + " (falla)"]
    fig, axs = plt.subplots(len(rows_f), len(MODELS), figsize=(2.1 * len(MODELS), 1.9 * len(rows_f)))
    for j, m in enumerate(MODELS):
        for i, k in enumerate(rows_f):
            ax = axs[i, j]; ax.axis("off")
            if k in cam_store[m]:
                ax.imshow(cam_store[m][k], cmap="inferno", vmin=0, vmax=1)
            if j == 0:
                ax.text(-0.08, .5, k.replace("clase:", ""), transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
            if i == 0:
                ax.set_title(LABEL[m], fontsize=10, fontweight="bold")
    fig.suptitle("Atención media (Grad-CAM de la clase predicha) · test, 5 clases", x=.02, ha="left", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(OUT / "fig_cam_medio_clase_modelo.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_cam_medio_clase_modelo.pdf", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    piv = E_sum.razon_marco.unstack().reindex(columns=[s for s in SRC_NICE if s in E.fuente.unique()]).reindex(order)
    w = .16
    for j, m in enumerate(order):
        ax.bar(np.arange(piv.shape[1]) + (j - 2) * w, piv.loc[m], w, label=m)
    ax.axhline(1, color="#555", lw=1, ls="--"); ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([SRC_NICE[s] for s in piv.columns], fontsize=8.5)
    ax.set_ylabel("energía en el marco / área del marco"); ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "fig_energia_marco_por_fuente.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_energia_marco_por_fuente.pdf", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    conds = list(masks)
    for j, m in enumerate(order):
        d = O[O.modelo == m].set_index("condicion").exactitud_balanceada.reindex(conds)
        ax.plot(range(len(conds)), d.values, "o-", lw=2, label=m)
    ax.axhline(.2, color="#999", ls="--", lw=1); ax.text(len(conds) - 1, .205, "azar", ha="right", color="#777", fontsize=8)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=8.5, rotation=15); ax.set_ylabel("exactitud balanceada (500 imágenes)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.13)); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(OUT / "fig_oclusion.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_oclusion.pdf", bbox_inches="tight"); plt.close()

    # ejemplos: mismas imágenes en los 5 modelos
    ex_classes = [c for c in CLASSES if c != "Silicosis"]
    def img_of(i):
        return np.asarray(tu.CXRDataset(test.iloc[[i]].reset_index(drop=True), DATA_ROOT, CLASSES, transform=T.Compose([T.Resize((224, 224))]))[0][0].convert("L"))
    def overlay(ax, im, cam, title):
        ax.imshow(im, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=.42, vmin=0, vmax=1); ax.axis("off"); ax.set_title(title, fontsize=7)
    fig, axs = plt.subplots(len(MODELS) + 1, len(ex_classes), figsize=(2.6 * len(ex_classes), 2.55 * (len(MODELS) + 1)))
    for j, c in enumerate(ex_classes):
        i0 = int(ex_ids[c][0]); axs[0, j].imshow(img_of(i0), cmap="gray"); axs[0, j].axis("off")
        axs[0, j].set_title(f"{c}\n({SRC_NICE[src[i0]]})", fontsize=9, fontweight="bold")
        for r, m in enumerate(MODELS, start=1):
            p, pr = ex_store[m]["pred_map"][i0]
            overlay(axs[r, j], img_of(i0), ex_store[m]["cam"][i0], f"{LABEL[m]}: {CLASSES[p]} ({pr:.2f})")
    plt.tight_layout(); plt.savefig(OUT / "fig_ejemplos_5_modelos.png", dpi=300, bbox_inches="tight"); plt.savefig(OUT / "fig_ejemplos_5_modelos.pdf", bbox_inches="tight"); plt.close()

    for m in MODELS:
        fig, axs = plt.subplots(len(ex_classes), 6, figsize=(12.6, 2.2 * len(ex_classes)))
        for i, c in enumerate(ex_classes):
            for k, idx in enumerate(ex_ids[c]):
                idx = int(idx); p, pr = ex_store[m]["pred_map"][idx]; im = img_of(idx)
                axs[i, 2 * k].imshow(im, cmap="gray"); axs[i, 2 * k].axis("off")
                overlay(axs[i, 2 * k + 1], im, ex_store[m]["cam"][idx], f"{CLASSES[p]} ({pr:.2f})")
                if k == 0:
                    axs[i, 0].set_title(f"{c}", fontsize=9, loc="left", fontweight="bold")
        fig.suptitle(f"{LABEL[m]}: Grad-CAM de la clase predicha (5 clases)", x=.01, ha="left", fontsize=10, fontweight="bold")
        plt.tight_layout(); plt.savefig(OUT / f"fig_ejemplos_{m}.png", dpi=200, bbox_inches="tight"); plt.close()
    print("\nlisto:", OUT)


if __name__ == "__main__":
    main()

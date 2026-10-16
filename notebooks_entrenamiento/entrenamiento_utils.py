"""Funciones compartidas por los notebooks de entrenamiento (Fase 2) en notebooks_entrenamiento/.

Cada notebook de arquitectura (VGG16, ResNet50, ViT, Swin, ...) solo define: qué modelo de
`timm` cargar, y llama a las funciones de aquí para dataset, entrenamiento en 2 fases,
evaluación y guardado de resultados — así los 10 notebooks quedan cortos y consistentes entre
sí, en vez de repetir la misma lógica 10 veces.

Uso desde un notebook en este mismo directorio:
    import entrenamiento_utils as tu
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CLASSES_6 = ["COVID-19", "Normal", "Pneumonia", "Silicosis", "STB", "Tuberculosis"]
CLASSES_5 = ["COVID-19", "Normal", "Pneumonia", "Silicosis", "Tuberculosis"]  # sin STB


# ---------------------------------------------------------------------------
# Rutas: el manifiesto se generó corriendo el notebook de consolidación desde
# notebooks_eda/, así que `path_final` quedó como "../dataset_final/images/...".
# Eso no es portable (no existe ese folder relativo en Colab). Esta función
# reconstruye la ruta real a partir de DATA_ROOT, sin importar desde dónde se
# generó el manifiesto ni dónde esté DATA_ROOT ahora (local o /content/... en Colab).
# ---------------------------------------------------------------------------

def resolve_path(path_final: str, data_root: Path) -> Path:
    parts = Path(path_final).parts
    if "dataset_final" in parts:
        idx = parts.index("dataset_final")
        rel = Path(*parts[idx + 1:])
    else:
        rel = Path(path_final).name
    return data_root / rel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CXRDataset(Dataset):
    """Lee imágenes a partir de un manifiesto (dataframe ya filtrado a UN split)."""

    def __init__(self, df: pd.DataFrame, data_root: Path, classes: list[str], transform=None):
        self.paths = [resolve_path(p, data_root) for p in df["path_final"]]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[c] for c in df["target_class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
        return im, self.labels[idx]


def load_manifest(manifest_path: Path, classes: list[str]) -> dict[str, pd.DataFrame]:
    """Lee el CSV de la variante elegida y lo separa en train/val/test."""
    df = pd.read_csv(manifest_path)
    df = df[df["target_class"].isin(classes)].copy()
    return {split: df[df["split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}


# ---------------------------------------------------------------------------
# Balanceo de clases: DOS mecanismos posibles, pero se usa solo UNO a la vez
# para no sobre-compensar el desbalance (aplicar ambos a la vez sesgaría el
# entrenamiento hacia las clases raras más de lo necesario):
#
#   1. compute_class_weights: pondera el ERROR por clase (lo que se usó antes).
#   2. make_weighted_sampler: cambia qué imágenes se ven en cada época — las
#      clases raras (Silicosis, STB) se muestrean con más frecuencia, en vez
#      de solo pesar más su error. Es la opción activa ahora en los notebooks
#      de entrenamiento (ver `train_two_phase`, se llama con class_weights=None
#      cuando el DataLoader de train ya usa este sampler).
# ---------------------------------------------------------------------------

def compute_class_weights(train_df: pd.DataFrame, classes: list[str]) -> torch.Tensor:
    counts = train_df["target_class"].value_counts()
    total = len(train_df)
    n_classes = len(classes)
    weights = [total / (n_classes * counts.get(c, 1)) for c in classes]
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(train_df: pd.DataFrame, classes: list[str]) -> WeightedRandomSampler:
    """Sampler que balancea las clases DENTRO de cada época: cada imagen se pondera por el
    inverso de la frecuencia de su clase, así las clases raras aparecen proporcionalmente más
    seguido en los batches (con reemplazo), en vez de solo pesar más su contribución al error."""
    counts = train_df["target_class"].value_counts()
    sample_weights = train_df["target_class"].map(lambda c: 1.0 / counts[c]).to_numpy()
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Congelar / descongelar por fracción — genérico para CNN y Transformers:
# para una CNN, "las últimas capas" son las últimas capas convolucionales;
# para un Transformer, son los últimos bloques de atención. En ambos casos,
# recorrer los parámetros EN ORDEN y descongelar la última fracción captura
# la misma idea sin necesitar una regla distinta por arquitectura.
# ---------------------------------------------------------------------------

def freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_last_fraction(module: nn.Module, fraction: float = 0.3):
    named = list(module.named_parameters())
    n_unfreeze = max(1, int(len(named) * fraction))
    for name, p in named[-n_unfreeze:]:
        p.requires_grad = True


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def model_size_mb(model: nn.Module) -> float:
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Entrenamiento en 2 fases (mismo esquema que ya validaron en IMJETA V2):
# fase 1 con el backbone congelado, solo el clasificador; fase 2 descongelando
# la última fracción del backbone a una tasa de aprendizaje menor, con
# early stopping sobre la pérdida de validación.
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return total_loss / n, correct / n


def train_two_phase(
    model, train_loader, val_loader, class_weights, device,
    phase1_epochs=8, phase1_lr=1e-3,
    phase2_epochs=12, phase2_lr=1e-5,
    unfreeze_fraction=0.3, patience=5,
    checkpoint_path: Path | None = None,
    log_fn=print,
):
    """`class_weights=None` cuando el balanceo ya se hace en el DataLoader (ver
    `make_weighted_sampler`) — aplicar pesos de clase Y muestreo balanceado a la vez
    sobre-compensaría el desbalance."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    history = {
        "phase": [], "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "epoch_time_s": [],
    }

    def _run_phase(phase_name, epochs, lr, best_val_loss, epochs_no_improve):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            epoch_time = time.time() - t0
            scheduler.step(val_loss)
            history["phase"].append(phase_name)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_time_s"].append(epoch_time)
            log_fn(f"[{phase_name}] epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} "
                   f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                   f"({epoch_time:.1f}s)")
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log_fn(f"[{phase_name}] early stopping (sin mejora en {patience} épocas)")
                    break
        return best_val_loss, epochs_no_improve

    # Fase 1: backbone congelado
    freeze_all(model)
    for p in model.get_classifier().parameters() if hasattr(model, "get_classifier") else []:
        p.requires_grad = True
    best_val_loss, epochs_no_improve = _run_phase("fase1_congelado", phase1_epochs, phase1_lr, float("inf"), 0)

    # Fase 2: descongelar la última fracción y afinar a LR más baja
    unfreeze_last_fraction(model, unfreeze_fraction)
    _run_phase("fase2_finetuning", phase2_epochs, phase2_lr, best_val_loss, 0)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history


# ---------------------------------------------------------------------------
# Evaluación: mismas métricas que ya usaron en el artículo V2 (accuracy,
# precision/recall/F1 macro y por clase, AUC-ROC uno-contra-todos, matriz de
# confusión) + intervalos de Wilson al 95% para precision/recall por clase
# (relevante dado que Silicosis/STB tienen pocas imágenes de test) + costo
# computacional (parámetros, tamaño en disco, tiempo de inferencia).
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    return ((centre - adj) / denom, (centre + adj) / denom)


@torch.no_grad()
def evaluate_model(model, test_loader, classes: list[str], device) -> dict:
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    t0 = time.time()
    n_images = 0
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        n_images += images.size(0)
    inference_time_ms = (time.time() - t0) / max(n_images, 1) * 1000

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_class = []
    for i, c in enumerate(classes):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        pred_pos = int((y_pred == i).sum())
        true_pos = int((y_true == i).sum())
        p_lo, p_hi = wilson_ci(tp, pred_pos) if pred_pos else (float("nan"), float("nan"))
        r_lo, r_hi = wilson_ci(tp, true_pos) if true_pos else (float("nan"), float("nan"))
        try:
            auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            auc = float("nan")
        per_class.append({
            "class": c, "precision": float(precision[i]), "precision_ci_lo": p_lo, "precision_ci_hi": p_hi,
            "recall": float(recall[i]), "recall_ci_lo": r_lo, "recall_ci_hi": r_hi,
            "f1": float(f1[i]), "support": int(support[i]), "auc_roc": float(auc),
        })

    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(len(classes))))
    except ValueError:
        auc_macro = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc_roc_macro": float(auc_macro),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "inference_time_ms_per_image": inference_time_ms,
        "n_test_images": int(n_images),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


# ---------------------------------------------------------------------------
# Graficado: mismas figuras para los 10 modelos (curvas, matriz de confusión,
# ROC uno-contra-todos, métricas por clase con IC de Wilson). Se guardan en
# PNG a 600 dpi (BioMedInformatics/MDPI exige resolución mínima de 600 dpi
# para figuras) y en PDF (vectorial, no depende de dpi, preferible si el
# sistema de envío lo acepta).
# ---------------------------------------------------------------------------

def _save_fig(fig, png_path: Path):
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_training_curves(history: dict, model_name: str, variant_name: str, output_dir: Path):
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for phase in hist_df["phase"].unique():
        sub = hist_df[hist_df["phase"] == phase]
        axes[0].plot(sub.index, sub["train_loss"], label=f"{phase} train")
        axes[0].plot(sub.index, sub["val_loss"], label=f"{phase} val", linestyle="--")
        axes[1].plot(sub.index, sub["train_acc"], label=f"{phase} train")
        axes[1].plot(sub.index, sub["val_acc"], label=f"{phase} val", linestyle="--")
    axes[0].set_title("Pérdida"); axes[0].set_xlabel("Época"); axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Época"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}__{variant_name}_curvas.png")
    return fig


def plot_confusion_matrix(metrics: dict, classes: list[str], model_name: str, variant_name: str, output_dir: Path):
    """Guarda DOS versiones: conteos crudos y normalizada por fila (%). Con clases tan
    desiguales en soporte de test (p. ej. Normal ~900 vs. STB ~110), la de conteos hace que
    los errores de las clases chicas casi no se vean — la normalizada corrige eso."""
    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (conteos) — {model_name} ({variant_name})")
    plt.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}__{variant_name}_matriz_confusion.png")

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True) * 100)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicho"); ax2.set_ylabel("Real")
    ax2.set_title(f"Matriz de confusión normalizada (%) — {model_name} ({variant_name})")
    plt.tight_layout()
    _save_fig(fig2, output_dir / f"{model_name}__{variant_name}_matriz_confusion_normalizada.png")
    return fig, fig2


def plot_roc_curves(metrics: dict, classes: list[str], model_name: str, variant_name: str, output_dir: Path):
    """Curvas ROC uno-contra-todos, una por clase, con el AUC ya calculado en `evaluate_model`."""
    y_true = np.array(metrics["y_true"])
    y_prob = np.array(metrics["y_prob"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        except ValueError:
            continue
        auc = metrics["per_class"][i]["auc_roc"]
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC uno-contra-todos — {model_name} ({variant_name})")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}__{variant_name}_roc.png")
    return fig


def plot_pr_curves(metrics: dict, classes: list[str], model_name: str, variant_name: str, output_dir: Path):
    """Curvas Precision-Recall uno-contra-todos, una por clase (AP = average precision).
    Más informativa que ROC cuando las clases están desbalanceadas en el test (Silicosis/STB
    aquí): ROC incluye los verdaderos negativos, que dominan y pueden maquillar una precisión
    mala en la clase rara; PR no tiene ese problema."""
    y_true = np.array(metrics["y_true"])
    y_prob = np.array(metrics["y_prob"])
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(classes):
        y_true_bin = (y_true == i).astype(int)
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, i])
            ap = average_precision_score(y_true_bin, y_prob[:, i])
        except ValueError:
            continue
        ax.plot(recall, precision, label=f"{c} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"Curvas Precision-Recall uno-contra-todos — {model_name} ({variant_name})")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}__{variant_name}_precision_recall.png")
    return fig


def plot_per_class_metrics(metrics: dict, model_name: str, variant_name: str, output_dir: Path):
    """Barras de precision/recall/F1 por clase, con barras de error = IC 95% de Wilson
    (precision/recall solamente — F1 no tiene una fórmula de IC tan directa, se deja sin barra)."""
    per_class = pd.DataFrame(metrics["per_class"])
    x = np.arange(len(per_class))
    width = 0.25

    def _err(col_lo, col_hi, col_val):
        lo = (per_class[col_val] - per_class[col_lo]).to_numpy()
        hi = (per_class[col_hi] - per_class[col_val]).to_numpy()
        return np.nan_to_num(np.vstack([lo, hi]), nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, per_class["precision"], width,
           yerr=_err("precision_ci_lo", "precision_ci_hi", "precision"), label="Precision", capsize=3)
    ax.bar(x, per_class["recall"], width,
           yerr=_err("recall_ci_lo", "recall_ci_hi", "recall"), label="Recall", capsize=3)
    ax.bar(x + width, per_class["f1"], width, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(per_class["class"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Métricas por clase (barras de error: IC 95% de Wilson) — {model_name} ({variant_name})")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, output_dir / f"{model_name}__{variant_name}_metricas_por_clase.png")
    return fig


# ---------------------------------------------------------------------------
# Guardado de resultados: mismo formato para los 10 modelos, para que el
# notebook de comparación final los pueda leer todos igual. Guarda:
#   1. El JSON completo (historial, métricas agregadas, costo, tiempo total de
#      entrenamiento, mejor época, GPU usada).
#   2. Las métricas por clase como CSV (fácil de abrir/pegar en el artículo).
#   3. Las predicciones crudas (y_true/y_pred/y_prob) en un .npz -- SIN esto no
#      se puede, más adelante, hacer pruebas de significancia entre modelos
#      (McNemar, DeLong) ni una curva ROC/PR combinada de los 10 modelos, y
#      solo se podría recuperar re-evaluando cada modelo desde su checkpoint.
#   4. Una fila en `comparacion_modelos.csv`, compartida entre TODOS los
#      modelos ya entrenados en `output_dir` — si el mismo modelo+variante ya
#      tenía una fila, se reemplaza (permite re-entrenar sin duplicar filas).
# ---------------------------------------------------------------------------

def saved_epoch_index(val_losses: list) -> int:
    """Índice de la época cuyo checkpoint SE GUARDÓ de verdad en `train_two_phase`: la ÚLTIMA que mejora la
    pérdida de validación en más de 1e-4 respecto al mejor anterior (mejor acumulado a través de las 2 fases).
    OJO: no es `argmin(val_loss)`: con dos épocas casi empatadas (diferencia < 1e-4) el argmin apunta a una
    época que nunca se guardó (ej. ViT-Base 5 clases: se guardó la 10, el argmin dice 11)."""
    best, idx = float("inf"), 0
    for i, v in enumerate(val_losses):
        if v < best - 1e-4:
            best, idx = v, i
    return idx


def environment_info() -> dict:
    """Versiones del entorno de ejecución, para reportarlas en el artículo y poder reproducir la corrida."""
    import platform
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except Exception:
            return None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": ver("torchvision"),
        "timm": ver("timm"),
        "numpy": ver("numpy"),
        "scikit-learn": ver("scikit-learn"),
        "pandas": ver("pandas"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
    }


def save_results(
    model_name: str, variant_name: str, history: dict, metrics: dict,
    params_info: dict, size_mb: float, output_dir: Path, extra: dict | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    total_training_time_s = float(sum(history["epoch_time_s"])) if history.get("epoch_time_s") else None
    if history.get("val_loss"):
        best_idx = saved_epoch_index(history["val_loss"])
        best_epoch_info = {"phase": history["phase"][best_idx], "epoch": history["epoch"][best_idx]}
    else:
        best_epoch_info = {"phase": None, "epoch": None}
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    record = {
        "model_name": model_name,
        "variant_name": variant_name,
        "params": params_info,
        "size_mb": size_mb,
        "total_training_time_s": total_training_time_s,
        "best_checkpoint": best_epoch_info,
        "gpu": gpu_name,
        "entorno": environment_info(),
        "extra": extra or {},
        "history": history,
        "metrics": {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred", "y_prob")},
    }
    out_path = output_dir / f"{model_name}__{variant_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    per_class_path = output_dir / f"{model_name}__{variant_name}_metricas_por_clase.csv"
    pd.DataFrame(metrics["per_class"]).to_csv(per_class_path, index=False)

    npz_path = output_dir / f"{model_name}__{variant_name}_predicciones.npz"
    np.savez_compressed(
        npz_path,
        y_true=np.array(metrics["y_true"]),
        y_pred=np.array(metrics["y_pred"]),
        y_prob=np.array(metrics["y_prob"]),
    )

    summary_row = {
        "model_name": model_name, "variant_name": variant_name,
        "accuracy": metrics["accuracy"], "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"], "f1_macro": metrics["f1_macro"],
        "auc_roc_macro": metrics["auc_roc_macro"],
        "total_params": params_info["total_params"], "trainable_params": params_info["trainable_params"],
        "size_mb": size_mb, "inference_time_ms_per_image": metrics["inference_time_ms_per_image"],
        "n_test_images": metrics["n_test_images"],
        "total_training_time_s": total_training_time_s,
        "best_epoch": best_epoch_info["epoch"], "best_phase": best_epoch_info["phase"],
        "gpu": gpu_name,
    }
    comp_path = output_dir / "comparacion_modelos.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[~((comp_df["model_name"] == model_name) & (comp_df["variant_name"] == variant_name))]
        comp_df = pd.concat([comp_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        comp_df = pd.DataFrame([summary_row])
    comp_df.to_csv(comp_path, index=False)

    return out_path

"""Funciones compartidas por los notebooks de EDA (Fase 0) en notebooks_eda/.

Cada notebook define su propia lógica de carga/mapeo de etiquetas (eso sí cambia
por fuente: carpetas por clase, CSV, DICOM, etc.), pero reutiliza de aquí las
partes que son iguales en todos: verificación de integridad, heurísticas de
calidad de imagen, hashing MD5/perceptual y las funciones de graficado.

Uso desde un notebook en este mismo directorio:
    import eda_utils as eu
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_THUMBNAIL_SIZE = (256, 256)  # las heurísticas de calidad no necesitan resolución completa;
                               # a esta escala son ~10-20x más rápidas y el resultado es equivalente


def list_images(folder: Path, exts=IMAGE_EXTS):
    """Todas las imágenes bajo `folder` (recursivo) con extensión reconocida."""
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def probe_image(path):
    """Verifica que la imagen abra y devuelve (ancho, alto, modo, error)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() invalida el objeto, se reabre para leer datos
            return im.size[0], im.size[1], im.mode, None
    except (UnidentifiedImageError, OSError) as e:
        return None, None, None, str(e)


def md5_file(path, chunk_size: int = 8192) -> str:
    """Hash exacto del archivo (detecta copias byte-idénticas, no sirve entre formatos distintos)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Heurísticas de calidad y hash perceptual: la lógica "de verdad" trabaja sobre
# un array 2D en escala de grises ya reducido a miniatura (_from_array). Las
# variantes _file/_path cargan con PIL; para fuentes que no son PIL (DICOM) se
# arma el array una sola vez con pydicom y se reutiliza para las tres cosas
# (bordes, brillo, phash) sin volver a decodificar — ver `probe_dicom`.
# ---------------------------------------------------------------------------

def _load_thumbnail_gray(path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail(_THUMBNAIL_SIZE)
        return np.array(im, dtype=np.float32)


def array_to_thumbnail_gray(arr: np.ndarray) -> np.ndarray:
    im = Image.fromarray(arr).convert("L")
    im.thumbnail(_THUMBNAIL_SIZE)
    return np.array(im, dtype=np.float32)


def content_ratio_from_array(arr: np.ndarray, border_std_threshold: float = 8.0):
    """Fracción del área que es contenido real (no margen uniforme en blanco/negro). `arr` ya en miniatura."""
    row_std = arr.std(axis=1)
    col_std = arr.std(axis=0)
    rows = np.where(row_std > border_std_threshold)[0]
    cols = np.where(col_std > border_std_threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0
    content_h = rows[-1] - rows[0] + 1
    content_w = cols[-1] - cols[0] + 1
    return (content_h * content_w) / (arr.shape[0] * arr.shape[1])


def glare_fraction_from_array(arr: np.ndarray, margin_frac: float = 0.1, bright_threshold: int = 245):
    """Fracción de píxeles casi-blancos en el CENTRO de la imagen. `arr` ya en miniatura."""
    h, w = arr.shape
    my, mx = int(h * margin_frac), int(w * margin_frac)
    center = arr[my:h - my, mx:w - mx]
    if center.size == 0:
        return 0.0
    return float((center > bright_threshold).mean())


def phash_from_array(arr: np.ndarray) -> str:
    """Hash perceptual (imagehash.phash): detecta imágenes visualmente iguales/casi-iguales
    aunque el archivo original sea de otro formato o resolución (a diferencia de md5_file)."""
    return str(imagehash.phash(Image.fromarray(arr.astype(np.uint8))))


def content_bbox_ratio(path, border_std_threshold: float = 8.0):
    """Ver `content_ratio_from_array`. Carga la imagen desde disco con PIL."""
    try:
        arr = _load_thumbnail_gray(path)
    except Exception:
        return None
    return content_ratio_from_array(arr, border_std_threshold)


def glare_fraction(path, margin_frac: float = 0.1, bright_threshold: int = 245):
    """Ver `glare_fraction_from_array`. Carga la imagen desde disco con PIL."""
    try:
        arr = _load_thumbnail_gray(path)
    except Exception:
        return None
    return glare_fraction_from_array(arr, margin_frac, bright_threshold)


def phash_file(path) -> str | None:
    """Ver `phash_from_array`. Carga la imagen desde disco con PIL."""
    try:
        arr = _load_thumbnail_gray(path)
    except Exception:
        return None
    return phash_from_array(arr)


def flag_quality_issues(df, path_col: str = "path", content_ratio_threshold: float = 0.55,
                         glare_threshold: float = 0.15, desc: str = ""):
    """Agrega content_ratio/glare_fraction/flag_bordes/flag_brillo/motivo a una COPIA de df."""
    out = df.copy()
    out["content_ratio"] = [content_bbox_ratio(p) for p in tqdm(out[path_col], desc=f"Bordes {desc}".strip())]
    out["glare_fraction"] = [glare_fraction(p) for p in tqdm(out[path_col], desc=f"Brillo {desc}".strip())]
    out["flag_bordes"] = out["content_ratio"] < content_ratio_threshold
    out["flag_brillo"] = out["glare_fraction"] > glare_threshold
    out["motivo"] = out.apply(
        lambda r: ", ".join(
            m for m, f in [
                ("bordes/recorte incompleto", r["flag_bordes"]),
                ("posible brillo/reflejo", r["flag_brillo"]),
            ] if f
        ),
        axis=1,
    )
    return out


# ---------------------------------------------------------------------------
# DICOM (RSNA, y más adelante VinDr-CXR)
# ---------------------------------------------------------------------------

def probe_dicom(path):
    """Decodifica un DICOM UNA sola vez. Devuelve (array_pixeles, error).

    Ojo: devuelve el array RAW tal cual lo decodifica pydicom. Para DICOM de 8 bits
    (p. ej. RSNA, ya en rango 0-255) esto es directamente utilizable. Para DICOM de
    alto rango dinámico real (p. ej. VinDr-CXR, 14 bits / 0-16383) hay que normalizar
    antes de convertir a escala de grises de 8 bits — usar `probe_dicom_normalized`.
    """
    import pydicom
    try:
        ds = pydicom.dcmread(path)
        return ds.pixel_array, None
    except Exception as e:
        return None, str(e)


def probe_dicom_normalized(path):
    """Como `probe_dicom`, pero para DICOM de alto rango dinámico (>8 bits reales):
    aplica VOI LUT (WindowCenter/WindowWidth de la cabecera, lo que vería un radiólogo)
    y reescala a uint8 (0-255) antes de devolver el array. Necesario porque
    `Image.fromarray(arr).convert("L")` sobre un array de 16 bits TRUNCA en vez de
    reescalar (satura casi todo a blanco) — se detectó al ver que VinDr-CXR marcaba
    100% de las imágenes como "brillo" y un phash degenerado con muchos falsos
    duplicados; RSNA no lo sufre porque ya es 8 bits (0-245) de origen.
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float64)
        if ds.BitsStored > 8:
            arr = apply_voi_lut(arr, ds)
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                arr = arr.max() - arr
            arr = arr - arr.min()
            max_val = arr.max()
            if max_val > 0:
                arr = arr / max_val * 255.0
        return arr.astype(np.uint8), None
    except Exception as e:
        return None, str(e)


def dicom_metadata(path, tags=("PatientAge", "PatientSex", "ViewPosition", "Modality", "BodyPartExamined")):
    """Lee solo los metadatos (sin decodificar píxeles) — mucho más rápido que `probe_dicom`."""
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return {t: str(ds.get(t)) if ds.get(t) is not None else None for t in tags}
    except Exception:
        return {t: None for t in tags}


# ---------------------------------------------------------------------------
# Graficado. `loader(path) -> array o imagen PIL` permite reusar estas mismas
# funciones con fuentes no-PIL (DICOM) sin duplicar el código de plotting.
# ---------------------------------------------------------------------------

def _default_loader(path):
    return Image.open(path)


def show_samples(data, class_col: str = "target_class", path_col: str = "path",
                  n_per_class: int = 4, seed: int = 42, save_path=None, loader=_default_loader):
    classes = sorted(data[class_col].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(3 * n_per_class, 3 * len(classes)))
    axes = np.array(axes).reshape(len(classes), -1)
    for i, cls in enumerate(classes):
        subset = data[data[class_col] == cls]
        sample = subset[path_col].sample(min(n_per_class, len(subset)), random_state=seed)
        for j, p in enumerate(sample):
            ax = axes[i, j]
            ax.imshow(loader(p), cmap="gray")
            ax.set_title(cls if j == 0 else "", fontsize=10, loc="left")
            ax.axis("off")
        for j in range(len(sample), axes.shape[1]):
            axes[i, j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def show_flagged(data, path_col: str = "path", label_cols=("target_class", "motivo"),
                  n: int = 12, seed: int = 42, save_path=None, loader=_default_loader):
    if len(data) == 0:
        print("No hay candidatas marcadas.")
        return
    sample = data.sample(min(n, len(data)), random_state=seed)
    cols = 4
    rows = -(-len(sample) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, sample.iterrows()):
        ax.imshow(loader(row[path_col]), cmap="gray")
        ax.set_title(" | ".join(str(row[c]) for c in label_cols), fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


def show_duplicate_group(group, path_col: str = "path", label_fn=None, loader=_default_loader):
    fig, axes = plt.subplots(1, len(group), figsize=(3 * len(group), 3))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, group.iterrows()):
        ax.imshow(loader(row[path_col]), cmap="gray")
        ax.set_title(label_fn(row) if label_fn else Path(row[path_col]).name, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

"""Genera una copia de `dataset_final/` con las imágenes ya reducidas a 224x224 --
`dataset_final_224/` -- para no repetir ese resize desde la resolución original en
cada época x cada modelo x cada variante durante el entrenamiento.

No cambia NADA de lo que ya se entrenó ni de lo que ve el modelo: los 10 notebooks de
entrenamiento ya reducen cada imagen a 224x224 como primer paso de la transformación
(`transforms.Resize(input_size)`, antes de cualquier aumento de datos) -- acá se hace
exactamente esa misma operación (mismo tamaño, misma interpolación bilineal, mismo
`.convert("RGB")` que hace `CXRDataset.__getitem__`) una sola vez, de antemano, en vez
de repetirla en caliente cada vez que se lee la imagen. El píxel final que ve el modelo
es idéntico; solo se ahorra el trabajo de CPU repetido de decodificar/reescalar
imágenes de hasta varios MB (algunas fuentes, p. ej. VinDr-CXR/Shenzhen-Montgomery,
llegan a más de 2500x2900 px) una y otra vez.

`dataset_final/` original NO se toca ni se borra -- esta es una copia nueva, separada.

Uso: `uv run python preresize_dataset.py` desde `notebooks_entrenamiento/`. Se puede
volver a correr sin problema (no vuelve a procesar una imagen si el archivo destino ya
existe).

Para usar la copia redimensionada en un notebook de entrenamiento, solo hay que apuntar
`DATA_ROOT` a `dataset_final_224` en vez de `dataset_final` (local o en Colab, subiendo
`dataset_final_224.zip` en vez de `dataset_final.zip`) -- el resto del pipeline no
cambia: `resolve_path()` en `entrenamiento_utils.py` reconstruye la ruta real contra
cualquier `DATA_ROOT` que se le pase, así que los manifiestos (que se copian tal cual,
sin editar ninguna ruta) funcionan igual contra la copia nueva.
"""
from __future__ import annotations

import shutil
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SRC_DIR = Path("../dataset_final")
DST_DIR = Path("../dataset_final_224")
TARGET_SIZE = (224, 224)  # mismo tamaño de entrada que usan los 10 modelos ya entrenados


def resize_one(paths: tuple[Path, Path]) -> str | None:
    src, dst = paths
    if dst.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            im.save(dst)
        return None
    except Exception as e:
        return f"{src}: {e}"


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def main():
    src_images = sorted((SRC_DIR / "images").rglob("*.png"))
    print(f"Imágenes encontradas en {SRC_DIR / 'images'}: {len(src_images)}")

    tasks = [(src, DST_DIR / src.relative_to(SRC_DIR)) for src in src_images]

    n_workers = cpu_count()
    print(f"Redimensionando con {n_workers} procesos en paralelo...")
    t0 = time.time()
    errors = []
    with Pool(n_workers) as pool:
        for err in tqdm(pool.imap_unordered(resize_one, tasks), total=len(tasks), desc="Redimensionando"):
            if err is not None:
                errors.append(err)
    print(f"Terminado en {time.time() - t0:.1f}s")

    if errors:
        print(f"\nImágenes con error ({len(errors)}):")
        for e in errors[:20]:
            print(" ", e)
    else:
        print("\nSin errores.")

    # Manifiestos: se copian tal cual, no hace falta editar ninguna ruta (ver docstring).
    (DST_DIR / "manifiestos").mkdir(parents=True, exist_ok=True)
    for f in (SRC_DIR / "manifiestos").glob("*.csv"):
        shutil.copy2(f, DST_DIR / "manifiestos" / f.name)
    print(f"Manifiestos copiados a {DST_DIR / 'manifiestos'}")

    size_before = dir_size_gb(SRC_DIR / "images")
    size_after = dir_size_gb(DST_DIR / "images")
    print(f"\nPeso original (dataset_final/images):     {size_before:.2f} GB")
    print(f"Peso redimensionado (dataset_final_224/images): {size_after:.2f} GB")
    print(f"Reducción: {(1 - size_after / size_before) * 100:.1f}%")


if __name__ == "__main__":
    main()

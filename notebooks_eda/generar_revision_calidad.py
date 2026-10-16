"""Genera las hojas de contacto (todas las candidatas, no solo una muestra de 12) y el
archivo `decisiones_calidad.csv` de partida para la revisión manual de cada dataset.

No excluye nada por sí solo. Por cada `resultados/eda/<dataset>/candidatas_baja_calidad.csv`
ya generado por el notebook de EDA de esa fuente, crea:

  resultados/eda/<dataset>/revision_manual/pagina_01.png, pagina_02.png, ...
      Hojas de contacto con TODAS las candidatas (no una muestra), cada una con un número
      de id, la clase y el motivo por el que se marcó — para poder revisarlas visualmente.

  resultados/eda/<dataset>/decisiones_calidad.csv
      Copia de candidatas_baja_calidad.csv + columna `decision`, inicializada en "mantener"
      para todas. Editar a mano solo las filas que, tras ver la hoja de contacto, se quieran
      excluir realmente (cambiar esa fila a "excluir"). El notebook de consolidación respeta
      esta columna.

  resultados/eda/<dataset>/exclusiones_manuales_adicionales.csv
      Plantilla VACÍA (solo encabezado) para los casos que el ojo humano detecta pero el
      algoritmo nunca marcó (p. ej. una radiografía donde no se ven los pulmones, o donde el
      borde es evidente pero no cruzó el umbral). No se intenta que el algoritmo los detecte
      — se agregan a mano, una fila por imagen: path, target_class, motivo (texto libre) y
      decision (queda en "excluir" por defecto, ya que si la estás agregando aquí es porque
      decidiste que no sirve). El notebook de consolidación la lee igual que decisiones_calidad.csv.

Uso: `uv run python generar_revision_calidad.py` desde `notebooks_eda/`. Se puede volver a
correr cuando se quiera (p. ej. al agregar VinDr-CXR) — nunca sobrescribe un archivo de
decisiones que ya exista, solo genera los que falten.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
import eda_utils as eu

RESULTADOS_DIR = Path("../resultados/eda")
PER_PAGE = 20  # 5 columnas x 4 filas


def _loader_for(dataset_name: str):
    if dataset_name in ("RSNA_Pneumonia", "VinDr_CXR"):
        def _dicom_loader(path):
            arr, _ = eu.probe_dicom(path)
            return arr
        return _dicom_loader
    return lambda path: Image.open(path)


def make_contact_sheets(dataset_name: str, df: pd.DataFrame, out_dir: Path):
    loader = _loader_for(dataset_name)
    n_pages = -(-len(df) // PER_PAGE)
    for page in range(n_pages):
        chunk = df.iloc[page * PER_PAGE:(page + 1) * PER_PAGE]
        cols = 5
        rows = -(-len(chunk) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, (_, row) in zip(axes, chunk.iterrows()):
            try:
                ax.imshow(loader(row["path"]), cmap="gray")
            except Exception as e:
                ax.text(0.5, 0.5, f"error:\n{e}", ha="center", va="center")
            title = f"id={row['id']} | {row['target_class']}\n{row['motivo']}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.tight_layout()
        out_path = out_dir / f"pagina_{page + 1:02d}.png"
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  {out_path} ({len(chunk)} imágenes)")


MANUAL_ADDITIONS_COLUMNS = ["path", "target_class", "motivo", "decision"]


def ensure_manual_additions_template(dataset_dir: Path):
    """Plantilla vacía para casos que el ojo humano detecta y el algoritmo nunca marcó."""
    dataset_name = dataset_dir.name
    additions_path = dataset_dir / "exclusiones_manuales_adicionales.csv"
    if additions_path.exists():
        print(f"{dataset_name}: ya existe {additions_path}, no se sobrescribe.")
        return
    pd.DataFrame(columns=MANUAL_ADDITIONS_COLUMNS).to_csv(additions_path, index=False)
    print(f"{dataset_name}: plantilla creada -> {additions_path} "
          f"(agregar ahí, a mano, una fila por cada caso que veas y el algoritmo no marcó; "
          f"columnas: {', '.join(MANUAL_ADDITIONS_COLUMNS)}; decision por defecto 'excluir')")


def process_dataset(candidatas_path: Path):
    dataset_name = candidatas_path.parent.name
    df = pd.read_csv(candidatas_path)
    if len(df) == 0:
        print(f"{dataset_name}: 0 candidatas, nada que revisar.")
        return

    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)
    df["decision"] = "mantener"  # valor por defecto; editar a mano las que se quieran excluir

    out_dir = candidatas_path.parent
    revision_dir = out_dir / "revision_manual"
    revision_dir.mkdir(parents=True, exist_ok=True)

    decisiones_path = out_dir / "decisiones_calidad.csv"
    if decisiones_path.exists():
        print(f"{dataset_name}: ya existe {decisiones_path}, no se sobrescribe (por si ya se editó a mano).")
    else:
        df.to_csv(decisiones_path, index=False)
        print(f"{dataset_name}: {len(df)} candidatas -> {decisiones_path}")

    print(f"{dataset_name}: generando hojas de contacto...")
    make_contact_sheets(dataset_name, df, revision_dir)


def main():
    for candidatas_path in sorted(RESULTADOS_DIR.glob("*/candidatas_baja_calidad.csv")):
        process_dataset(candidatas_path)

    # Plantilla de adiciones manuales para TODOS los datasets ya analizados, incluyendo los
    # que tuvieron 0 candidatas (SilicoData) — el ojo humano puede encontrar algo ahí también.
    for dataset_dir in sorted(p for p in RESULTADOS_DIR.iterdir() if p.is_dir()):
        ensure_manual_additions_template(dataset_dir)


if __name__ == "__main__":
    main()

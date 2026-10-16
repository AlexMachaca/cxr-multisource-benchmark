"""Benchmark de costo computacional de las 10 arquitecturas (parámetros, GMACs, latencia de inferencia).

Mide SOLO el modelo: el tensor de entrada ya está en memoria (en el dispositivo), sin lectura ni decodificación de imágenes, en modo
`inference_mode`, FP32, con calentamiento previo y sincronización de la GPU antes y después de cada medición. Los pesos no importan para la
latencia, por eso los modelos se crean sin preentrenar (misma arquitectura y mismo número de clases que en el entrenamiento: 5).

Protocolo
  - Entrada: tensor aleatorio N(0,1) de (B, 3, 224, 224), B = 1 y B = 32.
  - GPU (CUDA): calentamiento 30 iteraciones (B=1) / 10 (B=32); 200 mediciones (B=1) / 50 (B=32); cada una con torch.cuda.synchronize().
  - CPU: calentamiento 10 / 2; 50 mediciones (B=1) / 8 (B=32); NUM_THREADS hilos.
  - Se reporta la mediana y el rango intercuartílico (Q1-Q3) por iteración. La latencia por imagen es (tiempo del lote) / B.
  - GMACs: operaciones multiplicar-acumular de una imagen (torch.utils.flop_counter.FlopCounterMode / 2; solo convoluciones y
    multiplicaciones matriciales, que son prácticamente todo el costo).
Salida: resultados/costo_computacional/{latencia.csv, entorno.json}. Uso: .venv-gpu/bin/python notebooks_entrenamiento/benchmark_latencia.py
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "resultados" / "costo_computacional"
OUT.mkdir(parents=True, exist_ok=True)
N_CLASSES = 5
NUM_THREADS = 8  # núcleos físicos del Ryzen 7 5800H
MODELS = ["vgg16_bn", "resnet50", "efficientnet_b0", "mobilenetv3_small_100", "vit_base_patch16_224", "deit_base_patch16_224",
          "swin_tiny_patch4_window7_224", "coatnet_0_rw_224", "maxvit_tiny_tf_224", "custom_cnn_transformer"]


def custom_model(num_classes: int) -> nn.Module:
    """Reutiliza la definición del modelo propio tal como está en su cuaderno de entrenamiento (sin duplicar el código)."""
    nb = json.load(open(ROOT / "notebooks_entrenamiento" / "entrenamiento_custom_variantes.ipynb"))
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "class CNNTransformerHybrid" in "".join(c["source"]))
    src = src.split("model = CNNTransformerHybrid")[0]
    ns = {"nn": nn, "torch": torch, "torchvision": torchvision}
    exec(src, ns)
    return ns["CNNTransformerHybrid"](num_classes=num_classes, pretrained=False)


def build(name: str) -> nn.Module:
    m = custom_model(N_CLASSES) if name == "custom_cnn_transformer" else timm.create_model(name, pretrained=False, num_classes=N_CLASSES)
    return m.eval()


def gmacs(model, device) -> float:
    x = torch.randn(1, 3, 224, 224, device=device)
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        model(x)
    return fc.get_total_flops() / 2 / 1e9


def time_model(model, device, batch, warmup, reps) -> tuple[float, float, float]:
    x = torch.randn(batch, 3, 224, 224, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    ts = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(reps):
            sync(); t0 = time.perf_counter()
            model(x)
            sync(); ts.append((time.perf_counter() - t0) * 1000)
    q = np.percentile(ts, [25, 50, 75])
    return float(q[1]), float(q[0]), float(q[2])


def main():
    torch.set_num_threads(NUM_THREADS)
    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    env = {"torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0), "cpu": platform.processor() or "x86_64", "cpu_threads": NUM_THREADS,
           "fecha": time.strftime("%Y-%m-%d %H:%M"), "precision": "FP32", "entrada": "224x224x3"}
    json.dump(env, open(OUT / "entorno.json", "w"), indent=1, ensure_ascii=False)
    rows = []
    for name in MODELS:
        model = build(name)
        n_par = sum(p.numel() for p in model.parameters())
        row = {"modelo": name, "parametros": n_par, "size_mb": n_par * 4 / 1024 ** 2}
        row["GMACs"] = gmacs(model, cpu)
        model_g = model.to(gpu)
        for b, w, r in [(1, 30, 200), (32, 10, 50)]:
            med, q1, q3 = time_model(model_g, gpu, b, w, r)
            row.update({f"gpu_b{b}_ms": med, f"gpu_b{b}_q1": q1, f"gpu_b{b}_q3": q3, f"gpu_b{b}_ms_por_img": med / b})
        model_g.to(cpu); torch.cuda.empty_cache()
        for b, w, r in [(1, 10, 50), (32, 2, 8)]:
            med, q1, q3 = time_model(model, cpu, b, w, r)
            row.update({f"cpu_b{b}_ms": med, f"cpu_b{b}_q1": q1, f"cpu_b{b}_q3": q3, f"cpu_b{b}_ms_por_img": med / b})
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k in ("modelo", "parametros", "GMACs", "gpu_b1_ms", "gpu_b32_ms_por_img", "cpu_b1_ms", "cpu_b32_ms_por_img")}, flush=True)
        del model, model_g
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "latencia.csv", index=False)


if __name__ == "__main__":
    main()

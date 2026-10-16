# Clasificación conjunta de enfermedades pulmonares infecciosas y ocupacionales en radiografías de tórax

*English version: [README.md](README.md).*

Código y resultados agregados del artículo *Joint Classification of Infectious and Occupational Lung Diseases on Chest X-Rays: A Benchmark of
Convolutional, Transformer, and Hybrid Networks* (Machaca-Chuquimamani e Ibarra-Cabrera, enviado a *BioMedInformatics*, MDPI).
<!-- TODO al publicar: agregar el DOI del artículo y el del archivo de Zenodo -->

El estudio compara diez arquitecturas (4 CNN, 3 Transformers, 2 híbridas y un modelo propio CNN–Transformer) sobre un dataset curado que reúne siete
fuentes públicas de radiografías de tórax en seis clases (Normal, Neumonía, COVID-19, Tuberculosis, Silicosis y silicotuberculosis), con un protocolo
común de aprendizaje por transferencia en dos fases, cuatro variantes del dataset, estadística pareada, variabilidad entre corridas, evaluación por
fuente y análisis de oclusión y Grad-CAM.

> **Qué NO contiene este repositorio.** No se distribuyen radiografías, pesos entrenados, manifiestos por imagen ni predicciones por imagen: dos de las
> fuentes (SilicoData y VinDr-CXR) prohíben redistribuir los datos o el material derivado de ellos. En `docs/DATA_ACCESS.md` se indica cómo obtener los datos
> de sus proveedores originales. Todo lo demás (código, cuadernos, métricas agregadas) sí está aquí.

## Estructura
Los nombres de las carpetas son los del proyecto original (en español) para que todos los scripts funcionen sin cambiar rutas.

| Carpeta | Contenido |
|---|---|
| `notebooks_eda/` | Exploración y curación de cada fuente (`00`–`06`), cruce de hashes perceptuales entre fuentes (`07`) y construcción del dataset final y sus manifiestos (`08`). **Se borraron las salidas** porque incrustan radiografías. |
| `notebooks_entrenamiento/` | Cuadernos de entrenamiento de las 10 arquitecturas (`entrenamiento_<modelo>_variantes.ipynb`, los 40 entrenamientos principales; Google Colab, Tesla T4), `entrenamiento_<modelo>.ipynb` (primer entrenamiento de la variante de seis clases con exclusión, «corrida A»), `entrenamiento_resnet50_radimagenet.ipynb`, `entrenamiento_utils.py`, `preresize_dataset.py`, los cuadernos de análisis (`analisis_*`, `comparacion_modelos`) y `benchmark_latencia.py`. |
| `notebooks_app/` | `gradcam_clasificadores.py` y `explicabilidad.py`: pruebas de oclusión y Grad-CAM. |
| `figuras/` | Scripts que generan las figuras del artículo (Fig. 1–2 y 3–10) a partir de los archivos de resultados. |
| `Redaccion/manuscrito/generar_tablas_resultados.py` | Genera las Tablas 8–13 (LaTeX) a partir de los archivos de resultados. |
| `resultados/` | Solo resultados agregados (CSV/JSON): métricas por modelo, matrices de confusión (conteos), intervalos bootstrap, pruebas de McNemar, descomposición del efecto de la STB, análisis de réplicas, tablas por fuente, tablas de oclusión y latencia. |

## Entorno
- Entrenamiento (Google Colab): Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, timm 1.0.30, CUDA 13.0, Tesla T4.
- Análisis, latencia, oclusión y Grad-CAM (local): Python 3.12, PyTorch 2.14, torchvision 0.29, timm 1.0.29–1.0.30.
- Entorno de análisis en CPU: `uv sync` (usa `pyproject.toml`) o `pip install -r requirements.txt`; GPU: `pip install -r requirements-gpu.txt`.

## Cómo reproducir los resultados
1. Obtener los datasets (`docs/DATA_ACCESS.md`) y colocarlos en `Datasets/`.
2. Curación: ejecutar `notebooks_eda/00` … `08` en orden (semilla 42). Esto crea `dataset_final/` y los manifiestos.
3. `python notebooks_entrenamiento/preresize_dataset.py` crea `dataset_final_224/` (224×224, bilineal; píxeles idénticos a redimensionar durante el entrenamiento).
4. Entrenamiento: ejecutar los cuadernos de `notebooks_entrenamiento/` (esperan la estructura de carpetas del proyecto original; las rutas están en las primeras celdas).
5. Análisis: `analisis_4_variantes`, `analisis_por_fuente`, `analisis_replica_radimagenet`; luego
   `python figuras/figuras_resultados/generar_figuras_resultados.py` y `python Redaccion/manuscrito/generar_tablas_resultados.py`.

Para comprobar las cifras del artículo sin reentrenar bastan los archivos agregados de `resultados/`: `generar_tablas_resultados.py` y
`generar_figuras_resultados.py` se ejecutan a partir de ellos. Los cuadernos de análisis (`analisis_*`) necesitan las predicciones por imagen
(`*_predicciones.npz`) que escriben los cuadernos de entrenamiento y que no se distribuyen; sus salidas guardadas se conservan en los cuadernos.

El entrenamiento se hizo **sin semilla aleatoria fija** (ver el artículo, Secciones 3.9 y 4.7, donde se mide la variabilidad entre corridas), así que un
nuevo entrenamiento dará resultados que difieren de los del artículo en una magnitud similar a la allí reportada.

## Elementos del artículo → archivos
| Artículo | Archivos |
|---|---|
| Tablas 8–12 | `resultados/comparacion_variantes/`, `resultados/comparacion_replicas/`, generadas por `Redaccion/manuscrito/generar_tablas_resultados.py` |
| Tabla 13 | `resultados/costo_computacional/latencia.csv` (`notebooks_entrenamiento/benchmark_latencia.py`) |
| Figuras 3–8 | `figuras/figuras_resultados/generar_figuras_resultados.py` |
| Figuras 9–10 | `resultados/analisis_por_fuente/`, `resultados/gradcam_clasificadores/` |
| Métricas por clase de cada modelo | `resultados/resultados_entrenamiento/<variante>/<modelo>/` |

## Notas
- Los comentarios de los cuadernos y scripts están mayormente en español.
- `gradcam_clasificadores.py` escribe en `resultados/gradcam_clasificadores/INTERNO_*` archivos de imagen derivados de SilicoData; **no deben publicarse ni subirse**.
- Este código no es un dispositivo médico y no debe usarse para diagnosticar.

## Licencia y cita
Código: licencia MIT (ver `LICENSE`). Los datasets conservan sus propias licencias y condiciones de uso. Si usa este código, cite el artículo (ver `CITATION.cff`).

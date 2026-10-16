# Figuras del artículo

Las carpetas `figura_pipeline/` y `figura_arquitectura/` son las **versiones originales (v1)** y no se modifican.
Las versiones nuevas (v2) están en carpetas aparte y son las que usa el manuscrito (`Redaccion/manuscrito/figuras/`).

| Figura | Carpeta | Archivos | Script |
|---|---|---|---|
| 1. Pipeline del estudio | `figura_pipeline_v2/` | `figura_pipeline_v2_{EN,ES}.{svg,pdf,png}` | `generar_figura_pipeline_v2.py` |
| 2. Arquitectura del modelo propio | `figura_arquitectura_v2/` | `figura_arquitectura_v2_{EN,ES}.{svg,pdf,png}` | `generar_figura_arquitectura_v2.py` |
| 3. Macro-F1 con IC 95% (bootstrap) | `figuras_resultados/` | `fig3_bootstrap_macro_f1_{EN,ES}` | `generar_figuras_resultados.py` |
| 4. Macro-F1 por familia | `figuras_resultados/` | `fig4_f1_por_familia_{EN,ES}` | ídem |
| 5. Matrices de confusión (mejor modelo por variante) | `figuras_resultados/` | `fig5_matrices_confusion_{EN,ES}` | ídem |
| 6. Efecto de la clase STB | `figuras_resultados/` | `fig6_efecto_stb_{EN,ES}` | ídem |
| 7. Efecto de la exclusión por calidad | `figuras_resultados/` | `fig7_efecto_exclusion_calidad_{EN,ES}` | ídem |
| 8. Macro-F1 vs. parámetros | `figuras_resultados/` | `fig8_f1_vs_parametros_{EN,ES}` | ídem |

**Formatos:** PDF vectorial (el que se inserta en LaTeX) y PNG a 600 dpi para 17.5 cm de ancho. Las figuras 1 y 2 también
se generan en SVG editable.

## Regenerar

```bash
.venv/bin/python figuras/figura_pipeline_v2/generar_figura_pipeline_v2.py
.venv/bin/python figuras/figura_arquitectura_v2/generar_figura_arquitectura_v2.py
.venv/bin/python figuras/figuras_resultados/generar_figuras_resultados.py
cp figuras/figura_pipeline_v2/*.pdf figuras/figura_arquitectura_v2/*.pdf Redaccion/manuscrito/figuras/
```

**Cuando se reentrene la variante de 6 clases con exclusión:** primero re-ejecutar
`notebooks_entrenamiento/analisis_4_variantes.ipynb` (regenera los CSV de `resultados/comparacion_variantes/`) y después
`generar_figuras_resultados.py`. El mejor modelo de la Figura 5 se elige automáticamente (el de mayor macro-F1).

## Criterios de diseño (comunes a todas)

- **Paleta neutra:** texto gris pizarra, fondo blanco, sin degradados ni sombras.
  - Familias: CNN `#4A6A93` · Transformer `#5E8F72` · Híbrido `#B8873F` · Propio `#B596B8`.
  - Clases: COVID-19 `#A65B4E` · Neumonía `#C29A4B` · Normal `#6E9C7F` · TB `#3D5A80` · Silicosis `#B3A6CC` · STB `#4E868E`.
  - Comprobada con simulación de deuteranopía y protanopía: el par de colores más parecido tiene ΔE ≥ 18.
- **Sin título dentro de la imagen:** el título y la explicación van en la leyenda (*caption*) del artículo. Los paneles se marcan (a), (b).
- **Todo el texto en inglés** en la versión EN (requisito de MDPI) y en español en la versión ES (solo para revisión interna).
- **Radiografías reales:** solo de Kermany et al. (Mendeley Data, **CC BY 4.0**), en `_recursos_cxr/`.
  No se usan imágenes de SilicoData (DUA), VinDr-CXR (PhysioNet) ni Shenzhen/Montgomery (su ReadMe pide no compartir el
  dataset). La atribución va en la leyenda de cada figura que las use; ver `_recursos_cxr/ATRIBUCION.md`.
- **Cifras leídas de los datos** (manifiestos, CSV, predicciones o el propio modelo), con verificaciones automáticas
  (`assert`) para que una figura nunca muestre un número que no coincida con el texto.

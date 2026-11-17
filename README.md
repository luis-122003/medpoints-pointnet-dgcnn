# Clasificación multiclase de estructuras anatómicas en nubes de puntos: PointNet frente a DGCNN sobre MedPointS

Proyecto final de la Electiva de Ingeniería de Software (Universidad de San Buenaventura Cali).
Autor: Luis Edwar Mosquera Congo.

Clasificación de nubes de puntos anatómicas en 46 clases (órganos, huesos, vasos y músculos) con el conjunto
**MedPointS-cls** ([Hugging Face `wlsdzyzl/MedPointS-cls`](https://huggingface.co/datasets/wlsdzyzl/MedPointS-cls),
licencia MIT; 28.737 nubes de 2.048 puntos (x, y, z) en float32, derivadas de MedShapeNet).

- **Modelos:** PointNet (3,48 M de parámetros, 2.048 puntos por nube) y DGCNN (1,81 M, 1.024 puntos, grafo de 20 vecinos).
- **Partición:** estratificada 60/20/20 con semilla fija 2026 (17.249 / 5.744 / 5.744), guardada en `codigo/particion.npz`.
- **Protocolo:** 60 épocas, Adam 1e-3 con decaimiento coseno, lote 32, entropía cruzada ponderada por clase, aumento sin
  rotaciones ni espejos (escala, traslación y ruido), selección por F1 macro en validación (en empate, menor entropía
  cruzada), tres semillas (0, 1 y 2) y una sola evaluación en prueba, con intervalos de confianza por bootstrap.

## Resultados

Los resultados del entrenamiento en Kaggle (GPU Tesla T4) se agregan en `resultados/` al terminar la corrida.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `codigo/` | `medpoints_cls.py` (datos, PointNet, DGCNN, entrenamiento, métricas y figuras), `figuras_metodo.py` y `particion.npz` |
| `notebooks/` | `MedPointS_Kaggle.ipynb`, el cuaderno que se ejecuta en Kaggle |
| `figuras/` | Flujo de datos, arquitecturas, una nube por clase y distribución de clases |

## Reproducir

```bash
pip install pyarrow pandas torch scikit-learn matplotlib
python codigo/medpoints_cls.py --modelo pointnet --puntos 2048 --epocas 60 --semillas 0 1 2
python codigo/medpoints_cls.py --modelo dgcnn --puntos 1024 --epocas 60 --semillas 0 1 2
python codigo/medpoints_cls.py --modelo pointnet --epocas 1 --semillas 0 --max-muestras 600   # prueba rápida
```

El script descarga los dos archivos parquet de Hugging Face si no están en `medpoints_datos/`.

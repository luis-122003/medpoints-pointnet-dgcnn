# Clasificación multiclase de estructuras anatómicas en nubes de puntos: PointNet frente a DGCNN sobre MedPointS

Proyecto final de la Electiva de Ingeniería de Software (Universidad de San Buenaventura Cali).
Autor: Luis Edwar Mosquera Congo.

Clasificación de nubes de puntos anatómicas en 46 clases (órganos, huesos, vasos y músculos) con el conjunto
**MedPointS-cls** ([Hugging Face `wlsdzyzl/MedPointS-cls`](https://huggingface.co/datasets/wlsdzyzl/MedPointS-cls),
licencia MIT; 28.737 nubes de 2.048 puntos (x, y, z) en float32, derivadas de MedShapeNet).

- **Modelos:** PointNet (3,48 M de parámetros, 2.048 puntos por nube) y DGCNN (1,81 M, 1.024 puntos, grafo de 20 vecinos).
- **Partición:** estratificada 60/20/20 con semilla fija 2026 (17.249 / 5.744 / 5.744), guardada en `codigo/particion.npz`.
- **Protocolo:** hasta 50 épocas con parada temprana (10 épocas sin mejorar en validación), Adam 1e-3 con decaimiento coseno, lote 32, entropía cruzada ponderada por clase, aumento sin
  rotaciones ni espejos (escala, traslación y ruido), selección por F1 macro en validación (en empate, menor entropía
  cruzada), tres semillas (0, 1 y 2) y una sola evaluación en prueba, con intervalos de confianza por bootstrap.

## Resultados

Conjunto de prueba (5.744 nubes), media ± desviación de tres semillas, GPU Tesla T4 en Kaggle
([MedPointS_Kaggle](https://www.kaggle.com/code/luismc15/medpoints-kaggle), versión 2):

| Modelo | Exactitud | Exactitud balanceada | F1 macro | AUC OvR | Memoria GPU (GB) | s/época |
|---|---|---|---|---|---|---|
| PointNet (2.048 puntos) | 0,942 ± 0,001 | 0,918 ± 0,002 | 0,914 ± 0,002 | 0,998 ± 0,000 | 1,371 ± 0,016 | 36,7 ± 0,1 |
| DGCNN (1.024 puntos) | 0,951 ± 0,002 | 0,929 ± 0,001 | 0,926 ± 0,002 | 0,999 ± 0,000 | 2,876 ± 0,009 | 169,6 ± 0,1 |

DGCNN gana en las tres semillas, por 0,009 de exactitud, a cambio de 4,6 veces más tiempo por época. Líneas base publicadas en
MedPointS (Zhang et al., MICCAI 2025; otra partición y una sola corrida): PointNet 0,936, PointNet++ 0,935, DGCNN 0,940,
PCT 0,929, PointMamba 0,922. Ninguna corrida activó la parada temprana; las mejores épocas fueron 42, 43 y 48 (PointNet) y
42, 47 y 49 (DGCNN). Los valores salen de `resultados/resultados.csv`.

## Contenido

| Carpeta | Qué hay |
|---|---|
| `codigo/` | `medpoints_cls.py` (datos, PointNet, DGCNN, entrenamiento, métricas y figuras), `figuras_metodo.py` y `particion.npz` |
| `notebooks/` | `MedPointS_Kaggle.ipynb`, el cuaderno que se ejecuta en Kaggle, y `MedPointS_Kaggle_ejecutado.ipynb`, con todas sus salidas |
| `resultados/` | `resultados.csv`, métricas por clase, historiales, matrices de confusión, figuras, logs, entorno y los pesos `modelo_{pointnet,dgcnn}_seed{0,1,2}.pt` |
| `figuras/` | Flujo de datos, arquitecturas, una nube por clase y distribución de clases |

## Reproducir

```bash
pip install pyarrow pandas torch scikit-learn matplotlib
python codigo/medpoints_cls.py --modelo pointnet --puntos 2048 --epocas 50 --paciencia 10 --semillas 0 1 2
python codigo/medpoints_cls.py --modelo dgcnn --puntos 1024 --epocas 50 --paciencia 10 --semillas 0 1 2
python codigo/medpoints_cls.py --modelo pointnet --epocas 1 --semillas 0 --max-muestras 600   # prueba rápida
```

El script descarga los dos archivos parquet de Hugging Face si no están en `medpoints_datos/`.

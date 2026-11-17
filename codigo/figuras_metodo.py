"""Figuras 1 (flujo de datos) y 3 (arquitecturas) del informe, en el mismo estilo de cajas que las de los avances.
Uso: python figuras_metodo.py <carpeta_salida>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AZUL, BORDE_AZUL = "#e8eef8", "#2f5aa6"
VERDE, BORDE_VERDE = "#e8f4e8", "#2e8b3a"
AMBAR, BORDE_AMBAR = "#fcebc9", "#b8860b"
GRIS, BORDE_GRIS = "#f2f2f2", "#666666"
out = sys.argv[1]


def caja(ax, x, y, w, h, texto, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs)


def flecha(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.6))


# Figura 1: flujo de manejo de datos
fig, ax = plt.subplots(figsize=(20, 5.8))
pasos = ["MedPointS-cls\n(Hugging Face, parquet)\n28.737 nubes", "Verificación\nforma (N, 2.048, 3)\nfloat32, 46 clases",
         "Partición estratificada\n60 / 20 / 20\n(semilla 2026)", "Muestreo y aumento\n(solo entrenamiento):\nescala, traslación, ruido",
         "Entrada\n(3, P) con P = 2.048\n(PointNet) o 1.024 (DGCNN)"]
w, h, g, y = 3.3, 2.2, 0.5, 1.8
for i, p in enumerate(pasos):
    x = 0.2 + i * (w + g)
    caja(ax, x, y, w, h, p, AZUL, BORDE_AZUL, 13)
    flecha(ax, x + w + 0.05, x + w + g - 0.05, y + h / 2)
x = 0.2 + len(pasos) * (w + g)
caja(ax, x, y, 2.9, h, "Modelo:\nexactitud, F1 macro,\nAUC OvR", AMBAR, BORDE_AMBAR, 13)
ax.set_xlim(0, x + 3.1); ax.set_ylim(0.5, 5.3); ax.axis("off")
fig.savefig(os.path.join(out, "fig_flujo_medpoints.png"), dpi=100, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# Figura 3: arquitecturas
fig, ax = plt.subplots(figsize=(20, 7.2))
fila_pn = ["Entrada\n(3, 2.048)", "T-Net 3×3", "MLP compartido\n64-64", "T-Net 64×64",
           "MLP compartido\n64-128-1.024", "Max-pooling\nglobal", "FC 512-256\n46 + softmax"]
fila_dg = ["Entrada\n(3, 1.024)", "EdgeConv 64\nk-NN (k = 20)", "EdgeConv 64", "EdgeConv 128", "EdgeConv 256",
           "Conv1d 1.024\nmax + avg pool", "FC 512-256\n46 + softmax"]
for fila, y0, fc, ec, titulo in ((fila_pn, 4.3, AZUL, BORDE_AZUL, "PointNet (arriba): 3,48 M parámetros"),
                                  (fila_dg, 0.6, VERDE, BORDE_VERDE, "DGCNN (abajo): 1,81 M parámetros")):
    ax.text(0.2, y0 + 2.2, titulo, fontsize=15, fontweight="bold", va="bottom")
    w, h, g = 2.25, 1.75, 0.55
    for i, p in enumerate(fila):
        x = 0.2 + i * (w + g)
        c = (GRIS, BORDE_GRIS) if i == 0 else ((AMBAR, BORDE_AMBAR) if i == len(fila) - 1 else (fc, ec))
        caja(ax, x, y0, w, h, p, *c, 13)
        if i < len(fila) - 1:
            flecha(ax, x + w + 0.05, x + w + g - 0.05, y0 + h / 2)
ax.set_xlim(0, 0.2 + 7 * 2.8); ax.set_ylim(0.3, 7.0); ax.axis("off")
fig.savefig(os.path.join(out, "fig_arquitecturas_medpoints.png"), dpi=100, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("listo")

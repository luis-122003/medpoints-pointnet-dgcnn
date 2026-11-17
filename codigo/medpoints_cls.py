"""MedPointS-cls: clasificación multiclase de 46 estructuras anatómicas a partir de nubes de puntos (PointNet y DGCNN).

Uso:
    pip install pyarrow pandas torch scikit-learn matplotlib
    python medpoints_cls.py --modelo pointnet --epocas 60 --semillas 0 1 2
    python medpoints_cls.py --modelo dgcnn --puntos 1024 --epocas 60 --semillas 0 1 2
    python medpoints_cls.py --modelo pointnet --epocas 1 --semillas 0 --max-muestras 600   # prueba rápida

Datos: Hugging Face wlsdzyzl/MedPointS-cls (licencia MIT), 28.737 nubes de 2.048 puntos (x, y, z) en float32 y 46 clases.
Partición: estratificada 60 / 20 / 20 (entrenamiento / validación / prueba) con semilla fija 2026, igual para todos los
modelos y semillas; se guarda en particion.npz.

Genera (prefijo = nombre del modelo): resultados.csv (se van agregando filas), historial_<m>_seed*.csv,
metricas_por_clase_<m>_seed*.csv, matrices_confusion_<m>.npz, modelo_<m>_seed*.pt, fig_curvas_<m>.png,
fig_confusion_<m>.png, fig_predicciones_<m>.png, y una vez: fig_distribucion_clases.png, fig_muestra_nubes.png.
"""
import argparse
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CLASES = ["adrenalgland", "aorta", "autochthon", "bladder", "brain", "breast", "bronchie", "celiactrunk", "cheek",
          "clavicle", "colon", "costa", "duodenum", "esophagus", "eyeball", "femur", "gallbladder", "gluteusmaximus",
          "heart", "hip", "humerus", "iliacartery", "iliacvena", "iliopsoas", "inferiorvenacava", "kidney", "liver",
          "lung", "mediastinaltissue", "pancreas", "portalveinandsplenicvein", "smallbowel", "spleen", "stomach",
          "thymus", "thyroid", "trachea", "uterocervix", "uterus", "vertebrae", "gonads", "sacrum", "clavicula",
          "pulmonaryartery", "rib", "scapula"]
N_CLASES = len(CLASES)
URLS = [f"https://huggingface.co/api/datasets/wlsdzyzl/MedPointS-cls/parquet/default/train/{i}.parquet" for i in (0, 1)]
SEMILLA_PARTICION = 2026


# ------------------------------------------------------------------ datos
def descargar(carpeta):
    os.makedirs(carpeta, exist_ok=True)
    rutas = []
    for i, u in enumerate(URLS):
        r = os.path.join(carpeta, f"{i}.parquet")
        if not os.path.exists(r):
            print("descargando", u, flush=True)
            urllib.request.urlretrieve(u, r)
        rutas.append(r)
    return rutas


def cargar(rutas, max_muestras=None):
    """Devuelve X float32 (N, P, 3) e y int64 (N,)."""
    import pyarrow.parquet as pq
    Xs, ys = [], []
    for r in rutas:
        t = pq.read_table(r)
        col = t.column("data").combine_chunks()
        n = len(col)
        pts = col.flatten()                       # lista de puntos
        xyz = pts.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        P = len(pts) // n
        Xs.append(xyz.reshape(n, P, -1))
        lab = t.column("label").combine_chunks()
        lv = lab.flatten().to_numpy(zero_copy_only=False)
        k = len(lv) // n
        lv = lv.reshape(n, k)
        ys.append((lv[:, 0] if k == 1 else lv.argmax(1)).astype(np.int64))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if max_muestras:
        rng = np.random.default_rng(0)
        i = rng.permutation(len(y))[:max_muestras]
        X, y = X[i], y[i]
    assert X.ndim == 3 and X.shape[2] == 3, X.shape
    assert y.min() >= 0 and y.max() < N_CLASES, (y.min(), y.max())
    return X, y


def particion(y, archivo="particion.npz"):
    """Estratificada 60/20/20 con semilla fija; se reutiliza si ya existe."""
    if os.path.exists(archivo):
        d = np.load(archivo)
        if sum(len(d[k]) for k in d.files) == len(y):
            return {k: d[k] for k in ("train", "val", "test")}
    rng = np.random.default_rng(SEMILLA_PARTICION)
    idx = {"train": [], "val": [], "test": []}
    for c in range(N_CLASES):
        ic = rng.permutation(np.where(y == c)[0])
        n = len(ic)
        n_te, n_va = int(round(0.2 * n)), int(round(0.2 * n))
        if n >= 3:
            n_te, n_va = max(n_te, 1), max(n_va, 1)
        idx["test"] += ic[:n_te].tolist()
        idx["val"] += ic[n_te:n_te + n_va].tolist()
        idx["train"] += ic[n_te + n_va:].tolist()
    out = {k: np.array(sorted(v), dtype=np.int64) for k, v in idx.items()}
    np.savez(archivo, **out)
    return out


def resumen(X, y, part):
    print(f"X={X.shape} dtype={X.dtype} min={X.min():.3f} max={X.max():.3f}")
    for k in ("train", "val", "test"):
        print(f"{k:5s} n={len(part[k])} clases presentes={len(np.unique(y[part[k]]))}")


class Nubes(Dataset):
    def __init__(self, X, y, idx, puntos, aumento=False, seed=0):
        self.X, self.y, self.idx, self.p, self.aug = X, y, idx, puntos, aumento
        self.rng = np.random.default_rng(seed)
        # para evaluación: subconjunto fijo de puntos por muestra (reproducible)
        self.fijo = np.random.default_rng(123).permutation(X.shape[1])[:puntos]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        if self.aug:
            sel = self.rng.choice(self.X.shape[1], self.p, replace=False) if self.p < self.X.shape[1] else self.rng.permutation(self.X.shape[1])
            v = self.X[j, sel].copy()
            # sin rotaciones ni espejos: la orientación anatómica es informativa
            v = v * self.rng.uniform(0.8, 1.25, size=(1, 3)).astype(np.float32)          # escala anisotrópica
            v = v + self.rng.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)          # traslación
            v = v + np.clip(self.rng.normal(0, 0.01, v.shape), -0.05, 0.05).astype(np.float32)  # ruido
        else:
            v = self.X[j, self.fijo]
        return torch.from_numpy(np.ascontiguousarray(v.T, dtype=np.float32)), int(self.y[j])   # (3, P)


def pesos_clase(y):
    cnt = np.bincount(y, minlength=N_CLASES).astype(np.float32)
    return torch.tensor(cnt.sum() / (N_CLASES * np.maximum(cnt, 1)))


# ------------------------------------------------------------------ modelos
class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, k * k))
        nn.init.zeros_(self.fc[-1].weight); nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        t = self.fc(self.conv(x).max(2)[0]).view(-1, self.k, self.k)
        return t + torch.eye(self.k, device=x.device, dtype=t.dtype)


class PointNet(nn.Module):
    """Qi et al. (2017): T-Net de entrada y de características, MLP compartido, max-pooling global."""

    def __init__(self, n_clases=N_CLASES):
        super().__init__()
        self.t1 = TNet(3)
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.t2 = TNet(64)
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, n_clases))
        self.reg = None

    def forward(self, x):
        x = torch.bmm(self.t1(x).transpose(1, 2), x)
        x = self.mlp1(x)
        t2 = self.t2(x)
        x = torch.bmm(t2.transpose(1, 2), x)
        eye = torch.eye(64, device=x.device, dtype=t2.dtype)
        self.reg = ((torch.bmm(t2, t2.transpose(1, 2)) - eye) ** 2).sum((1, 2)).mean()
        return self.head(self.mlp2(x).max(2)[0])


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def grafo(x, k):
    B, C, N = x.shape
    idx = knn(x.float(), k) + torch.arange(B, device=x.device).view(-1, 1, 1) * N
    xt = x.transpose(2, 1).contiguous()
    vec = xt.view(B * N, -1)[idx.view(-1)].view(B, N, k, C)
    xt = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat((vec - xt, xt), 3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    """Wang et al. (2019): cuatro EdgeConv con grafo k-NN dinámico (k = 20), max y avg pooling global."""

    def __init__(self, n_clases=N_CLASES, k=20, emb=1024):
        super().__init__()
        self.k = k
        bloque = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.c1, self.c2, self.c3, self.c4 = bloque(6, 64), bloque(128, 64), bloque(128, 128), bloque(256, 256)
        self.c5 = nn.Sequential(nn.Conv1d(512, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Linear(emb * 2, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                  nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                  nn.Linear(256, n_clases))
        self.reg = None

    def forward(self, x):
        x1 = self.c1(grafo(x, self.k)).max(-1)[0]
        x2 = self.c2(grafo(x1, self.k)).max(-1)[0]
        x3 = self.c3(grafo(x2, self.k)).max(-1)[0]
        x4 = self.c4(grafo(x3, self.k)).max(-1)[0]
        x = self.c5(torch.cat((x1, x2, x3, x4), 1))
        return self.head(torch.cat((x.max(2)[0], x.mean(2)), 1))


MODELOS = {"pointnet": PointNet, "dgcnn": DGCNN}


def n_param(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ entrenamiento y métricas
@torch.no_grad()
def predecir(m, dl, dev):
    m.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
            P.append(F.softmax(m(x.to(dev)).float(), 1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def metricas(P, Y):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    pred = P.argmax(1)
    presentes = np.unique(Y)
    try:
        Pp = P[:, presentes] / np.maximum(P[:, presentes].sum(1, keepdims=True), 1e-12)
        auc = roc_auc_score(Y, Pp, multi_class="ovr", average="macro", labels=presentes) if len(presentes) > 2 else float("nan")
    except ValueError:
        auc = float("nan")
    nll = float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None)).mean())
    return {"acc": accuracy_score(Y, pred), "acc_balanceada": balanced_accuracy_score(Y, pred),
            "f1_macro": f1_score(Y, pred, average="macro", labels=presentes), "auc_ovr": auc, "nll": nll,
            "cm": confusion_matrix(Y, pred, labels=list(range(N_CLASES)))}


def bootstrap(P, Y, B=2000, seed=0):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    pred = P.argmax(1); n = len(Y); A, Fm = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        A.append((pred[i] == Y[i]).mean())
        Fm.append(f1_score(Y[i], pred[i], average="macro", labels=np.unique(Y[i])))
    return np.percentile(A, [2.5, 97.5]), np.percentile(Fm, [2.5, 97.5])


def entrenar(nombre, X, y, part, puntos, epocas, lr, batch, seed, dev):
    """Adam + coseno, entropía cruzada ponderada, precisión mixta en GPU. Selección: mejor F1 macro en validación
    (empate: menor entropía cruzada); prueba una sola vez. Memoria pico medida desde la 2.ª época."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(Nubes(X, y, part["train"], puntos, True, seed), batch_size=batch, shuffle=True, generator=g,
                       drop_last=True, num_workers=2, persistent_workers=True)
    dl_va = DataLoader(Nubes(X, y, part["val"], puntos), batch_size=64, num_workers=2)
    dl_te = DataLoader(Nubes(X, y, part["test"], puntos), batch_size=64, num_workers=2)
    m = MODELOS[nombre]().to(dev)
    w = pesos_clase(y[part["train"]]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    scaler = torch.amp.GradScaler(enabled=(dev == "cuda"))
    mejor, estado, mejor_ep, hist = (-1.0, -float("inf")), None, 0, []
    t0 = time.time()
    for ep in range(epocas):
        m.train(); suma, n = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, enabled=(dev == "cuda")):
                out = m(xb)
            loss = F.cross_entropy(out.float(), yb, weight=w)
            if m.reg is not None:
                loss = loss + 0.001 * m.reg.float()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            suma += loss.item() * len(yb); n += len(yb)
        sch.step()
        if ep == 0 and dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = metricas(*predecir(m, dl_va, dev))
        hist.append({"epoca": ep + 1, "loss_train": suma / n, "val_loss": r["nll"], "val_acc": r["acc"], "val_f1_macro": r["f1_macro"]})
        clave = (round(r["f1_macro"], 6), -r["nll"])
        if clave > mejor:
            mejor, mejor_ep = clave, ep + 1
            estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"{nombre} seed {seed} ep {ep+1:3d} loss {suma/n:.3f} val loss {r['nll']:.3f} acc {r['acc']:.3f} F1 {r['f1_macro']:.3f} "
              f"({time.time()-t0:.0f} s)", flush=True)
    s_ep = (time.time() - t0) / epocas
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    print(f"{nombre} seed {seed}: mejor época {mejor_ep} (val F1 {mejor[0]:.3f}, val loss {-mejor[1]:.3f})", flush=True)
    m.load_state_dict(estado)
    P, Y = predecir(m, dl_te, dev)
    return m, metricas(P, Y), hist, mejor_ep, s_ep, mem, (P, Y)


def reporte_por_clase(cm, archivo):
    import pandas as pd
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    pd.DataFrame({"clase": CLASES, "soporte": cm.sum(1), "precision": prec, "recall": rec, "f1": f1}).round(3).to_csv(archivo, index=False)


# ------------------------------------------------------------------ figuras
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def graficas_datos(X, y, part):
    plt = _plt()
    if not os.path.exists("fig_distribucion_clases.png"):
        cnt = {k: np.bincount(y[part[k]], minlength=N_CLASES) for k in part}
        x = np.arange(N_CLASES); w = 0.28
        plt.figure(figsize=(14, 3.8))
        for i, k in enumerate(("train", "val", "test")):
            plt.bar(x + (i - 1) * w, cnt[k], w, label=k)
        plt.xticks(x, CLASES, rotation=75, ha="right", fontsize=7); plt.ylabel("nubes"); plt.legend(); plt.tight_layout()
        plt.savefig("fig_distribucion_clases.png", dpi=200); plt.close()
    if not os.path.exists("fig_muestra_nubes.png"):
        fig = plt.figure(figsize=(16, 9))
        for c in range(N_CLASES):
            ic = np.where(y == c)[0]
            ax = fig.add_subplot(5, 10, c + 1, projection="3d")
            if len(ic):
                v = X[ic[0]]
                ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.4, c=v[:, 2], cmap="viridis")
            ax.set_title(CLASES[c], fontsize=6); ax.set_axis_off()
        plt.tight_layout(); plt.savefig("fig_muestra_nubes.png", dpi=200); plt.close()


def figura_curvas(nombre, historiales):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    for seed, h in historiales.items():
        e = [r["epoca"] for r in h]
        ax[0].plot(e, [r["loss_train"] for r in h], label=f"semilla {seed}")
        ax[1].plot(e, [r["val_loss"] for r in h], label=f"semilla {seed}")
        ax[2].plot(e, [r["val_f1_macro"] for r in h], label=f"semilla {seed}")
    for a, t in zip(ax, ("pérdida de entrenamiento", "entropía cruzada en validación", "F1 macro en validación")):
        a.set_xlabel("época"); a.set_ylabel(t); a.legend()
    plt.tight_layout(); plt.savefig(f"fig_curvas_{nombre}.png", dpi=200); plt.close()


def figura_confusion(nombre, cm):
    plt = _plt()
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(11, 10)); plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(N_CLASES), CLASES, rotation=90, fontsize=6); plt.yticks(range(N_CLASES), CLASES, fontsize=6)
    plt.xlabel("predicha"); plt.ylabel("real"); plt.colorbar(fraction=0.046); plt.tight_layout()
    plt.savefig(f"fig_confusion_{nombre}.png", dpi=200); plt.close()


def figura_predicciones(nombre, X, part, P, Y):
    plt = _plt()
    rng = np.random.default_rng(0)
    clases = [c for c in range(N_CLASES) if np.any(Y == c)]
    elegidas = rng.choice(clases, size=min(12, len(clases)), replace=False)
    fig = plt.figure(figsize=(16, 7))
    for k, c in enumerate(elegidas):
        i = int(rng.choice(np.where(Y == c)[0]))
        v = X[part["test"][i]]
        ax = fig.add_subplot(2, 6, k + 1, projection="3d")
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.4, c=v[:, 2], cmap="viridis"); ax.set_axis_off()
        p = int(P[i].argmax())
        ax.set_title(f"real: {CLASES[Y[i]]}\npred: {CLASES[p]} ({P[i].max():.2f})", fontsize=8,
                     color="green" if p == Y[i] else "red")
    plt.tight_layout(); plt.savefig(f"fig_predicciones_{nombre}.png", dpi=200); plt.close()


# ------------------------------------------------------------------ programa principal
def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=list(MODELOS), required=True)
    ap.add_argument("--datos", default="medpoints_datos", help="carpeta con 0.parquet y 1.parquet (se descargan si faltan)")
    ap.add_argument("--puntos", type=int, default=2048)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-muestras", type=int, default=None, help="solo para pruebas rápidas")
    a = ap.parse_args()

    X, y = cargar(descargar(a.datos), a.max_muestras)
    part = particion(y, "particion.npz" if not a.max_muestras else "particion_prueba.npz")
    resumen(X, y, part)
    graficas_datos(X, y, part)
    print("parámetros:", n_param(MODELOS[a.modelo]()), "| puntos por nube:", a.puntos, flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    filas, cms, hists = [], {}, {}
    for seed in a.semillas:
        m, r, h, mejor_ep, s_ep, mem, (P, Y) = entrenar(a.modelo, X, y, part, a.puntos, a.epocas, a.lr, a.batch, seed, dev)
        fila = dict(modelo=a.modelo, puntos=a.puntos, seed=seed, acc=r["acc"], acc_balanceada=r["acc_balanceada"],
                    f1_macro=r["f1_macro"], auc_ovr=r["auc_ovr"], mejor_epoca=mejor_ep, s_por_epoca=s_ep, mem_gpu_gb=mem,
                    parametros=n_param(m))
        if seed == a.semillas[0]:
            (a_lo, a_hi), (f_lo, f_hi) = bootstrap(P, Y)
            fila.update(ic_acc_lo=a_lo, ic_acc_hi=a_hi, ic_f1_lo=f_lo, ic_f1_hi=f_hi)
            figura_predicciones(a.modelo, X, part, P, Y)
        print(fila, flush=True)
        filas.append(fila); cms[seed] = r["cm"]; hists[seed] = h
        previo = pd.read_csv("resultados.csv") if os.path.exists("resultados.csv") else pd.DataFrame()
        if len(previo):
            previo = previo[~((previo["modelo"] == a.modelo) & (previo["seed"] == seed))]
        pd.concat([previo, pd.DataFrame([fila])], ignore_index=True).to_csv("resultados.csv", index=False)
        pd.DataFrame(h).to_csv(f"historial_{a.modelo}_seed{seed}.csv", index=False)
        reporte_por_clase(r["cm"], f"metricas_por_clase_{a.modelo}_seed{seed}.csv")
        torch.save(m.state_dict(), f"modelo_{a.modelo}_seed{seed}.pt")
        np.savez(f"matrices_confusion_{a.modelo}.npz", **{f"seed{k}": v for k, v in cms.items()})

    df = pd.DataFrame(filas)
    print(df[["acc", "acc_balanceada", "f1_macro", "auc_ovr", "s_por_epoca", "mem_gpu_gb"]].agg(["mean", "std"]).round(4).to_string())
    figura_curvas(a.modelo, hists)
    figura_confusion(a.modelo, sum(cms.values()))


if __name__ == "__main__":
    main()

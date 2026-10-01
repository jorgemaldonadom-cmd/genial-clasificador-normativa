"""Sistema de tracking de métricas y diagnóstico de overfitting / underfitting (Semana 3).

Componentes:
  * MetricsTracker        registro por época (o por paso) de métricas de entrenamiento y validación, con exportación a CSV/JSON.
  * EpochTrainer          entrenamiento por épocas de un clasificador lineal de scikit-learn (SGDClassifier.partial_fit) con tracking.
  * train_torch_softmax   bucle de entrenamiento manual en PyTorch (regresión softmax sobre TF-IDF) con logging por época.
  * KerasTrackerCallback  callback de Keras que vuelca cada época al tracker (se define siempre; se usa solo si TensorFlow está instalado).
  * learning_curve_tracked / validation_curve_tracked   envoltorios de sklearn.model_selection con validación por grupo.
  * diagnose_epochs / diagnose_size   diagnóstico cuantitativo a partir de las curvas (gap, mínimo de pérdida de validación, pendiente).
  * plot_*                figuras profesionales (300 DPI, títulos, ejes, leyendas, rejilla, anotaciones).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, learning_curve, validation_curve

# ----------------------------------------------------------------------------------------------------------------------
# 1. Tracking
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class MetricsTracker:
    """Registra métricas por época. Cada llamada a `log` añade una fila; `frame()` devuelve el historial como DataFrame."""

    run_name: str
    params: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def log(self, epoch: int, **metrics) -> None:
        self.rows.append({"run": self.run_name, "epoch": int(epoch), "t_seg": round(time.time() - self.t0, 3), **{k: float(v) for k, v in metrics.items()}})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def best_epoch(self, metric: str = "val_loss", mode: str = "min") -> int:
        df = self.frame()
        return int(df.loc[df[metric].idxmin() if mode == "min" else df[metric].idxmax(), "epoch"])

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(out_dir / f"{self.run_name}_historial.csv", index=False)
        meta = {"run": self.run_name, "params": self.params, "n_epochs": len(self.rows), "segundos": round(time.time() - self.t0, 2)}
        (out_dir / f"{self.run_name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return out_dir / f"{self.run_name}_historial.csv"


def _metrics(model, X, y, classes, prefix: str) -> dict:
    proba = model.predict_proba(X)
    cls = np.asarray(model.classes_)                      # las columnas de predict_proba siguen el orden de model.classes_
    pred = cls[proba.argmax(axis=1)]
    return {f"{prefix}_loss": log_loss(y, proba, labels=cls), f"{prefix}_acc": accuracy_score(y, pred), f"{prefix}_f1": f1_score(y, pred, average="macro")}


class EpochTrainer:
    """Entrena un SGDClassifier (regresión logística con descenso de gradiente estocástico) época a época con `partial_fit`,
    registrando pérdida, exactitud y F1 macro en entrenamiento y validación. Permite early stopping por paciencia."""

    def __init__(self, classes, alpha: float = 1e-4, penalty: str = "l2", class_weight=None, seed: int = 2026, eta0: float = 0.5):
        self.classes = list(classes)
        self.model = SGDClassifier(loss="log_loss", penalty=penalty, alpha=alpha, learning_rate="optimal", eta0=eta0, class_weight=class_weight, random_state=seed, tol=None, max_iter=1)

    def fit(self, Xtr, ytr, Xva, yva, epochs: int = 60, tracker: MetricsTracker | None = None, patience: int | None = None, verbose: bool = False):
        rng = np.random.default_rng(self.model.random_state)
        n = Xtr.shape[0]; best, best_epoch, wait, best_state = np.inf, 0, 0, None
        for ep in range(1, epochs + 1):
            idx = rng.permutation(n)
            self.model.partial_fit(Xtr[idx], np.asarray(ytr)[idx], classes=self.classes)
            m = {**_metrics(self.model, Xtr, ytr, self.classes, "train"), **_metrics(self.model, Xva, yva, self.classes, "val")}
            if tracker is not None:
                tracker.log(ep, **m)
            if verbose and ep % 10 == 0:
                print(f"época {ep:3d} · train loss {m['train_loss']:.3f} acc {m['train_acc']:.3f} · val loss {m['val_loss']:.3f} acc {m['val_acc']:.3f} f1 {m['val_f1']:.3f}")
            if m["val_loss"] < best - 1e-4:
                best, best_epoch, wait = m["val_loss"], ep, 0
                best_state = (self.model.coef_.copy(), self.model.intercept_.copy())
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    break
        self.best_epoch_, self.best_val_loss_ = best_epoch, best
        if patience is not None and best_state is not None:   # restaura los pesos de la mejor época (early stopping)
            self.model.coef_, self.model.intercept_ = best_state
        return self


class KerasTrackerCallback:
    """Callback para TensorFlow/Keras: vuelca `logs` de cada época al MetricsTracker.
    Se define sin importar TensorFlow para que el módulo cargue en entornos sin esa dependencia; si `tensorflow` está
    instalado, `build()` devuelve una subclase real de `tf.keras.callbacks.Callback`."""

    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker

    def build(self):
        import tensorflow as tf  # noqa: F401  (solo si está instalado)

        tracker = self.tracker

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                tracker.log(epoch + 1, train_loss=logs.get("loss", np.nan), train_acc=logs.get("accuracy", np.nan), val_loss=logs.get("val_loss", np.nan), val_acc=logs.get("val_accuracy", np.nan))

        return _CB()


def train_torch_softmax(Xtr, ytr, Xva, yva, classes, epochs: int = 60, lr: float = 0.05, weight_decay: float = 1e-4, hidden: int = 0, class_weight: np.ndarray | None = None, seed: int = 2026, tracker: MetricsTracker | None = None, patience: int | None = None):
    """Bucle de entrenamiento manual en PyTorch con logging por época. `hidden=0` → regresión softmax; `hidden>0` → MLP de una capa."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    cls = {c: i for i, c in enumerate(classes)}
    Xt = torch.tensor(Xtr.toarray(), dtype=torch.float32); yt = torch.tensor([cls[c] for c in ytr])
    Xv = torch.tensor(Xva.toarray(), dtype=torch.float32); yv = torch.tensor([cls[c] for c in yva])
    d = Xt.shape[1]
    model = nn.Linear(d, len(classes)) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, len(classes)))
    w = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_ep, wait, best_state = np.inf, 0, 0, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad(); loss = crit(model(Xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lt, lv = model(Xt), model(Xv)
            m = {"train_loss": float(crit(lt, yt)), "train_acc": float((lt.argmax(1) == yt).float().mean()), "train_f1": f1_score(yt.numpy(), lt.argmax(1).numpy(), average="macro"),
                 "val_loss": float(crit(lv, yv)), "val_acc": float((lv.argmax(1) == yv).float().mean()), "val_f1": f1_score(yv.numpy(), lv.argmax(1).numpy(), average="macro")}
        if tracker is not None:
            tracker.log(ep, **m)
        if m["val_loss"] < best:
            best, best_ep, wait, best_state = m["val_loss"], ep, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience is not None and wait >= patience:
                break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best


# ----------------------------------------------------------------------------------------------------------------------
# 2. Curvas con validación por grupo
# ----------------------------------------------------------------------------------------------------------------------


def group_cv(n_splits: int, seed: int):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def learning_curve_tracked(estimator, X, y, groups, sizes=np.linspace(0.1, 1.0, 8), n_splits: int = 5, seed: int = 2026, scoring: str = "f1_macro") -> pd.DataFrame:
    cv = list(group_cv(n_splits, seed).split(X, y, groups))
    n, tr, va = learning_curve(estimator, X, y, train_sizes=sizes, cv=cv, scoring=scoring, n_jobs=-1, shuffle=False)
    return pd.DataFrame({"n_train": n, "train_mean": tr.mean(1), "train_std": tr.std(1), "val_mean": va.mean(1), "val_std": va.std(1), "gap": tr.mean(1) - va.mean(1)})


def validation_curve_tracked(estimator, X, y, groups, param_name: str, param_range, n_splits: int = 5, seed: int = 2026, scoring: str = "f1_macro") -> pd.DataFrame:
    cv = list(group_cv(n_splits, seed).split(X, y, groups))
    tr, va = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring=scoring, n_jobs=-1)
    return pd.DataFrame({"param": list(param_range), "train_mean": tr.mean(1), "train_std": tr.std(1), "val_mean": va.mean(1), "val_std": va.std(1), "gap": tr.mean(1) - va.mean(1)})


def cv_train_val(estimator, X, y, groups, n_splits: int = 5, seed: int = 2026) -> dict:
    """F1 macro y exactitud en entrenamiento y validación por pliegue (grupo) → medias, desviaciones y gap."""
    rows = []
    for tr, va in group_cv(n_splits, seed).split(X, y, groups):
        m = clone(estimator).fit(X[tr], y[tr])
        rows.append({"train_f1": f1_score(y[tr], m.predict(X[tr]), average="macro"), "val_f1": f1_score(y[va], m.predict(X[va]), average="macro"),
                     "train_acc": accuracy_score(y[tr], m.predict(X[tr])), "val_acc": accuracy_score(y[va], m.predict(X[va]))})
    df = pd.DataFrame(rows)
    out = {k: float(df[k].mean()) for k in df.columns}; out.update({k + "_std": float(df[k].std()) for k in df.columns})
    out["gap_f1"] = out["train_f1"] - out["val_f1"]; out["gap_acc"] = out["train_acc"] - out["val_acc"]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 3. Diagnóstico cuantitativo
# ----------------------------------------------------------------------------------------------------------------------


def diagnose_epochs(hist: pd.DataFrame, gap_overfit: float = 0.15, low_perf: float = 0.55) -> dict:
    """Diagnóstico a partir de un historial por época (columnas train_loss, val_loss, train_acc, val_acc, train_f1, val_f1).

    Reglas (umbrales explícitos y modificables):
      * overfitting   : gap final de exactitud > gap_overfit  O  la pérdida de validación sube ≥ 5 % desde su mínimo mientras la de entrenamiento baja;
      * underfitting  : exactitud de entrenamiento final < low_perf (ambas curvas convergen a un rendimiento pobre);
      * óptimo        : ninguna de las dos.
    """
    h = hist.sort_values("epoch").reset_index(drop=True); last = h.iloc[-1]
    i_min = int(h["val_loss"].idxmin()); ep_min = int(h.loc[i_min, "epoch"]); vmin = float(h.loc[i_min, "val_loss"])
    rebote = float(last["val_loss"] / vmin - 1.0)                       # cuánto subió la pérdida de validación tras su mínimo
    train_baja = bool(last["train_loss"] < h["train_loss"].iloc[max(0, i_min)] - 1e-4)
    gap_acc = float(last["train_acc"] - last["val_acc"]); gap_f1 = float(last["train_f1"] - last["val_f1"]) if "train_f1" in h else np.nan
    gap_max = float((h["train_acc"] - h["val_acc"]).max()); gap_loss = float(last["val_loss"] - last["train_loss"])
    # tendencia del gap en el último tercio
    tercio = h.iloc[len(h) * 2 // 3:]; slope_gap = float(np.polyfit(tercio["epoch"], tercio["train_acc"] - tercio["val_acc"], 1)[0]) if len(tercio) > 2 else np.nan
    if last["train_acc"] < low_perf:
        dx = "underfitting"
    elif gap_acc > gap_overfit or (rebote >= 0.05 and train_baja):
        dx = "overfitting"
    else:
        dx = "óptimo"
    return {"diagnóstico": dx, "epoca_min_val_loss": ep_min, "val_loss_min": vmin, "val_loss_final": float(last["val_loss"]), "rebote_val_loss": rebote,
            "train_loss_final": float(last["train_loss"]), "gap_loss_final": gap_loss, "train_acc_final": float(last["train_acc"]), "val_acc_final": float(last["val_acc"]),
            "gap_acc_final": gap_acc, "gap_acc_max": gap_max, "gap_f1_final": gap_f1, "val_f1_final": float(last["val_f1"]) if "val_f1" in h else np.nan,
            "val_f1_max": float(h["val_f1"].max()) if "val_f1" in h else np.nan, "pendiente_gap_ultimo_tercio": slope_gap, "n_epocas": int(last["epoch"])}


def diagnose_size(lc: pd.DataFrame) -> dict:
    """Diagnóstico de la curva de aprendizaje por tamaño: gap final, pendiente de validación global y en el último tramo
    (en puntos porcentuales de F1 por cada 100 normas adicionales) y convergencia de las curvas."""
    lc = lc.sort_values("n_train"); last = lc.iloc[-1]; tail = lc.iloc[-3:]
    slope_tail = float(np.polyfit(tail["n_train"], tail["val_mean"], 1)[0] * 100 * 100)
    slope_all = float(np.polyfit(lc["n_train"], lc["val_mean"], 1)[0] * 100 * 100)
    ganancia_total = float((last["val_mean"] - lc.iloc[0]["val_mean"]) * 100)
    if last["gap"] >= 0.10 and slope_tail > 0.5:
        lectura = "gap grande y validación aún creciente → varianza alta: más datos y/o más regularización"
    elif last["gap"] >= 0.10:
        lectura = "gap grande y validación estancada → varianza alta con techo de datos/etiquetas"
    elif last["val_mean"] < 0.55:
        lectura = "curvas convergen a rendimiento bajo → sesgo alto (underfitting)"
    else:
        lectura = "curvas convergen con rendimiento aceptable"
    return {"n_max": int(last["n_train"]), "train_final": float(last["train_mean"]), "val_final": float(last["val_mean"]), "gap_final": float(last["gap"]),
            "pendiente_val_por_100_global": slope_all, "pendiente_val_por_100_tramo_final": slope_tail, "ganancia_val_total_puntos": ganancia_total,
            "val_std_final": float(last["val_std"]), "convergen": bool(last["gap"] < 0.10), "lectura": lectura}


# ----------------------------------------------------------------------------------------------------------------------
# 4. Figuras profesionales
# ----------------------------------------------------------------------------------------------------------------------

TRAIN_C, VAL_C, ACC, INK, MUTED = "#2a78d6", "#eb6834", "#7A1F3D", "#1b1a19", "#6e6862"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": ":", "legend.frameon": False})


def plot_epoch_curves(hist: pd.DataFrame, title: str, path: str | Path, metric: str = "acc", dx: dict | None = None):
    """Figura con dos paneles: (A) pérdida train/val por época, (B) exactitud o F1 train/val por época. Anota el mínimo de val_loss
    (punto de early stopping) y el gap final."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_loss"], color=TRAIN_C, lw=2, label="entrenamiento")
    ax.plot(hist["epoch"], hist["val_loss"], color=VAL_C, lw=2, label="validación")
    i = hist["val_loss"].idxmin(); ep, v = hist.loc[i, "epoch"], hist.loc[i, "val_loss"]
    ax.axvline(ep, color=MUTED, ls="--", lw=1); ax.scatter([ep], [v], color=VAL_C, zorder=5, s=40)
    ax.annotate(f"mín. val loss = {v:.3f}\népoca {int(ep)} (early stopping)", (ep, v), textcoords="offset points", xytext=(10, 18), fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    last = hist.iloc[-1]
    ax.annotate(f"gap final = {last['val_loss'] - last['train_loss']:.3f}", (last["epoch"], (last["val_loss"] + last["train_loss"]) / 2), textcoords="offset points", xytext=(-95, 0), fontsize=8.5, color=ACC)
    ax.set_xlabel("época"); ax.set_ylabel("pérdida (log-loss)"); ax.set_title("A · Pérdida: entrenamiento vs validación"); ax.legend(loc="center right")
    ax = axes[1]
    tr, va = f"train_{metric}", f"val_{metric}"; lab = "exactitud" if metric == "acc" else "F1 macro"
    ax.plot(hist["epoch"], hist[tr], color=TRAIN_C, lw=2, label="entrenamiento")
    ax.plot(hist["epoch"], hist[va], color=VAL_C, lw=2, label="validación")
    ax.fill_between(hist["epoch"], hist[va], hist[tr], color=ACC, alpha=0.08, label="gap (varianza)")
    j = hist[va].idxmax(); ax.scatter([hist.loc[j, "epoch"]], [hist.loc[j, va]], color=VAL_C, zorder=5, s=40)
    ax.annotate(f"máx. val {lab} = {hist.loc[j, va]:.3f} (época {int(hist.loc[j, 'epoch'])})", (hist.loc[j, "epoch"], hist.loc[j, va]), textcoords="offset points", xytext=(8, -16), fontsize=8.5, color=INK)
    ax.annotate(f"gap final = {last[tr] - last[va]:.3f}", (last["epoch"], (last[tr] + last[va]) / 2), textcoords="offset points", xytext=(-80, 0), fontsize=8.5, color=ACC)
    ax.set_ylim(0, 1.02); ax.set_xlabel("época"); ax.set_ylabel(lab); ax.set_title(f"B · {lab.capitalize()}: entrenamiento vs validación"); ax.legend(loc="lower right")
    sub = f" · diagnóstico: {dx['diagnóstico'].upper()}" if dx else ""
    fig.suptitle(title + sub, fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, facecolor="white"); return fig


def plot_learning_curve(lc: pd.DataFrame, title: str, path: str | Path, ylabel: str = "F1 macro", dx: dict | None = None):
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(lc["n_train"], lc["train_mean"], "o-", color=TRAIN_C, lw=2, label="entrenamiento (media ± DE)")
    ax.fill_between(lc["n_train"], lc["train_mean"] - lc["train_std"], lc["train_mean"] + lc["train_std"], color=TRAIN_C, alpha=0.15)
    ax.plot(lc["n_train"], lc["val_mean"], "o-", color=VAL_C, lw=2, label="validación (media ± DE)")
    ax.fill_between(lc["n_train"], lc["val_mean"] - lc["val_std"], lc["val_mean"] + lc["val_std"], color=VAL_C, alpha=0.15)
    last = lc.iloc[-1]
    ax.annotate(f"gap = {last['gap']:.2f}", (last["n_train"], (last["train_mean"] + last["val_mean"]) / 2), textcoords="offset points", xytext=(-55, 0), fontsize=9, color=ACC)
    ax.annotate(f"val {last['val_mean']:.3f} con n = {int(last['n_train'])}", (last["n_train"], last["val_mean"]), textcoords="offset points", xytext=(-125, -22), fontsize=8.5, color=INK)
    if dx:
        ax.text(0.02, 0.04, f"pendiente de validación: {dx['pendiente_val_por_100_global']:+.1f} pp de F1 por 100 normas (global) · {dx['pendiente_val_por_100_tramo_final']:+.1f} pp (último tramo)\n{dx['lectura']}", transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_ylim(0, 1.02); ax.set_xlabel("normas de entrenamiento (n)"); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.72))
    fig.tight_layout(); fig.savefig(path, facecolor="white"); return fig


def plot_validation_curve(vc: pd.DataFrame, param_label: str, title: str, path: str | Path, log: bool = True, ylabel: str = "F1 macro", chosen=None):
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = vc["param"]
    ax.plot(x, vc["train_mean"], "o-", color=TRAIN_C, lw=2, label="entrenamiento (media ± DE)")
    ax.fill_between(x, vc["train_mean"] - vc["train_std"], vc["train_mean"] + vc["train_std"], color=TRAIN_C, alpha=0.15)
    ax.plot(x, vc["val_mean"], "o-", color=VAL_C, lw=2, label="validación (media ± DE)")
    ax.fill_between(x, vc["val_mean"] - vc["val_std"], vc["val_mean"] + vc["val_std"], color=VAL_C, alpha=0.15)
    j = vc["val_mean"].idxmax(); ax.scatter([vc.loc[j, "param"]], [vc.loc[j, "val_mean"]], color=VAL_C, s=60, zorder=5)
    ax.annotate(f"mejor validación: {vc.loc[j, 'param']:g}\nF1 = {vc.loc[j, 'val_mean']:.3f} · gap = {vc.loc[j, 'gap']:.2f}", (vc.loc[j, "param"], vc.loc[j, "val_mean"]), textcoords="offset points", xytext=(10, -40), fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    if chosen is not None:
        ax.axvline(chosen, color=ACC, ls="--", lw=1); ax.text(chosen, 0.62, f"  base ({chosen:g})", fontsize=8.5, color=ACC, va="bottom")
    if log:
        ax.set_xscale("log")
    ax.text(0.02, 0.04, "← más regularización / menos complejidad (sesgo)          más complejidad / menos regularización (varianza) →", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02); ax.set_xlabel(param_label); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.70))
    fig.tight_layout(); fig.savefig(path, facecolor="white"); return fig


def plot_before_after(tabla: pd.DataFrame, path: str | Path, title: str = "Antes y después de las estrategias de mejora (CV por grupo, 5 pliegues)"):
    """Barras agrupadas: F1 macro de entrenamiento y validación por configuración, con el gap anotado."""
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(tabla)); w = 0.38
    ax.bar(x - w / 2, tabla["train_f1"], w, color=TRAIN_C, label="F1 macro entrenamiento")
    ax.bar(x + w / 2, tabla["val_f1"], w, color=VAL_C, label="F1 macro validación")
    for i, r in enumerate(tabla.itertuples()):
        ax.text(i - w / 2, r.train_f1 + 0.01, f"{r.train_f1:.2f}", ha="center", fontsize=8.5)
        ax.text(i + w / 2, r.val_f1 + 0.01, f"{r.val_f1:.2f}", ha="center", fontsize=8.5)
        ax.text(i, 0.03, f"gap {r.gap_f1:.2f}", ha="center", fontsize=8.5, color=ACC, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c.replace(": ", ":\n", 1) for c in tabla["configuración"]], fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("F1 macro"); ax.set_title(title); ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(path, facecolor="white"); return fig

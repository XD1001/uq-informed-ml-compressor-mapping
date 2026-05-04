# ===== Deep Ensemble with repeat runs + Excel export (aligned with GPR & overlay) =====
from __future__ import annotations

import os
import random
from typing import List, Tuple, Dict, Optional, Sequence, Any, cast

import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# NumPy 2 removed np.float_; preserve alias used in runtime typing casts.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

# for appending to existing Excel
try:
    import openpyxl  # noqa: F401
except Exception:
    pass

"""
Deep Ensemble Learning (DEL) with aleatoric + epistemic UQ
Using multiple MLPs that output [mu, log_var] and trained via Gaussian NLL.

Upgrades / Alignments vs. GPR & overlay:
- Accept fixed_indices to guarantee the same validation split across models
- Accept uq_excel_path + uq_sheet_name to write per-point UQ into a shared workbook
- Inference timing API (infer_repeats/warmup/n_samples) with mean/std for overlay preview
- Metrics/return schema compatible with overlay: y_std, y_std_ale, y_std_epi, metrics keys
- Keep repeat-runs Excel pipeline (DE-only) for standalone usage
- (NEW) Per-point UQ export now uses the SAME long/tidy schema & sheet handling as GPR
  via _build_uq_long_table + _write_df_to_excel
"""

# -------------------- Global seeding (base; runs will offset) --------------------
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


# -------------------- Utilities --------------------
def get_current_and_parent_dir() -> Tuple[str, str]:
    cur = os.getcwd()
    return cur, os.path.dirname(cur)


def load_data(file: str, sheet: str) -> pd.DataFrame:
    if file.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=sheet)
    if df.empty:
        raise ValueError("Loaded data is empty")
    return df.dropna()


def split_data(
    data: pd.DataFrame, input_cols: Sequence[str], output_cols: Sequence[str]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    x = data[input_cols].to_numpy(dtype=np.float64, copy=True)
    y = data[output_cols].to_numpy(dtype=np.float64, copy=True)
    return x, y


def plot_validation_group(
    y_valid: NDArray[np.float_],
    y_pred: NDArray[np.float_],
    y_std_total: NDArray[np.float_],
    y_name: Sequence[str],
) -> None:
    plt.figure(figsize=(10, 5))
    for i, _ in enumerate(y_name):
        plt.subplot(1, 2, i + 1)
        plt.errorbar(
            y_valid[:, i],
            y_pred[:, i],
            yerr=y_std_total[:, i],
            fmt="o",
            alpha=0.5,
            ecolor="lightgray",
        )
        y_min, y_max = float(y_valid[:, i].min()), float(y_valid[:, i].max())
        xmin, xmax = max(0.0, 0.95 * y_min), 1.05 * y_max
        plt.plot([xmin, xmax], [xmin, xmax], lw=1)
        x_line = np.linspace(xmin, xmax, 200, dtype=np.float64)
        plt.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.2, label="±10%"
        )
        r2_v = float(r2_score(y_valid[:, i], y_pred[:, i]))
        mse_v = float(mean_squared_error(y_valid[:, i], y_pred[:, i]))
        mape_v = float(
            mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i])
        )
        plt.text(
            0.65 * y_max,
            0.9 * y_max,
            f"$R^2$={r2_v:.3f}\nMSE={mse_v:.3f}\nMAPE={mape_v:.3f}",
            fontsize=11,
        )
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        plt.xlabel(f"True {y_name[i]}")
        plt.ylabel(f"Pred {y_name[i]}")
        plt.title(f"{y_name[i]} Prediction (with total uncertainty)")
    plt.tight_layout()
    plt.show()


def plot_loss_history(history: List[float], name: str) -> None:
    if not history:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xticks(range(1, len(history) + 1))
    plt.xlabel("Fold")
    plt.ylabel(name)
    plt.title(name)
    plt.grid()
    plt.show()


# -------------------- Gaussian NLL utilities --------------------
def gaussian_nll_vector(
    y_true: NDArray[np.float_],
    y_mean: NDArray[np.float_],
    y_var: NDArray[np.float_],
    *,
    add_const_2pi: bool = False,
) -> NDArray[np.float_]:
    eps = 1e-12
    v = np.maximum(y_var, eps)
    term1 = 0.5 * np.log(v) + (
        0.5 * np.log(2 * np.pi) if add_const_2pi else 0.0
    )
    term2 = 0.5 * ((y_true - y_mean) ** 2) / v
    return term1 + term2


def mixture_gaussian_nll_vector(
    y_true: NDArray[np.float_],
    mus: NDArray[np.float_],
    vars_: NDArray[np.float_],
    *,
    add_const_2pi: bool = False,
) -> NDArray[np.float_]:
    eps = 1e-12
    v = np.maximum(vars_, eps)
    log_comp = -0.5 * (np.log(v) + ((y_true[None, :] - mus) ** 2) / v)
    if add_const_2pi:
        log_comp = log_comp - 0.5 * np.log(2 * np.pi)
    max_l = np.max(log_comp, axis=0)
    log_mean_exp = max_l + np.log(
        np.mean(np.exp(log_comp - max_l[None, :]), axis=0)
    )
    return -log_mean_exp


# -------------------- Model definition --------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        act = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }.get(activation.lower(), nn.ReLU())
        dims = [input_dim] + list(hidden_layers)
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), act]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        self.base = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class DeepEnsembleRegressor:
    """
    Each base model outputs [mu, log_var]; trained by Gaussian NLL.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        n_estimators: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        n_epochs: int = 100,
        patience: int = 10,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        optimizer_name: str = "adam",
        adaptive_lr: bool = True,
        device: Optional[str] = None,
        logvar_clamp: Tuple[float, float] = (-10.0, 5.0),
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.output_dim = 2
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.adaptive_lr = adaptive_lr
        self.logvar_clamp = logvar_clamp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models: List[nn.Module] = []
        self.opts: List[optim.Optimizer] = []
        self.scheds: List[Optional[optim.lr_scheduler.ReduceLROnPlateau]] = []

        for _ in range(n_estimators):
            net = nn.Sequential(
                MLP(input_dim, self.hidden_layers, activation, dropout_rate),
                nn.Linear(self.hidden_layers[-1], self.output_dim),
            ).to(self.device)
            opt = self._make_optimizer(net)
            self.models.append(net)
            self.opts.append(opt)
            self.scheds.append(
                optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=max(1, patience // 5)
                )
                if adaptive_lr
                else None
            )

    def _make_optimizer(self, model: nn.Module) -> optim.Optimizer:
        name = self.optimizer_name.lower()
        if name == "sgd":
            return optim.SGD(
                model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        if name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate)
        if name == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    @staticmethod
    def _gaussian_nll(
        mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # s = log(var)
        return 0.5 * ((y - mu) ** 2 * torch.exp(-s) + s)

    def fit(
        self,
        X_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        X_val: Optional[NDArray[np.float_]] = None,
        y_val: Optional[NDArray[np.float_]] = None,
    ) -> None:
        Xtr = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        ytr = torch.tensor(
            y_train.reshape(-1, 1), dtype=torch.float32, device=self.device
        )

        dl = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=self.batch_size, shuffle=True
        )
        ntr = int(Xtr.shape[0])

        has_val = X_val is not None and y_val is not None
        Xv: Optional[torch.Tensor] = None
        Yv: Optional[torch.Tensor] = None
        if has_val:
            Xv = torch.tensor(
                cast(NDArray[np.float_], X_val),
                dtype=torch.float32,
                device=self.device,
            )
            Yv = torch.tensor(
                cast(NDArray[np.float_], y_val).reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )

        for model, opt, sched in zip(self.models, self.opts, self.scheds):
            best = float("inf")
            best_state: Optional[Dict[str, torch.Tensor]] = None
            no_imp = 0
            for _ in range(self.n_epochs):
                model.train()
                tot = 0.0
                for xb, yb in dl:
                    opt.zero_grad()
                    out = model(xb)
                    mu, s = out[:, :1], out[:, 1:2]
                    s = torch.clamp(
                        s, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
                    )
                    loss = self._gaussian_nll(mu, s, yb).mean()
                    loss.backward()
                    opt.step()
                    tot += float(loss.item()) * int(xb.size(0))
                tot /= float(ntr)

                if has_val and Xv is not None and Yv is not None:
                    model.eval()
                    with torch.no_grad():
                        outv = model(Xv)
                        muv, sv = (
                            outv[:, :1],
                            torch.clamp(
                                outv[:, 1:2],
                                min=self.logvar_clamp[0],
                                max=self.logvar_clamp[1],
                            ),
                        )
                        vloss = float(
                            self._gaussian_nll(muv, sv, Yv).mean().item()
                        )
                    if sched is not None:
                        sched.step(vloss)
                    if vloss < best - 1e-9:
                        best, no_imp = vloss, 0
                        best_state = {
                            k: v.detach().clone()
                            for k, v in model.state_dict().items()
                        }
                    else:
                        no_imp += 1
                    if no_imp >= self.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

    def predict_components(
        self, X: NDArray[np.float_]
    ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        mus, vars_ = [], []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(Xt)
                mu = out[:, 0]
                s = torch.clamp(
                    out[:, 1],
                    min=self.logvar_clamp[0],
                    max=self.logvar_clamp[1],
                )
                var = torch.exp(s)
                mus.append(mu.cpu().numpy().astype(np.float64, copy=False))
                vars_.append(var.cpu().numpy().astype(np.float64, copy=False))
        return np.stack(mus, axis=0), np.stack(vars_, axis=0)

    def predict(
        self, X: NDArray[np.float_]
    ) -> Tuple[
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
        NDArray[np.float_],
    ]:
        mus, vars_ = self.predict_components(X)
        mu_ens = np.mean(mus, axis=0, dtype=np.float64)
        var_ale = np.mean(vars_, axis=0, dtype=np.float64)
        var_epi = np.var(mus, axis=0, ddof=0)
        var_tot = var_ale + var_epi
        eps = 1e-12
        return (
            mu_ens,
            np.sqrt(np.maximum(var_tot, eps)),
            np.sqrt(np.maximum(var_ale, eps)),
            np.sqrt(np.maximum(var_epi, eps)),
        )

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "input_dim": self.input_dim,
                "hidden_layers": self.hidden_layers,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "patience": self.patience,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "optimizer_name": self.optimizer_name,
                "adaptive_lr": self.adaptive_lr,
                "logvar_clamp": self.logvar_clamp,
                "state_dicts": [m.state_dict() for m in self.models],
            },
            filepath,
        )

    @classmethod
    def load(
        cls, filepath: str, device: Optional[str] = None
    ) -> "DeepEnsembleRegressor":
        st: Dict[str, Any] = torch.load(
            filepath, map_location=device or torch.device("cpu")
        )
        hl = st.get("hidden_layers", [100, 50])
        hl_list = list(hl) if isinstance(hl, (list, tuple)) else [int(hl)]
        logvar_tuple: Tuple[float, float] = cast(
            Tuple[float, float], tuple(st.get("logvar_clamp", (-10.0, 5.0)))
        )
        model = cls(
            input_dim=int(st["input_dim"]),
            hidden_layers=[int(x) for x in hl_list],
            n_estimators=int(st["n_estimators"]),
            learning_rate=float(st["learning_rate"]),
            batch_size=int(st["batch_size"]),
            n_epochs=int(st["n_epochs"]),
            patience=int(st["patience"]),
            dropout_rate=float(st["dropout_rate"]),
            activation=str(st["activation"]),
            optimizer_name=str(st["optimizer_name"]),
            adaptive_lr=bool(st.get("adaptive_lr", True)),
            device=device,
            logvar_clamp=logvar_tuple,
        )
        for m, sd in zip(model.models, st["state_dicts"]):
            m.load_state_dict(sd)
        return model


# -------------------- Normalization & K-fold CV --------------------
def normalize_xy(
    X: NDArray[np.float_], Y: NDArray[np.float_]
) -> Tuple[
    NDArray[np.float_], NDArray[np.float_], StandardScaler, StandardScaler
]:
    xs, ys = StandardScaler(), StandardScaler()
    return (
        xs.fit_transform(X).astype(np.float64),
        ys.fit_transform(Y).astype(np.float64),
        xs,
        ys,
    )


def de_k_fold_cross_validation(
    X: NDArray[np.float_],
    Y: NDArray[np.float_],
    k: int = 5,
    patience: int = 10,
    *,
    n_estimators: int = 10,
    hidden_layers: Sequence[int] = (100, 50),
    lr: float = 1e-3,
    batch: int = 32,
    epochs: int = 1000,
    dropout: float = 0.1,
    activation: str = "relu",
    model_dir: str = "./models/DE",
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    X_norm, Y_norm, x_scaler, y_scaler = normalize_xy(X, Y)

    history_mape: List[float] = []
    history_mse: List[float] = []
    history_r2: List[float] = []
    history_nll: List[float] = []

    fold = 1
    for tr_idx, va_idx in kf.split(X_norm):
        print(f"\n===== DE Fold {fold}/{k} =====")
        Xtr, Xva = X_norm[tr_idx], X_norm[va_idx]
        Ytr, Yva = Y_norm[tr_idx], Y_norm[va_idx]

        models: List[DeepEnsembleRegressor] = []
        for out_i in range(Y.shape[1]):
            ens = DeepEnsembleRegressor(
                input_dim=int(X.shape[1]),
                hidden_layers=list(hidden_layers),
                n_estimators=n_estimators,
                learning_rate=lr,
                batch_size=batch,
                n_epochs=epochs,
                patience=patience,
                dropout_rate=dropout,
                activation=activation,
                optimizer_name="adam",
                adaptive_lr=True,
                logvar_clamp=(-10.0, 5.0),
            )
            ens.fit(Xtr, Ytr[:, out_i], X_val=Xva, y_val=Yva[:, out_i])
            models.append(ens)
            ens.save(f"{model_dir}/DE_model_fold{fold}_output{out_i}.pt")

        joblib.dump(x_scaler, f"{model_dir}/DE_x_scaler_fold{fold}.pkl")
        joblib.dump(y_scaler, f"{model_dir}/DE_y_scaler_fold{fold}.pkl")

        Y_pred = np.zeros_like(Yva)
        nlls: List[float] = []
        for out_i, ens in enumerate(models):
            mu, std_tot, _, _ = ens.predict(Xva)
            Y_pred[:, out_i] = mu
            nll_vec = gaussian_nll_vector(
                Yva[:, out_i], mu, std_tot**2, add_const_2pi=False
            )
            nlls.append(float(np.mean(nll_vec)))

        mape = float(mean_absolute_percentage_error(Yva, Y_pred))
        mse = float(mean_squared_error(Yva, Y_pred))
        r2 = float(r2_score(Yva, Y_pred))
        nll = float(np.mean(nlls))
        print(
            f"Fold-{fold}  MAPE={mape:.4f}  MSE={mse:.4f}  R2={r2:.4f}  NLL={nll:.6f}"
        )

        history_mape.append(mape)
        history_mse.append(mse)
        history_r2.append(r2)
        history_nll.append(nll)
        fold += 1

    plot_loss_history(history_mape, "MAPE (norm)")
    plot_loss_history(history_mse, "MSE (norm)")
    plot_loss_history(history_r2, "R2 (norm)")
    plot_loss_history(history_nll, "Gaussian NLL (norm)")


# -------------------- Validation container & plot --------------------
class _ValidationResult:
    """
    Duck-typed result container; extended to include y_std_ale / y_std_epi.
    """

    def __init__(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_names: List[str],
        y_std: Optional[np.ndarray] = None,  # TOTAL std (original units)
        y_lo: Optional[np.ndarray] = None,
        y_hi: Optional[np.ndarray] = None,
        fig: Optional[Figure] = None,
        metrics: Optional[dict] = None,
        settings: Optional[dict] = None,
        *,
        y_std_ale: Optional[np.ndarray] = None,  # NEW
        y_std_epi: Optional[np.ndarray] = None,  # NEW
    ) -> None:
        self.name = name
        self.y_true = y_true
        self.y_pred = y_pred
        self.output_names = output_names
        self.y_std = y_std
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.fig = fig
        self.metrics = metrics or {}
        self.settings = settings or {}
        self.y_std_ale = y_std_ale
        self.y_std_epi = y_std_epi

    def check(self) -> None:
        assert self.y_true.shape == self.y_pred.shape
        if self.y_std is not None:
            assert self.y_std.shape == self.y_pred.shape
        if self.y_std_ale is not None:
            assert self.y_std_ale.shape == self.y_pred.shape
        if self.y_std_epi is not None:
            assert self.y_std_epi.shape == self.y_pred.shape
        if (self.y_lo is not None) ^ (self.y_hi is not None):
            raise ValueError(
                "y_lo and y_hi must both be provided or both None."
            )


def _make_validation_figure(
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray],
    y_name: List[str],
) -> Figure:
    fig, axes = plt.subplots(1, len(y_name), figsize=(10, 5))
    if len(y_name) == 1:
        axes = [axes]

    for i, _ in enumerate(y_name):
        ax = axes[i]
        y_min, y_max = float(y_valid[:, i].min()), float(y_valid[:, i].max())
        xmin, xmax = max(0.0, 0.95 * y_min), 1.05 * y_max
        ax.plot([xmin, xmax], [xmin, xmax], lw=1)
        x_line = np.linspace(xmin, xmax, 200, dtype=np.float64)
        ax.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.2, label="±10%"
        )

        if y_std is not None:
            ax.errorbar(
                y_valid[:, i],
                y_pred[:, i],
                yerr=y_std[:, i],
                fmt="o",
                alpha=0.6,
                ecolor="lightgray",
            )
        else:
            ax.scatter(y_valid[:, i], y_pred[:, i], alpha=0.85)

        r2_v = float(r2_score(y_valid[:, i], y_pred[:, i]))
        mse_v = float(mean_squared_error(y_valid[:, i], y_pred[:, i]))
        mape_v = float(
            mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i])
        )
        ax.text(
            0.85 * xmax,
            0.65 * xmax,
            f"$R^2$={r2_v:.3f}\nMSE={mse_v:.3f}\nMAPE={mape_v:.3f}",
            fontsize=11,
            ha="right",
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel(f"True {y_name[i]}", fontsize=14)
        ax.set_ylabel(f"Pred {y_name[i]}", fontsize=14)
        ax.set_title(
            f"{y_name[i]} Prediction (with total uncertainty)", fontsize=14
        )
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# -------------------- Helper: batch builder & timing & unified prediction --------------------
def _make_batch(
    X: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(2025)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty X for timing batch.")
    if n >= n_samples:
        idx = rng.choice(n, size=n_samples, replace=False)
    else:
        idx = rng.choice(n, size=n_samples, replace=True)
    return X[idx]


def _de_predict_with_decomposition(
    models: List[DeepEnsembleRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_raw: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_norm = x_scaler.transform(X_raw).astype(np.float64)
    n, m = X_norm.shape[0], len(models)
    y_pred_norm = np.zeros((n, m), dtype=np.float64)
    y_std_tot_norm = np.zeros_like(y_pred_norm)
    y_std_ale_norm = np.zeros_like(y_pred_norm)
    y_std_epi_norm = np.zeros_like(y_pred_norm)

    for j, ens in enumerate(models):
        mu, std_tot, std_ale, std_epi = ens.predict(X_norm)
        y_pred_norm[:, j] = mu
        y_std_tot_norm[:, j] = std_tot
        y_std_ale_norm[:, j] = std_ale
        y_std_epi_norm[:, j] = std_epi

    scales = cast(
        NDArray[np.float_],
        getattr(y_scaler, "scale_", np.ones((m,), dtype=np.float64)),
    )
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_std_tot = y_std_tot_norm * scales.reshape(1, -1)
    y_std_ale = y_std_ale_norm * scales.reshape(1, -1)
    y_std_epi = y_std_epi_norm * scales.reshape(1, -1)
    var_ale = y_std_ale**2
    var_epi = y_std_epi**2
    var_tot = y_std_tot**2
    return y_pred, y_std_tot, y_std_ale, y_std_epi, var_ale, var_epi, var_tot


def _measure_inference_time(
    models: List[DeepEnsembleRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_ref: np.ndarray,
    *,
    repeats: int = 30,
    warmup: int = 5,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    Xb = _make_batch(X_ref, n_samples=n_samples, rng=rng)

    # warmup
    for _ in range(warmup):
        _ = _de_predict_with_decomposition(models, x_scaler, y_scaler, Xb)

    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = _de_predict_with_decomposition(models, x_scaler, y_scaler, Xb)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms


# -------------------- (NEW) Excel helpers to match GPR tidy per-point sheet --------------------
def _build_uq_long_table(
    *,
    model_name: str,
    output_names: List[str],
    input_cols: List[str],
    X_val_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std_total: np.ndarray,
    y_std_epi: np.ndarray,
    y_std_ale: np.ndarray,
) -> pd.DataFrame:
    """
    Build a tidy long-format table with per-sample UQ decomposition
    and attach raw input features for convenience (e.g., T_suc axis).
    Columns: model, point, output, y_true, y_pred, std_total, std_epistemic, std_aleatory, <input cols...>
    """
    n, m = y_true.shape
    rows: List[Dict[str, Any]] = []
    for j, out_name in enumerate(output_names):
        for i in range(n):
            row: Dict[str, Any] = {
                "model": model_name,
                "point": int(i),  # 0-based index in validation set
                "output": out_name,
                "y_true": float(y_true[i, j]),
                "y_pred": float(y_pred[i, j]),
                "std_total": float(y_std_total[i, j]),
                "std_epistemic": float(y_std_epi[i, j]),
                "std_aleatory": float(y_std_ale[i, j]),
            }
            for k, col in enumerate(input_cols):
                row[col] = float(X_val_raw[i, k])
            rows.append(row)
    return pd.DataFrame(rows)


def _write_df_to_excel(
    excel_path: str, sheet_name: str, df: pd.DataFrame
) -> None:
    """
    Append/replace a sheet if the file exists (requires openpyxl).
    If openpyxl append fails, overwrite the file with a new one.
    """
    excel_dir = os.path.dirname(excel_path) or "."
    os.makedirs(excel_dir, exist_ok=True)

    if os.path.exists(excel_path):
        try:
            with pd.ExcelWriter(
                excel_path,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace",
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(
                f"[OK] Appended/replaced sheet '{sheet_name}' in existing file: {os.path.abspath(excel_path)}"
            )
            return
        except Exception as e:
            print(f"[WARN] openpyxl append failed ({e}). Overwriting the file.")

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"[OK] Wrote new Excel file: {os.path.abspath(excel_path)}")


# -------------------- One-run standard interface (returns all std components) --------------------
def run_model_del(
    file: str,
    sheet: str,
    *,
    split_seed: int = 1234,
    k_folds: int = 5,
    patience: int = 100,
    n_estimators: int = 10,
    hidden_layers: Sequence[int] = (100, 50),
    lr: float = 1e-3,
    batch: int = 32,
    epochs: int = 1000,
    dropout: float = 0.1,
    activation: str = "relu",
    model_dir: str = "./models/DE",
    # ---- NEW: overlay-aligned parameters ----
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: Optional[str] = None,
    infer_repeats: int = 30,
    infer_warmup: int = 5,
    infer_n_samples: int = 1000,
) -> _ValidationResult:
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_names = [r"$\dot W_{elec}$", r"$\dot m$"]

    # Seeding per run
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    # 1) Load & split (support fixed_indices for strong consistency with overlay)
    df = load_data(file, sheet)
    X, Y = split_data(df, input_cols, output_cols)

    if fixed_indices is None:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, shuffle=True
        )
        val_idx = None
    else:
        train_idx, val_idx = fixed_indices
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 2) Train CV on TRAIN split only (saves fold-1 artifacts)
    de_k_fold_cross_validation(
        X_train,
        Y_train,
        k=k_folds,
        patience=patience,
        n_estimators=n_estimators,
        hidden_layers=hidden_layers,
        lr=lr,
        batch=batch,
        epochs=epochs,
        dropout=dropout,
        activation=activation,
        model_dir=model_dir,
    )

    # 3) Load fold-1 ensemble & scalers
    models: List[DeepEnsembleRegressor] = []
    for i in range(Y.shape[1]):
        models.append(
            DeepEnsembleRegressor.load(
                f"{model_dir}/DE_model_fold1_output{i}.pt"
            )
        )
    x_scaler: StandardScaler = joblib.load(f"{model_dir}/DE_x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_dir}/DE_y_scaler_fold1.pkl")

    # 4) Unified prediction + decomposition in ORIGINAL units
    Y_pred, Y_std_total, Y_std_ale, Y_std_epi, var_ale, var_epi, var_tot = (
        _de_predict_with_decomposition(models, x_scaler, y_scaler, X_val)
    )

    # 5) Figure
    fig = _make_validation_figure(
        Y_val.astype(np.float64), Y_pred, Y_std_total, output_names
    )

    # 6) Metrics（原始单位 + 归一化空间 NLL）
    mse_each = [
        mean_squared_error(Y_val[:, j], Y_pred[:, j]) for j in range(Y.shape[1])
    ]
    mape_each = [
        mean_absolute_percentage_error(Y_val[:, j], Y_pred[:, j])
        for j in range(Y.shape[1])
    ]
    r2_each = [r2_score(Y_val[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]

    # NLL in NORMALIZED space using total variance
    X_val_norm = x_scaler.transform(X_val).astype(np.float64)
    Y_val_norm = y_scaler.transform(Y_val).astype(np.float64)
    # approximate normalized mean/var from original-space outputs
    y_pred_norm = y_scaler.transform(Y_pred)
    y_std_tot_norm = Y_std_total / cast(NDArray[np.float_], y_scaler.scale_)
    nll_each_norm: List[float] = []
    for j in range(Y.shape[1]):
        nll_vec = gaussian_nll_vector(
            Y_val_norm[:, j], y_pred_norm[:, j], y_std_tot_norm[:, j] ** 2
        )
        nll_each_norm.append(float(np.mean(nll_vec)))

    # 7) Inference timing (ms / 1000 samples) with warmup/repeats — overlay expects mean/std keys
    infer_mean_ms, infer_std_ms = _measure_inference_time(
        models,
        x_scaler,
        y_scaler,
        X_val,
        repeats=infer_repeats,
        warmup=infer_warmup,
        n_samples=infer_n_samples,
    )
    print(
        f"[Inference-DE] {infer_mean_ms:.1f} ± {infer_std_ms:.1f} ms / {infer_n_samples} samples "
        f"(repeats={infer_repeats}, warmup={infer_warmup})"
    )

    metrics = {
        # per-output metrics
        "MSE_each": mse_each,
        "MAPE_each": mape_each,
        "R2_each": r2_each,
        "NLL_each_norm": nll_each_norm,
        # across-output summaries for this run
        "MSE_mean": float(np.mean(mse_each)),
        "MSE_std": float(np.std(mse_each, ddof=1))
        if len(mse_each) > 1
        else 0.0,
        "MAPE_mean": float(np.mean(mape_each)),
        "R2_mean": float(np.mean(r2_each)),
        "R2_std": float(np.std(r2_each, ddof=1)) if len(r2_each) > 1 else 0.0,
        "NLL_mean_norm": float(np.mean(nll_each_norm)),
        "NLL_std_norm": float(np.std(nll_each_norm, ddof=1))
        if len(nll_each_norm) > 1
        else 0.0,
        # inference timing (overlay-compatible keys)
        "Infer_ms_per_1000_mean": float(infer_mean_ms),
        "Infer_ms_per_1000_std": float(infer_std_ms),
        # legacy-style (kept for backward compatibility)
        "infer_ms_per_1000_overall": float(infer_mean_ms),
        "infer_ms_per_1000_each": [],  # not measured per-output in this path
        "Infer_repeats": infer_repeats,
        "Infer_warmup": infer_warmup,
        "Infer_n_samples": infer_n_samples,
    }
    settings = {
        "k_folds": k_folds,
        "patience": patience,
        "lr": lr,
        "batch": batch,
        "epochs": epochs,
        "dropout": dropout,
        "activation": activation,
        "split_seed": split_seed,
        "model_dir": model_dir,
    }

    # 8) Optional: write per-point UQ into a shared Excel workbook (GPR-style tidy sheet)
    if uq_excel_path is not None:
        sheet_long = uq_sheet_name or "DE_UQ"
        uq_df = _build_uq_long_table(
            model_name="DE (Deep Ensemble)",
            output_names=output_names,
            input_cols=input_cols,
            X_val_raw=X_val,
            y_true=Y_val.astype(np.float64),
            y_pred=Y_pred,
            y_std_total=Y_std_total,
            y_std_epi=Y_std_epi,
            y_std_ale=Y_std_ale,
        )
        _write_df_to_excel(uq_excel_path, sheet_long, uq_df)

    return _ValidationResult(
        name="DE (Deep Ensemble)",
        y_true=Y_val.astype(np.float64),
        y_pred=Y_pred,
        y_std=Y_std_total,
        y_std_ale=Y_std_ale,
        y_std_epi=Y_std_epi,
        y_lo=None,
        y_hi=None,
        output_names=output_names,
        fig=fig,
        metrics=metrics,
        settings=settings,
    )


# -------------------- Repeat runs + Excel export (standalone, kept) --------------------
def _col_desc_map() -> Dict[str, str]:
    return {
        "Run": "Run index starting from 1",
        "Seed": "Random seed used for train/val split",
        # Cross-output summaries (per run)
        "R2_mean": "Mean R^2 over outputs (original units) per run",
        "R2_std": "Std of R^2 over outputs (per run, ddof=1)",
        "MSE_mean": "Mean MSE over outputs (original units) per run",
        "MSE_std": "Std of MSE over outputs (per run, ddof=1)",
        "NLL_mean_norm": "Mean Gaussian NLL over outputs (normalized space) per run",
        "NLL_std_norm": "Std of Gaussian NLL over outputs (normalized, ddof=1)",
        # Per-output metrics (per run)
        "W_dot_R2": "R^2 for W_dot (per run, original units)",
        "W_dot_MSE": "MSE for W_dot (per run, original units)",
        "W_dot_NLL_norm": "Gaussian NLL for W_dot (normalized space, per run)",
        "m_dot_R2": "R^2 for m_dot (per run, original units)",
        "m_dot_MSE": "MSE for m_dot (per run, original units)",
        "m_dot_NLL_norm": "Gaussian NLL for m_dot (normalized space, per run)",
        # Inference times
        "infer_ms_per_1000_overall": "End-to-end inference time (ms) per 1000 samples across all outputs (per run)",
        "infer_ms_per_1000_W_dot": "Inference time (ms) per 1000 samples for W_dot only (per run)",
        "infer_ms_per_1000_m_dot": "Inference time (ms) per 1000 samples for m_dot only (per run)",
        # Uncertainty summaries (per run)
        "W_dot_pred_mean": "Mean of predicted W_dot over validation samples",
        "W_dot_pred_std": "Std of predicted W_dot over validation samples",
        "W_dot_total_std_mean": "Mean TOTAL predictive std of W_dot (original units)",
        "W_dot_total_std_std": "Std TOTAL predictive std of W_dot",
        "W_dot_ale_std_mean": "Mean ALEATORY predictive std of W_dot",
        "W_dot_ale_std_std": "Std ALEATORY predictive std of W_dot",
        "W_dot_epi_std_mean": "Mean EPISTEMIC predictive std of W_dot",
        "W_dot_epi_std_std": "Std EPISTEMIC predictive std of W_dot",
        "m_dot_pred_mean": "Mean of predicted m_dot over validation samples",
        "m_dot_pred_std": "Std of predicted m_dot over validation samples",
        "m_dot_total_std_mean": "Mean TOTAL predictive std of m_dot (original units)",
        "m_dot_total_std_std": "Std TOTAL predictive std of m_dot",
        "m_dot_ale_std_mean": "Mean ALEATORY predictive std of m_dot",
        "m_dot_ale_std_std": "Std ALEATORY predictive std of m_dot",
        "m_dot_epi_std_mean": "Mean EPISTEMIC predictive std of m_dot",
        "m_dot_epi_std_std": "Std EPISTEMIC predictive std of m_dot",
    }


def _mean_std_over_samples(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0)
    std = (
        np.std(arr, axis=0, ddof=1)
        if arr.shape[0] > 1
        else np.zeros(arr.shape[1])
    )
    return mean, std


def repeat_runs_to_excel(
    file: str,
    sheet: str,
    excel_filename: str,
    *,
    n_runs: int = 10,
    base_seed: int = 1234,
    k_folds: int = 5,
    patience: int = 100,
    n_estimators: int = 10,
    hidden_layers: Sequence[int] = (100, 50),
    lr: float = 1e-3,
    batch: int = 32,
    epochs: int = 1000,
    dropout: float = 0.1,
    activation: str = "tanh",
    model_dir: str = "./models/DE",
) -> str:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seeds = [base_seed + i for i in range(n_runs)]

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Repeat-DE] Run {run_idx}/{n_runs}, seed={seed}")
        res = run_model_del(
            file,
            sheet,
            split_seed=seed,
            k_folds=k_folds,
            patience=patience,
            n_estimators=n_estimators,
            hidden_layers=hidden_layers,
            lr=lr,
            batch=batch,
            epochs=epochs,
            dropout=dropout,
            activation=activation,
            model_dir=model_dir,
        )
        res.check()
        assert (
            res.y_std is not None
            and res.y_std_ale is not None
            and res.y_std_epi is not None
        )

        # sample-wise to summary per-run
        pred_mean, pred_std = _mean_std_over_samples(res.y_pred)
        total_mean, total_std = _mean_std_over_samples(res.y_std)
        ale_mean, ale_std = _mean_std_over_samples(res.y_std_ale)
        epi_mean, epi_std = _mean_std_over_samples(res.y_std_epi)

        metrics: Dict[str, Any] = res.metrics or {}
        r2_each = list(metrics.get("R2_each", []))
        mse_each = list(metrics.get("MSE_each", []))
        nll_each_norm = list(metrics.get("NLL_each_norm", []))

        r2_mean = float(
            metrics.get("R2_mean", np.mean(r2_each) if r2_each else np.nan)
        )
        r2_std = float(
            metrics.get(
                "R2_std",
                np.std(r2_each, ddof=1) if len(r2_each) > 1 else np.nan,
            )
        )
        mse_mean = float(
            metrics.get("MSE_mean", np.mean(mse_each) if mse_each else np.nan)
        )
        mse_std = float(
            metrics.get(
                "MSE_std",
                np.std(mse_each, ddof=1) if len(mse_each) > 1 else np.nan,
            )
        )
        nll_mean_norm = float(
            metrics.get(
                "NLL_mean_norm",
                np.mean(nll_each_norm) if nll_each_norm else np.nan,
            )
        )
        nll_std_norm = float(
            metrics.get(
                "NLL_std_norm",
                np.std(nll_each_norm, ddof=1)
                if len(nll_each_norm) > 1
                else np.nan,
            )
        )

        infer_overall = float(metrics.get("infer_ms_per_1000_overall", np.nan))
        infer_each: List[float] = list(
            metrics.get("infer_ms_per_1000_each", [np.nan, np.nan])
        )

        # Build row (indexes: 0 -> W_dot, 1 -> m_dot)
        row: Dict[str, Any] = {
            "Run": run_idx,
            "Seed": seed,
            # Cross-output summaries
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "NLL_mean_norm": nll_mean_norm,
            "NLL_std_norm": nll_std_norm,
            # Per-output metrics
            "W_dot_R2": float(r2_each[0]) if len(r2_each) > 0 else np.nan,
            "W_dot_MSE": float(mse_each[0]) if len(mse_each) > 0 else np.nan,
            "W_dot_NLL_norm": float(nll_each_norm[0])
            if len(nll_each_norm) > 0
            else np.nan,
            "m_dot_R2": float(r2_each[1]) if len(r2_each) > 1 else np.nan,
            "m_dot_MSE": float(mse_each[1]) if len(mse_each) > 1 else np.nan,
            "m_dot_NLL_norm": float(nll_each_norm[1])
            if len(nll_each_norm) > 1
            else np.nan,
            # Inference timing
            "infer_ms_per_1000_overall": infer_overall,
            "infer_ms_per_1000_W_dot": float(infer_each[0])
            if len(infer_each) > 0
            else np.nan,
            "infer_ms_per_1000_m_dot": float(infer_each[1])
            if len(infer_each) > 1
            else np.nan,
            # Uncertainty summaries for outputs
            "W_dot_pred_mean": float(pred_mean[0]),
            "W_dot_pred_std": float(pred_std[0]),
            "W_dot_total_std_mean": float(total_mean[0]),
            "W_dot_total_std_std": float(total_std[0]),
            "W_dot_ale_std_mean": float(ale_mean[0]),
            "W_dot_ale_std_std": float(ale_std[0]),
            "W_dot_epi_std_mean": float(epi_mean[0]),
            "W_dot_epi_std_std": float(epi_std[0]),
            "m_dot_pred_mean": float(pred_mean[1]),
            "m_dot_pred_std": float(pred_std[1]),
            "m_dot_total_std_mean": float(total_mean[1]),
            "m_dot_total_std_std": float(total_std[1]),
            "m_dot_ale_std_mean": float(ale_mean[1]),
            "m_dot_ale_std_std": float(ale_std[1]),
            "m_dot_epi_std_mean": float(epi_mean[1]),
            "m_dot_epi_std_std": float(epi_std[1]),
        }
        rows.append(row)

    per_run_df = pd.DataFrame(rows)

    # -------- NEW: per-output metrics sheet (across runs) --------
    df_runs = per_run_df.copy()
    out_labels: List[str] = []
    for col in df_runs.columns:
        if col.endswith("_R2") and not col.startswith("R2_"):
            out_labels.append(col[:-3])
    out_labels = sorted(list(set(out_labels)))

    per_output_rows: List[Dict[str, Any]] = []
    for lbl in out_labels:
        r2_vals = pd.to_numeric(
            df_runs.get(f"{lbl}_R2", pd.Series(dtype=float)), errors="coerce"
        )
        mse_vals = pd.to_numeric(
            df_runs.get(f"{lbl}_MSE", pd.Series(dtype=float)), errors="coerce"
        )
        nll_vals = pd.to_numeric(
            df_runs.get(f"{lbl}_NLL_norm", pd.Series(dtype=float)),
            errors="coerce",
        )
        row_out = {
            "output": lbl,
            "R2_mean": float(np.nanmean(r2_vals)),
            "R2_std": float(np.nanstd(r2_vals, ddof=1))
            if r2_vals.count() > 1
            else float("nan"),
            "MSE_mean": float(np.nanmean(mse_vals)),
            "MSE_std": float(np.nanstd(mse_vals, ddof=1))
            if mse_vals.count() > 1
            else float("nan"),
            "NLL_norm_mean": float(np.nanmean(nll_vals)),
            "NLL_norm_std": float(np.nanstd(nll_vals, ddof=1))
            if nll_vals.count() > 1
            else float("nan"),
        }
        per_output_rows.append(row_out)

    per_output_df = pd.DataFrame(
        per_output_rows,
        columns=[
            "output",
            "R2_mean",
            "R2_std",
            "MSE_mean",
            "MSE_std",
            "NLL_norm_mean",
            "NLL_norm_std",
        ],
    )
    # -------------------------------------------------------------

    # Cross-run mean/std (numeric columns only) for per_run sheet
    numeric_cols = per_run_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    mean_row: Dict[str, Any] = {
        col: float(per_run_df[col].mean()) for col in numeric_cols
    }
    std_row: Dict[str, Any] = {
        col: float(per_run_df[col].std(ddof=1)) for col in numeric_cols
    }
    mean_row["Run"] = "MEAN_across_runs"
    mean_row["Seed"] = float("nan")
    std_row["Run"] = "STD_across_runs"
    std_row["Seed"] = float("nan")

    per_run_df = pd.concat(
        [per_run_df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
        ignore_index=True,
    )

    # README + SETTINGS sheets
    desc_map = _col_desc_map()
    readme_df = pd.DataFrame(
        {
            "column": list(desc_map.keys()),
            "description": list(desc_map.values()),
        }
    )
    settings_df = pd.DataFrame(
        {
            "param": [
                "n_runs",
                "base_seed",
                "k_folds",
                "patience",
                "n_estimators",
                "hidden_layers",
                "lr",
                "batch",
                "epochs",
                "dropout",
                "activation",
                "model_dir",
                "file",
                "sheet",
            ],
            "value": [
                n_runs,
                base_seed,
                k_folds,
                patience,
                n_estimators,
                list(hidden_layers),
                lr,
                batch,
                epochs,
                dropout,
                activation,
                model_dir,
                file,
                sheet,
            ],
        }
    )

    with pd.ExcelWriter(excel_filename) as writer:
        per_run_df.to_excel(writer, sheet_name="per_run_stats", index=False)
        per_output_df.to_excel(
            writer, sheet_name="per_output_metrics", index=False
        )
        readme_df.to_excel(writer, sheet_name="README", index=False)
        settings_df.to_excel(writer, sheet_name="SETTINGS", index=False)

    abs_path = os.path.abspath(excel_filename)
    print(f"\n[OK] Saved DE Excel summary to: {abs_path}")
    return abs_path


# -------------------- Script entry (kept for standalone) --------------------
def main_process(
    file_path: str,
    sheet_name: str,
    *,
    repeat_runs: int = 0,
    excel_filename: Optional[str] = None,
) -> None:
    """
    If repeat_runs>0 and excel_filename provided, run repeated pipeline and export Excel.
    Else run the original single-run demo (train on train split, load fold-1, evaluate on held-out val, CSV outputs).
    """
    if repeat_runs and excel_filename:
        repeat_runs_to_excel(
            file=file_path,
            sheet=sheet_name,
            excel_filename=excel_filename,
            n_runs=repeat_runs,
            base_seed=1234,
            k_folds=5,
            patience=100,
            n_estimators=10,
            hidden_layers=(100, 50),
            lr=1e-3,
            batch=32,
            epochs=1000,
            dropout=0.1,
            activation="tanh",
            model_dir="./models/DE",
        )
        return

    # ---- Original single-run path (kept for parity) ----
    data = load_data(file_path, sheet_name)
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_name = [r"$\dot W_{elec}$", r"$\dot m$"]

    X, Y = split_data(data, input_cols, output_cols)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Training data shape: {X_train.shape}, {Y_train.shape}")

    # k-fold CV on the training split
    t_start = time.time()
    de_k_fold_cross_validation(
        X_train,
        Y_train,
        k=5,
        patience=100,
        n_estimators=10,
        hidden_layers=(100, 50),
        lr=1e-3,
        batch=32,
        epochs=1000,
        dropout=0.1,
        activation="tanh",
        model_dir="./models/DE",
    )
    t_end = time.time()
    print(f"\n[Info] k-fold CV training time: {t_end - t_start:.1f} seconds")

    # Load fold-1 models & scalers; evaluate on held-out validation split
    models: List[DeepEnsembleRegressor] = []
    for i in range(Y.shape[1]):
        models.append(
            DeepEnsembleRegressor.load(
                f"./models/DE/DE_model_fold1_output{i}.pt"
            )
        )
    x_scaler: StandardScaler = joblib.load("./models/DE/DE_x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load("./models/DE/DE_y_scaler_fold1.pkl")

    # Decomposition
    y_pred, y_std_tot, y_std_ale, y_std_epi, v_ale, v_epi, v_tot = (
        _de_predict_with_decomposition(models, x_scaler, y_scaler, X_val)
    )

    # quick NLLs (normalized)
    Y_val_norm = y_scaler.transform(Y_val)
    y_pred_norm = y_scaler.transform(y_pred)
    y_std_tot_norm = y_std_tot / cast(NDArray[np.float_], y_scaler.scale_)
    nll_gauss = [
        float(
            np.mean(
                gaussian_nll_vector(
                    Y_val_norm[:, j],
                    y_pred_norm[:, j],
                    y_std_tot_norm[:, j] ** 2,
                )
            )
        )
        for j in range(Y.shape[1])
    ]

    # Timing (single measurement)
    t0 = time.perf_counter()
    _ = _de_predict_with_decomposition(models, x_scaler, y_scaler, X_val)
    t1 = time.perf_counter()
    infer_ms_per_1000_overall = (
        (t1 - t0) * 1e3 * (1000.0 / max(1, X_val.shape[0]))
    )
    print(
        f"Inference overall: {infer_ms_per_1000_overall:.2f} ms / 1000 samples (single-shot, demo path)"
    )

    plot_validation_group(
        Y_val.astype(np.float64), y_pred, y_std_tot, output_name
    )

    # CSVs for single run (kept)
    os.makedirs("outputs", exist_ok=True)
    out_df = pd.DataFrame(
        np.hstack(
            [y_pred, y_std_tot, y_std_ale, y_std_epi, v_ale, v_epi, v_tot]
        ),
        columns=[
            "y1_pred",
            "y2_pred",
            "y1_std_total",
            "y2_std_total",
            "y1_std_ale",
            "y2_std_ale",
            "y1_std_epi",
            "y2_std_epi",
            "y1_var_ale",
            "y2_var_ale",
            "y1_var_epi",
            "y2_var_epi",
            "y1_var_total",
            "y2_var_total",
        ],
    )
    out_df.to_csv("./outputs/DE_uncertainty_decomposition.csv", index=False)
    pd.DataFrame(
        {"output": output_cols, "nll_gaussian_norm": nll_gauss}
    ).to_csv("./outputs/DE_nll_summary.csv", index=False)
    print("\nSaved:")
    print("  models/DE/DE_model_fold1_output*.pt (and scalers)")
    print("  outputs/DE_uncertainty_decomposition.csv")
    print("  outputs/DE_nll_summary.csv")


# -------------------- Script entry --------------------
if __name__ == "__main__":
    file_path = "./data/case_2_scroll.csv"
    sheet_name = "case_2_scroll"
    # file_path = "./data/case_1_rotary.csv"
    # sheet_name = "case_1_rotary"

    # Option A: 原始单次运行（保持兼容）
    main_process(file_path, sheet_name)

    # Option B: 多次运行 + 导出 Excel（默认 10 次）
    # main_process(
    #     file_path,
    #     sheet_name,
    #     repeat_runs=10,
    #     excel_filename=f"outputs/DE_repeat_{sheet_name}.xlsx",
    # )

# Deep Ensemble Learning (DEL) with aleatoric + epistemic UQ
# (VI scroll compressor) - repeated runs & Excel export

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# NumPy 2 removed np.float_; preserve alias used in runtime typing casts.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

# ---------------------- Reproducibility ---------------------- #
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# ---------------------- Canonical IO + Aliases ---------------------- #
INPUT_ORDER: List[str] = [
    "P_suc",
    "T_suc",
    "T_dis",
    "P_dis",
    "T_inj",
    "P_inj",
    "frequency",
]
OUTPUT_ORDER: List[str] = ["m_inj", "m_suc", "W_total"]  # internal order

INPUT_ALIASES: Dict[str, Sequence[str]] = {
    "P_suc": [
        "P_suc",
        "Psuc",
        "P_suction",
        "Suction_P",
        "P_suc_bar",
        "P_suc[kPa]",
    ],
    "T_suc": ["T_suc", "Tsuc", "T_suction", "Suction_T"],
    "T_dis": ["T_dis", "Tdis", "T_discharge", "Discharge_T"],
    "P_dis": ["P_dis", "Pdis", "P_discharge", "Discharge_P"],
    "T_inj": ["T_inj", "Tinj", "T_injection", "Injection_T", "T_mid"],
    "P_inj": ["P_inj", "Pinj", "P_injection", "Injection_P", "P_mid"],
    "frequency": ["frequency", "freq", "f", "rpm", "Hz"],
}

OUTPUT_ALIASES: Dict[str, Sequence[str]] = {
    "m_inj": ["m_inj", "mdot_inj", "m_dot_inj", "m_injection"],
    "m_suc": ["m_suc", "mdot_suc", "m_dot_suc", "m_total", "m_suction"],
    "W_total": ["W_total", "W", "W_total_kW", "Wdot", "W_dot", "Power", "P_in"],
}

# LaTeX-friendly titles for plots
OUTPUT_TITLES: List[str] = [
    r"$\dot m_{\mathrm{inj}}$",
    r"$\dot m_{\mathrm{suc}}$",
    r"$\dot W_{\mathrm{total}}$",
]


# ---------------------- Utilities ---------------------- #
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


def resolve_columns(
    df: pd.DataFrame,
    desired_to_aliases: Dict[str, Sequence[str]],
    strict: bool = True,
) -> Dict[str, str]:
    col_lut = {c.lower(): c for c in df.columns}
    mapping: Dict[str, str] = {}
    for desired, aliases in desired_to_aliases.items():
        found: Optional[str] = None
        for a in aliases:
            key = a.lower()
            if key in col_lut:
                found = col_lut[key]
                break
        if found is None:
            msg = f"Required column '{desired}' not found. Tried aliases: {aliases}"
            if strict:
                raise KeyError(msg)
            else:
                print("[warn]", msg)
        else:
            mapping[desired] = found
    return mapping


def split_data(
    data: pd.DataFrame, input_cols: Sequence[str], output_cols: Sequence[str]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    x = data[list(input_cols)].to_numpy(dtype=np.float64, copy=True)
    y = data[list(output_cols)].to_numpy(dtype=np.float64, copy=True)
    return x, y


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


# ---------------------- Plotting ---------------------- #
def _auto_grid(n_plots: int) -> Tuple[int, int]:
    cols = min(3, max(1, n_plots))
    rows = int(np.ceil(n_plots / cols))
    return rows, cols


def plot_validation_group(
    y_valid: NDArray[np.float_],
    y_pred: NDArray[np.float_],
    y_std_total: NDArray[np.float_],
    y_name: Sequence[str],
) -> None:
    rows, cols = _auto_grid(len(y_name))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4.5 * rows))
    axes_arr = (
        np.array(axes).reshape(-1)
        if isinstance(axes, np.ndarray)
        else np.array([axes])
    )

    for i, _ in enumerate(y_name):
        ax = axes_arr[i]
        y_min, y_max = float(y_valid[:, i].min()), float(y_valid[:, i].max())
        xmin, xmax = max(0.0, 0.95 * y_min), 1.05 * y_max

        ax.errorbar(
            y_valid[:, i],
            y_pred[:, i],
            yerr=y_std_total[:, i],
            fmt="o",
            alpha=0.55,
            ecolor="lightgray",
        )
        ax.plot([xmin, xmax], [xmin, xmax], lw=1)
        x_line = np.linspace(xmin, xmax, 200, dtype=np.float64)
        ax.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.25, label="±10%"
        )

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
        ax.set_xlabel(f"True {y_name[i]}")
        ax.set_ylabel(f"Pred {y_name[i]}")
        ax.set_title(f"{y_name[i]} Prediction (Total UQ)")
        ax.grid(True, alpha=0.25)

    for j in range(len(y_name), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    plt.show()


def plot_loss_history(history: List[float], name: str) -> None:
    if not history:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Fold")
    plt.ylabel(name)
    plt.title(name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------- NLLs ---------------------- #
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
    mus: NDArray[np.float_],  # [M, N]
    vars_: NDArray[np.float_],  # [M, N]
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


# ---------------------- Network ---------------------- #
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


# ---------------------- Deep Ensemble ---------------------- #
class DeepEnsembleRegressor:
    """
    Each base model outputs [mu, log_var] for a SCALAR target.
    Train with Gaussian NLL: 0.5 * ((y-mu)^2 * exp(-s) + s)
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
        self.output_dim = 2  # [mu, log_var]
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
        Xv_t: Optional[torch.Tensor] = None
        Yv_t: Optional[torch.Tensor] = None
        if has_val:
            Xv_t = torch.tensor(
                cast(NDArray[np.float_], X_val),
                dtype=torch.float32,
                device=self.device,
            )
            Yv_t = torch.tensor(
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

                if has_val and Xv_t is not None and Yv_t is not None:
                    model.eval()
                    with torch.no_grad():
                        outv = model(Xv_t)
                        muv = outv[:, :1]
                        sv = torch.clamp(
                            outv[:, 1:2],
                            min=self.logvar_clamp[0],
                            max=self.logvar_clamp[1],
                        )
                        vloss = float(
                            self._gaussian_nll(muv, sv, Yv_t).mean().item()
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
        mus, vars_ = self.predict_components(X)  # [M,N], [M,N]
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


# ---------------------- K-fold CV ---------------------- #
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
    history_nll_g: List[float] = []

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

        # Metrics (normalized space)
        Y_pred = np.zeros_like(Yva)
        nll_g_list: List[float] = []
        for out_i, ens in enumerate(models):
            mu, std_tot, _, _ = ens.predict(Xva)
            Y_pred[:, out_i] = mu
            nll_vec = gaussian_nll_vector(
                Yva[:, out_i], mu, std_tot**2, add_const_2pi=False
            )
            nll_g_list.append(float(np.mean(nll_vec)))

        mape = float(mean_absolute_percentage_error(Yva, Y_pred))
        mse = float(mean_squared_error(Yva, Y_pred))
        r2 = float(r2_score(Yva, Y_pred))
        nll_g = float(np.mean(nll_g_list))
        print(
            f"Fold-{fold}  MAPE={mape:.4f}  MSE={mse:.4f}  R2={r2:.4f}  NLL(Gauss)={nll_g:.6f}"
        )

        history_mape.append(mape)
        history_mse.append(mse)
        history_r2.append(r2)
        history_nll_g.append(nll_g)
        fold += 1

    plot_loss_history(history_mape, "MAPE (norm)")
    plot_loss_history(history_mse, "MSE (norm)")
    plot_loss_history(history_r2, "R2 (norm)")
    plot_loss_history(history_nll_g, "Gaussian NLL (norm)")


# ---------------------- Helpers for repeated runs ---------------------- #
def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _train_and_predict_once(
    df: pd.DataFrame,
    *,
    split_seed: int,
    hidden_layers: Sequence[int] = (100, 50),
    n_estimators: int = 10,
    lr: float = 1e-3,
    batch: int = 32,
    epochs: int = 1000,
    patience: int = 100,
    dropout: float = 0.1,
    activation: str = "tanh",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[float],
    List[float],
    List[float],
]:
    """Single run: train with CV on train split; load fold-1; predict on val.
    Returns:
      (Y_val, y_pred, y_std_tot, y_std_ale, y_std_epi,
       r2_each, mse_each, nll_each_norm)
      where R2/MSE are ORIGINAL units; NLL is in NORMALIZED space.
    """
    # Resolve columns
    in_map = resolve_columns(df, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(df, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]

    X, Y = split_data(df, input_cols, output_cols)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=split_seed, shuffle=True
    )

    # Train on TRAIN split only (saves fold-1 artifacts)
    de_k_fold_cross_validation(
        X_train,
        Y_train,
        k=5,
        patience=patience,
        n_estimators=n_estimators,
        hidden_layers=hidden_layers,
        lr=lr,
        batch=batch,
        epochs=epochs,
        dropout=dropout,
        activation=activation,
    )

    # Load fold-1 artifacts
    models: List[DeepEnsembleRegressor] = []
    for i in range(Y.shape[1]):
        models.append(
            DeepEnsembleRegressor.load(
                f"./models/DE/DE_model_fold1_output{i}.pt"
            )
        )
    x_scaler: StandardScaler = joblib.load("./models/DE/DE_x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load("./models/DE/DE_y_scaler_fold1.pkl")

    X_val_norm = x_scaler.transform(X_val).astype(np.float64)

    # Predict per output (normalized), then rescale
    n_val, n_out = X_val_norm.shape[0], Y.shape[1]
    y_pred_norm = np.zeros((n_val, n_out), dtype=np.float64)
    y_std_tot_norm = np.zeros_like(y_pred_norm)
    y_std_ale_norm = np.zeros_like(y_pred_norm)
    y_std_epi_norm = np.zeros_like(y_pred_norm)

    for j, ens in enumerate(models):
        mus, vars_ = ens.predict_components(X_val_norm)  # [M,N],[M,N]
        mu = np.mean(mus, axis=0, dtype=np.float64)
        var_ale = np.mean(vars_, axis=0, dtype=np.float64)
        var_epi = np.var(mus, axis=0, ddof=0)
        var_tot = var_ale + var_epi

        y_pred_norm[:, j] = mu
        y_std_tot_norm[:, j] = np.sqrt(np.maximum(var_tot, 1e-12))
        y_std_ale_norm[:, j] = np.sqrt(np.maximum(var_ale, 1e-12))
        y_std_epi_norm[:, j] = np.sqrt(np.maximum(var_epi, 1e-12))

    scales = cast(
        NDArray[np.float_], getattr(y_scaler, "scale_", np.ones(n_out))
    )
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_std_tot = y_std_tot_norm * scales.reshape(1, -1)
    y_std_ale = y_std_ale_norm * scales.reshape(1, -1)
    y_std_epi = y_std_epi_norm * scales.reshape(1, -1)

    # -------- NEW: per-output metrics --------
    r2_each: List[float] = [
        float(r2_score(Y_val[:, j], y_pred[:, j])) for j in range(n_out)
    ]
    mse_each: List[float] = [
        float(mean_squared_error(Y_val[:, j], y_pred[:, j]))
        for j in range(n_out)
    ]
    # NLL in NORMALIZED space
    Y_val_norm = y_scaler.transform(Y_val).astype(np.float64)
    nll_each_norm: List[float] = []
    for j in range(n_out):
        nll_vec = gaussian_nll_vector(
            Y_val_norm[:, j],
            y_pred_norm[:, j],
            y_std_tot_norm[:, j] ** 2,
            add_const_2pi=False,
        )
        nll_each_norm.append(float(np.mean(nll_vec)))
    # ----------------------------------------

    return (
        Y_val.astype(np.float64),
        y_pred,
        y_std_tot,
        y_std_ale,
        y_std_epi,
        r2_each,
        mse_each,
        nll_each_norm,
    )


def repeat_runs_to_excel(
    file_path: str,
    sheet_name: str,
    *,
    excel_filename: str,
    n_runs: int = 10,
    base_seed: int = 1234,
    hidden_layers: Sequence[int] = (100, 50),
    n_estimators: int = 10,
    lr: float = 1e-3,
    batch: int = 32,
    epochs: int = 1000,
    patience: int = 100,
    dropout: float = 0.1,
    activation: str = "tanh",
) -> str:
    """Repeat DEL training/eval and write per-run + across-run mean±std to Excel.

    Excel列顺序按题意：m_suc, m_inj, W_dot（内部 W_total）。
    """
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)
    df = load_data(file_path, sheet_name)

    # excel 变量名与内部索引的映射
    excel_labels = ["m_suc", "m_inj", "W_dot"]
    internal_name_for_excel = {
        "m_suc": "m_suc",
        "m_inj": "m_inj",
        "W_dot": "W_total",
    }
    internal_index = {"m_inj": 0, "m_suc": 1, "W_total": 2}

    per_run_rows: List[Dict[str, Any]] = []

    for r in range(1, n_runs + 1):
        seed = base_seed + 100 * r
        print(f"\n=== Repeated run {r}/{n_runs} (seed={seed}) ===")
        _seed_all(seed)

        (
            Y_val,
            y_pred,
            y_std_tot,
            y_std_ale,
            y_std_epi,
            r2_each,
            mse_each,
            nll_each_norm,
        ) = _train_and_predict_once(
            df,
            split_seed=seed,
            hidden_layers=hidden_layers,
            n_estimators=n_estimators,
            lr=lr,
            batch=batch,
            epochs=epochs,
            patience=patience,
            dropout=dropout,
            activation=activation,
        )

        # 按验证集样本进行聚合（均值 & 样本内标准差）
        row: Dict[str, Any] = {
            "Run": r,
            "Seed": seed,
            "N_val": int(Y_val.shape[0]),
        }

        def _add_stats(label_excel: str) -> None:
            internal_key = internal_name_for_excel[label_excel]
            j = internal_index[internal_key]
            # 预测值的样本平均与样本标准差
            row[f"{label_excel}_pred_mean"] = float(np.mean(y_pred[:, j]))
            row[f"{label_excel}_pred_std_within_run"] = float(
                np.std(y_pred[:, j], ddof=1)
            )
            # 不确定性均值（样本平均）
            row[f"{label_excel}_std_total_mean"] = float(
                np.mean(y_std_tot[:, j])
            )
            row[f"{label_excel}_std_ale_mean"] = float(np.mean(y_std_ale[:, j]))
            row[f"{label_excel}_std_epi_mean"] = float(np.mean(y_std_epi[:, j]))

        for lbl in excel_labels:
            _add_stats(lbl)

        # -------- NEW: add per-output metrics to per-run row --------
        # Map indices back to excel label order: m_suc(1), m_inj(0), W_dot(2)
        metric_index_map = {
            "m_suc": internal_index["m_suc"],  # 1
            "m_inj": internal_index["m_inj"],  # 0
            "W_dot": internal_index["W_total"],  # 2
        }
        for lbl in excel_labels:
            j = metric_index_map[lbl]
            row[f"{lbl}_R2"] = (
                float(r2_each[j]) if j < len(r2_each) else float("nan")
            )
            row[f"{lbl}_MSE"] = (
                float(mse_each[j]) if j < len(mse_each) else float("nan")
            )
            row[f"{lbl}_NLL_norm"] = (
                float(nll_each_norm[j])
                if j < len(nll_each_norm)
                else float("nan")
            )

        # across-outputs summary (within this run)
        row["R2_mean"] = (
            float(np.mean(r2_each)) if len(r2_each) else float("nan")
        )
        row["R2_std"] = (
            float(np.std(r2_each, ddof=1)) if len(r2_each) > 1 else float("nan")
        )
        row["MSE_mean"] = (
            float(np.mean(mse_each)) if len(mse_each) else float("nan")
        )
        row["MSE_std"] = (
            float(np.std(mse_each, ddof=1))
            if len(mse_each) > 1
            else float("nan")
        )
        row["NLL_norm_mean"] = (
            float(np.mean(nll_each_norm))
            if len(nll_each_norm)
            else float("nan")
        )
        row["NLL_norm_std"] = (
            float(np.std(nll_each_norm, ddof=1))
            if len(nll_each_norm) > 1
            else float("nan")
        )
        # ------------------------------------------------------------

        per_run_rows.append(row)

    per_run_df = pd.DataFrame(per_run_rows)

    # 计算跨运行的均值与标准差，并追加两行（对所有数值列）
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

    # 另外给一个 summary 工作表：把每个数值列拆成 (mean, std) 便于阅读（排除最后两行）
    summary_rows: List[Dict[str, Any]] = []
    base_df = per_run_df.iloc[:-2] if len(per_run_df) >= 2 else per_run_df
    for col in numeric_cols:
        summary_rows.append(
            {
                "metric": col,
                "MEAN_across_runs": float(base_df[col].mean()),
                "STD_across_runs": float(base_df[col].std(ddof=1)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # -------- NEW: per-output cross-run mean/std sheet --------
    def _series_clean(col: str) -> pd.Series:
        mask_numeric_run = per_run_df["Run"].apply(
            lambda v: isinstance(v, (int, np.integer))
        )
        return pd.to_numeric(
            per_run_df.loc[mask_numeric_run, col], errors="coerce"
        ).dropna()

    per_output_rows: List[Dict[str, Any]] = []
    for out_lbl in ["m_suc", "m_inj", "W_dot"]:
        s_r2 = _series_clean(f"{out_lbl}_R2")
        s_mse = _series_clean(f"{out_lbl}_MSE")
        s_nll = _series_clean(f"{out_lbl}_NLL_norm")
        per_output_rows.append(
            {
                "output": out_lbl,
                "R2_mean": float(s_r2.mean()) if len(s_r2) else float("nan"),
                "R2_std": float(s_r2.std(ddof=1))
                if len(s_r2) > 1
                else float("nan"),
                "MSE_mean": float(s_mse.mean()) if len(s_mse) else float("nan"),
                "MSE_std": float(s_mse.std(ddof=1))
                if len(s_mse) > 1
                else float("nan"),
                "NLL_norm_mean": float(s_nll.mean())
                if len(s_nll)
                else float("nan"),
                "NLL_norm_std": float(s_nll.std(ddof=1))
                if len(s_nll) > 1
                else float("nan"),
            }
        )
    per_output_df = pd.DataFrame(per_output_rows)
    # ---------------------------------------------------------

    with pd.ExcelWriter(excel_filename) as writer:
        per_run_df.to_excel(writer, index=False, sheet_name="per-run")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        per_output_df.to_excel(
            writer, index=False, sheet_name="per_output_metrics"
        )  # NEW

    print(f"\nExcel saved -> {excel_filename}")
    return excel_filename


# ---------------------- Main process ---------------------- #
def main_process(
    file_path: str,
    sheet_name: str,
    *,
    excel_filename: str,
    n_runs: int = 10,
    base_seed: int = 1234,
) -> None:
    """入口：只做重复运行并导出到 Excel（满足题目要求）。"""
    repeat_runs_to_excel(
        file_path,
        sheet_name,
        excel_filename=excel_filename,
        n_runs=n_runs,
        base_seed=base_seed,
        hidden_layers=(100, 50),
        n_estimators=10,
        lr=1e-3,
        batch=32,
        epochs=1000,
        patience=100,
        dropout=0.1,
        activation="relu",
    )


# ---------------------- Excel per-point writer (overlay hook) ---------------------- #
def _write_uq_excel_sheet(
    excel_path: str,
    sheet_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std_total: Optional[np.ndarray],
    y_std_ale: Optional[np.ndarray],
    y_std_epi: Optional[np.ndarray],
    output_order: List[str],
) -> None:
    """
    写入逐点 UQ 到指定 Excel 工作表（覆盖同名 sheet）。
    列：val_index, each *_true, *_pred, *_std_total, *_std_ale, *_std_epi
    """
    N, M = y_true.shape
    cols: List[str] = ["val_index"]
    mats: List[np.ndarray] = [np.arange(N, dtype=int).reshape(-1, 1)]

    for j, name in enumerate(output_order):
        cols.append(f"{name}_true")
        mats.append(y_true[:, [j]])
    for j, name in enumerate(output_order):
        cols.append(f"{name}_pred")
        mats.append(y_pred[:, [j]])

    def _add_block(tag: str, arr_opt: Optional[np.ndarray]) -> None:
        if arr_opt is None:
            for _j, name in enumerate(output_order):
                cols.append(f"{name}_{tag}")
                mats.append(np.full((N, 1), np.nan))
        else:
            for _j, name in enumerate(output_order):
                cols.append(f"{name}_{tag}")
                mats.append(arr_opt[:, [_j]])

    _add_block("std_total", y_std_total)
    _add_block("std_ale", y_std_ale)
    _add_block("std_epi", y_std_epi)

    df = pd.DataFrame(np.hstack(mats), columns=cols)
    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)
    if os.path.exists(excel_path):
        with pd.ExcelWriter(
            excel_path, mode="a", if_sheet_exists="replace", engine="openpyxl"
        ) as wr:
            df.to_excel(wr, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as wr:
            df.to_excel(wr, sheet_name=sheet_name, index=False)


# ---------------------- Unified overlay (kept) ---------------------- #
class _ValidationResult:
    """(保持原有) 供统一可视化脚本使用。"""

    def __init__(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_names: List[str],
        y_std: Optional[np.ndarray] = None,
        y_lo: Optional[np.ndarray] = None,
        y_hi: Optional[np.ndarray] = None,
        fig: Optional[Figure] = None,
        metrics: Optional[dict] = None,
        settings: Optional[dict] = None,
        # ==== 新增：与 overlay 长表写入对齐 ====
        y_std_ale: Optional[np.ndarray] = None,
        y_std_epi: Optional[np.ndarray] = None,
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
        # ==== 新增 ====
        self.y_std_ale = y_std_ale
        self.y_std_epi = y_std_epi

    def check(self) -> None:
        assert self.y_true.shape == self.y_pred.shape
        if self.y_std is not None:
            assert self.y_std.shape == self.y_pred.shape
        # ==== 新增 ====
        if self.y_std_ale is not None:
            assert self.y_std_ale.shape == self.y_pred.shape
        if self.y_std_epi is not None:
            assert self.y_std_epi.shape == self.y_pred.shape
        # ==============
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
    rows, cols = _auto_grid(len(y_name))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4.5 * rows))
    axes_arr = (
        np.array(axes).reshape(-1)
        if isinstance(axes, np.ndarray)
        else np.array([axes])
    )

    for i, _ in enumerate(y_name):
        ax = axes_arr[i]
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
        ax.set_title(f"{y_name[i]} Prediction (Total UQ)", fontsize=14)
        ax.grid(True, alpha=0.25)

    for j in range(len(y_name), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    return fig


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
    # ===== 新增：与 GPR/BNN 对齐的 overlay 参数 =====
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: Optional[str] = None,
) -> _ValidationResult:
    """保留：单次训练-加载-可视化的统一接口。"""
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    df = load_data(file, sheet)
    in_map = resolve_columns(df, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(df, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]

    X, Y = split_data(df, input_cols, output_cols)

    # 固定验证集（若提供）或正常划分
    if fixed_indices is not None:
        tr_idx, va_idx = fixed_indices
        X_train, X_val = X[tr_idx], X[va_idx]
        Y_train, Y_val = Y[tr_idx], Y[va_idx]
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, shuffle=True
        )

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

    models: List[DeepEnsembleRegressor] = []
    for i in range(Y.shape[1]):
        models.append(
            DeepEnsembleRegressor.load(
                f"{model_dir}/DE_model_fold1_output{i}.pt"
            )
        )
    x_scaler: StandardScaler = joblib.load(f"{model_dir}/DE_x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_dir}/DE_y_scaler_fold1.pkl")

    X_val_norm = x_scaler.transform(X_val).astype(np.float64)
    n_val, n_out = X_val_norm.shape[0], Y.shape[1]

    # ===== 修改：计算总/偶然/认知不确定性（规范化空间上分解，再放缩回原单位） =====
    y_pred_norm = np.zeros((n_val, n_out), dtype=np.float64)
    y_std_tot_norm = np.zeros_like(y_pred_norm)
    y_std_ale_norm = np.zeros_like(y_pred_norm)
    y_std_epi_norm = np.zeros_like(y_pred_norm)

    for j, ens in enumerate(models):
        mus, vars_ = ens.predict_components(X_val_norm)  # [M,N],[M,N]
        mu = np.mean(mus, axis=0, dtype=np.float64)
        var_ale = np.mean(vars_, axis=0, dtype=np.float64)
        var_epi = np.var(mus, axis=0, ddof=0)
        var_tot = var_ale + var_epi

        y_pred_norm[:, j] = mu
        y_std_tot_norm[:, j] = np.sqrt(np.maximum(var_tot, 1e-12))
        y_std_ale_norm[:, j] = np.sqrt(np.maximum(var_ale, 1e-12))
        y_std_epi_norm[:, j] = np.sqrt(np.maximum(var_epi, 1e-12))

    scales = cast(
        NDArray[np.float_], getattr(y_scaler, "scale_", np.ones(n_out))
    )
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_std = y_std_tot_norm * scales.reshape(1, -1)
    y_std_ale = y_std_ale_norm * scales.reshape(1, -1)
    y_std_epi = y_std_epi_norm * scales.reshape(1, -1)

    fig = _make_validation_figure(
        Y_val.astype(np.float64), y_pred, y_std, OUTPUT_TITLES
    )

    mse_each = [
        mean_squared_error(Y_val[:, j], y_pred[:, j]) for j in range(n_out)
    ]
    mape_each = [
        mean_absolute_percentage_error(Y_val[:, j], y_pred[:, j])
        for j in range(n_out)
    ]
    r2_each = [r2_score(Y_val[:, j], y_pred[:, j]) for j in range(n_out)]

    metrics = {
        "MSE_each": mse_each,
        "MAPE_each": mape_each,
        "R2_each": r2_each,
        "MSE_mean": float(np.mean(mse_each)),
        "MAPE_mean": float(np.mean(mape_each)),
        "R2_mean": float(np.mean(r2_each)),
        "n_estimators": n_estimators,
        "hidden_layers": list(hidden_layers),
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
    }

    vr = _ValidationResult(
    name="DE (Deep Ensemble)",
    y_true=Y_val.astype(np.float64),
    y_pred=y_pred,
    y_std=y_std,          # total
    y_lo=None,
    y_hi=None,
    output_names=OUTPUT_TITLES,
    fig=fig,
    metrics=metrics,
    settings=settings,
    # ==== 新增：让 overlay 的长表写入能读到 ====
    y_std_ale=y_std_ale,
    y_std_epi=y_std_epi,
)
    vr.check()

    # ===== 新增：可选 Excel 逐点写入（与 GPR/BNN 一致；列名用内部顺序 OUTPUT_ORDER） =====
    if (uq_excel_path is not None) and (uq_sheet_name is not None):
        _write_uq_excel_sheet(
            uq_excel_path,
            uq_sheet_name,
            vr.y_true,
            vr.y_pred,
            y_std,  # total
            y_std_ale,  # aleatory
            y_std_epi,  # epistemic
            OUTPUT_ORDER,
        )
        print(
            f"[DE] Wrote per-point UQ to '{uq_excel_path}' sheet='{uq_sheet_name}'"
        )

    return vr


# ---------------------- Script entry ---------------------- #
if __name__ == "__main__":
    # 用户可在这里自定义 Excel 文件名与重复次数
    file_path = "./data/case_3_vi.csv"
    sheet_name = "case_3_vi"

    excel_file = f"./outputs/DE_repeat_{sheet_name}.xlsx"
    # 方式 A：多次运行 + Excel 导出（推荐；默认重复 10 次）
    main_process(
        file_path,
        sheet_name,
        excel_filename=excel_file,
        n_runs=10,
        base_seed=1234,
    )

    # 方式 B：只跑一次并导出详细 CSV
    # main_process(file_path, sheet_name, excel_filename=None)

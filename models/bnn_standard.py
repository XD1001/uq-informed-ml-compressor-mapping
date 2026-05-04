# ===== Bayesian Neural Network (Pyro MCMC) with repeat runs + Excel export =====
# Compatible with previous GPR/DEL scripts' Excel schema; aligned to overlay

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pyro.infer import MCMC, NUTS, Predictive
from pyro.nn import PyroModule, PyroSample
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# NumPy 2 removed np.float_; preserve alias used in runtime typing casts.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

"""
Bayesian Neural Network (BNN) with PyTorch + Pyro for UQ.

Upgrades in this version (aligned with overlay/GPR/DE):
- Return ValidationResult-like container: y_std (total), y_std_ale, y_std_epi
- NLL computed in NORMALIZED space with TOTAL variance (matches GPR/DE)
- Inference time (ms / 1000 samples): mean + std in metrics (keys match GPR)
- Repeat runs to Excel with per_run_stats & per_output_metrics (field names align)
- (NEW) Save per-point UQ decomposition to a user-specified Excel (sheet append)
  via run_model_bnn(uq_excel_path=..., uq_sheet_name=...)
"""

# ---- Base seeds (each repeat uses seed offset) ----
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
pyro.set_rng_seed(1234)


# -------------------- I/O & preprocessing --------------------
def load_data(file: str, sheet: str) -> pd.DataFrame:
    if file.lower().endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file, sheet_name=sheet)
    if data.empty:
        raise ValueError("Loaded data is empty")
    return data.dropna().reset_index(drop=True)


def split_data(
    data: pd.DataFrame, input_cols: Sequence[str], output_cols: Sequence[str]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    x = data[list(input_cols)].to_numpy(dtype=np.float64, copy=True)
    y = data[list(output_cols)].to_numpy(dtype=np.float64, copy=True)
    return x, y


def normalize(
    x: NDArray[np.float_], y: NDArray[np.float_]
) -> Tuple[
    NDArray[np.float_], NDArray[np.float_], StandardScaler, StandardScaler
]:
    xs, ys = StandardScaler(), StandardScaler()
    x_n = xs.fit_transform(x.astype(np.float64))
    y_n = ys.fit_transform(y.astype(np.float64))
    return x_n.astype(np.float64), y_n.astype(np.float64), xs, ys


def denormalize(
    x_norm: NDArray[np.float_], scaler: StandardScaler
) -> NDArray[np.float_]:
    y = scaler.inverse_transform(x_norm)
    return np.asarray(y, dtype=np.float64)


def _bnn_meta_path(model_dir: str) -> str:
    return os.path.join(model_dir, "BNN_metadata.json")


def _save_bnn_artifacts(
    *,
    model_dir: str,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    posterior: Dict[str, torch.Tensor],
    layer_sizes: Sequence[int],
    activation: str,
    prior_scale: float,
    sigma_prior: str,
    dropout_rate: float,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    output_names: Sequence[str],
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(x_scaler, os.path.join(model_dir, "BNN_x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(model_dir, "BNN_y_scaler.pkl"))
    torch.save(
        {k: v.detach().cpu() for k, v in posterior.items()},
        os.path.join(model_dir, "BNN_posterior_samples.pt"),
    )
    meta = {
        "layer_sizes": [int(v) for v in layer_sizes],
        "activation": str(activation),
        "prior_scale": float(prior_scale),
        "sigma_prior": str(sigma_prior),
        "dropout_rate": float(dropout_rate),
        "input_cols": list(input_cols),
        "output_cols": list(output_cols),
        "output_names": list(output_names),
    }
    with open(_bnn_meta_path(model_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_bnn_predictive_from_cache(
    model_dir: str = "./models/BNN",
) -> Tuple[Predictive, StandardScaler, StandardScaler, Dict[str, Any]]:
    meta_path = _bnn_meta_path(model_dir)
    post_path = os.path.join(model_dir, "BNN_posterior_samples.pt")
    x_scaler_path = os.path.join(model_dir, "BNN_x_scaler.pkl")
    y_scaler_path = os.path.join(model_dir, "BNN_y_scaler.pkl")

    for p in [meta_path, post_path, x_scaler_path, y_scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing BNN cache artifact: {p}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = cast(Dict[str, Any], json.load(f))

    model = BayesianNN(
        layer_sizes=[int(v) for v in meta["layer_sizes"]],
        activation=str(meta["activation"]),
        prior_scale=float(meta["prior_scale"]),
        sigma_prior=str(meta["sigma_prior"]),
        dropout_rate=float(meta["dropout_rate"]),
    )
    posterior = cast(
        Dict[str, torch.Tensor],
        torch.load(post_path, map_location="cpu"),
    )
    predictive = Predictive(
        model,
        posterior_samples=posterior,
        return_sites=["_RETURN", "sigma"],
    )
    x_scaler = cast(StandardScaler, joblib.load(x_scaler_path))
    y_scaler = cast(StandardScaler, joblib.load(y_scaler_path))
    return predictive, x_scaler, y_scaler, meta


def predict_with_cached_bnn(
    X_raw: NDArray[np.float_],
    model_dir: str = "./models/BNN",
) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], Dict[str, Any]]:
    predictive, x_scaler, y_scaler, meta = load_bnn_predictive_from_cache(model_dir)
    X_norm = x_scaler.transform(np.asarray(X_raw, dtype=np.float64))
    sites = predictive(torch.from_numpy(X_norm).float())
    mus = np.asarray(sites["_RETURN"].detach().cpu().numpy(), dtype=np.float64)
    sigmas = np.asarray(sites["sigma"].detach().cpu().numpy(), dtype=np.float64)

    mu_mean_norm = mus.mean(axis=0)
    epi_var_norm = mus.var(axis=0, ddof=0)
    ale_var_vec_norm = (sigmas**2).mean(axis=0)
    ale_var_norm = np.broadcast_to(
        ale_var_vec_norm.reshape(1, -1), mu_mean_norm.shape
    )
    total_var_norm = ale_var_norm + epi_var_norm

    y_pred = denormalize(mu_mean_norm, y_scaler)
    scales = np.asarray(
        getattr(y_scaler, "scale_", np.ones(y_pred.shape[1])), dtype=np.float64
    ).reshape(1, -1)
    y_std_total = np.sqrt(total_var_norm) * scales
    y_std_ale = np.sqrt(ale_var_norm) * scales
    y_std_epi = np.sqrt(epi_var_norm) * scales
    return y_pred, y_std_total, y_std_ale, y_std_epi, meta


# -------------------- Plot helpers --------------------
def plot_validation_group(
    y_valid: NDArray[np.float_],
    y_pred: NDArray[np.float_],
    y_std: NDArray[np.float_],
    y_name: Sequence[str],
    save: bool = False,
) -> None:
    plt.figure(figsize=(10, 5))
    for i, _ in enumerate(y_name):
        plt.subplot(1, 2, i + 1)
        plt.errorbar(
            y_valid[:, i],
            y_pred[:, i],
            yerr=y_std[:, i],
            fmt="o",
            alpha=0.5,
            ecolor="lightgray",
        )
        min_val = float(y_valid[:, i].min())
        max_val = float(y_valid[:, i].max())
        xmin, xmax = max(0.0, 0.95 * min_val), 1.05 * max_val
        plt.plot([xmin, xmax], [xmin, xmax], color="blue", lw=1)
        x = np.linspace(xmin, max(xmax, 1e-12), 200, dtype=np.float64)
        plt.fill_between(
            x, 0.9 * x, 1.1 * x, color="lightgray", alpha=0.4, label="$±10%$"
        )
        plt.text(
            0.85 * xmax,
            0.65 * xmax,
            f"$R^2$={r2_score(y_valid[:, i], y_pred[:, i]):.3f}\n"
            f"MSE={mean_squared_error(y_valid[:, i], y_pred[:, i]):.3f}\n"
            f"MAPE={mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i]):.3f}",
            fontsize=11,
            ha="right",
        )
        plt.xlim(xmin, xmax)
        plt.ylim(xmin, xmax)
        plt.xlabel(f"True {y_name[i]}")
        plt.ylabel(f"Pred {y_name[i]}")
        plt.title(f"{y_name[i]} Prediction with Uncertainty")
    plt.tight_layout()
    if save:
        os.makedirs("figure", exist_ok=True)
        plt.savefig(
            f"figure/bnn_validation_{y_name[0]}_{y_name[-1]}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    plt.show()


# -------------------- BNN (Pyro) --------------------
class BayesianNN(PyroModule):
    """
    Simple fully-connected BNN with Normal priors and per-output homoscedastic noise sigma.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        prior_scale: float = 5.0,
        sigma_prior: str = "uniform",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output layers"
        self.activation = self._get_activation(activation)
        self.sigma_prior = sigma_prior
        self.dropout = nn.Dropout(dropout_rate)

        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = int(layer_sizes[i]), int(layer_sizes[i + 1])
            lin = cast(Any, PyroModule[nn.Linear])(in_dim, out_dim)
            # Priors
            lin.weight = PyroSample(
                dist.Normal(0.0, prior_scale * np.sqrt(2.0 / max(1, in_dim)))
                .expand([out_dim, in_dim])   # type: ignore
                .to_event(2)
            )
            lin.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([out_dim]).to_event(1)   # type: ignore
            )
            layers.append(lin)
        self.layers = cast(Any, PyroModule[nn.ModuleList])(layers)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.activation(self.layers[0](x))
        h = self.dropout(h)
        for li in range(1, len(self.layers) - 1):
            h = self.activation(self.layers[li](h))
            h = self.dropout(h)
        mu = self.layers[-1](h)

        # per-output observation noise prior
        out_dim = int(mu.shape[-1])
        if self.sigma_prior == "uniform":
            sigma = torch.abs(
                pyro.sample(
                    "sigma",
                    dist.Uniform(1e-5, 0.25).expand([out_dim]).to_event(1),
                )
            )
        else:
            sigma = torch.abs(
                pyro.sample(
                    "sigma", dist.HalfCauchy(1.0).expand([out_dim]).to_event(1)
                )
            )

        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
        return mu  # Predictive will collect this under "_RETURN"

    @staticmethod
    def _get_activation(act: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
        }.get(act, nn.Tanh())


# -------------------- Validation container（与 overlay 对齐） --------------------
class _ValidationResult:
    """
    Duck-typed container aligned with GPR/DEL versions.
    """

    def __init__(
        self,
        name: str,
        y_true: NDArray[np.float_],
        y_pred: NDArray[np.float_],
        output_names: List[str],
        y_std: Optional[NDArray[np.float_]] = None,  # total std (orig units)
        y_lo: Optional[NDArray[np.float_]] = None,
        y_hi: Optional[NDArray[np.float_]] = None,
        fig: Optional[Figure] = None,
        metrics: Optional[dict] = None,
        settings: Optional[dict] = None,
        *,
        y_std_ale: Optional[NDArray[np.float_]] = None,  # NEW
        y_std_epi: Optional[NDArray[np.float_]] = None,  # NEW
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
    y_valid: NDArray[np.float_],
    y_pred: NDArray[np.float_],
    y_std: Optional[NDArray[np.float_]],
    y_name: List[str],
) -> Figure:
    fig, axes = plt.subplots(1, len(y_name), figsize=(10, 5))
    if len(y_name) == 1:
        axes = [axes]  # type: ignore[list-item]
    for i, _ in enumerate(y_name):
        ax = axes[i]  # type: ignore[index]
        min_val, max_val = (
            float(y_valid[:, i].min()),
            float(y_valid[:, i].max()),
        )
        xmin, xmax = max(0.0, 0.95 * min_val), 1.05 * max_val
        x_line = np.linspace(xmin, max(xmax, 1e-12), 200, dtype=np.float64)
        ax.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.4, label="$±10%$"
        )
        ax.plot([xmin, xmax], [xmin, xmax], lw=1)
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
            ax.scatter(y_valid[:, i], y_pred[:, i], alpha=0.8)
        r2 = r2_score(y_valid[:, i], y_pred[:, i])
        mse = mean_squared_error(y_valid[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i])
        ax.text(
            0.85 * xmax,
            0.65 * xmax,
            f"$R^2$={r2:.3f}\nMSE={mse:.3f}\nMAPE={mape:.3f}",
            fontsize=12,
            ha="right",
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel(f"True {y_name[i]}", fontsize=14)
        ax.set_ylabel(f"Pred {y_name[i]}", fontsize=14)
        ax.set_title(
            f"{y_name[i]} Prediction with Uncertainty (total)", fontsize=14
        )
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# -------------------- 统一的推理计时（与 GPR/DE 对齐，单位：ms/1000 样本） --------------------
def _make_1000_samples(
    X: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(2025)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty X for timing.")
    if n >= n_samples:
        idx = rng.choice(n, size=n_samples, replace=False)
    else:
        idx = rng.choice(n, size=n_samples, replace=True)
    return X[idx]


def _bnn_single_predict_norm(
    predictive: Predictive, X_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Xt = torch.from_numpy(X_norm).float()
    sites = predictive(Xt)
    mus = sites["_RETURN"].detach().cpu().numpy()  # (S, N, O)
    sigmas = sites["sigma"].detach().cpu().numpy()  # (S, O)
    return mus, sigmas


def measure_inference_time_bnn(
    predictive: Predictive,
    x_scaler: StandardScaler,
    X_ref_raw: np.ndarray,
    *,
    repeats: int = 30,
    warmup: int = 5,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    Xb = _make_1000_samples(X_ref_raw, n_samples=n_samples, rng=rng)
    Xb_norm = x_scaler.transform(Xb).astype(np.float64)

    # warmup
    for _ in range(warmup):
        _ = _bnn_single_predict_norm(predictive, Xb_norm)

    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = _bnn_single_predict_norm(predictive, Xb_norm)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms, ddof=1)) if repeats > 1 else 0.0
    return mean_ms, std_ms


# -------------------- (NEW) Excel writing helpers for per-point UQ ------------
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


# -------------------- One-run interface（返回全部 UQ 组件） --------------------
def run_model_bnn(
    file: str,
    sheet: str,
    *,
    split_seed: int = 1234,
    layer_sizes: Optional[List[int]] = None,
    activation: str = "relu",
    prior_scale: float = 1.0,
    sigma_prior: str = "uniform",
    dropout_rate: float = 0.0,
    num_samples: int = 500,
    warmup_steps: int = 500,
    # 与 GPR 统一的推理计时参数
    infer_repeats: int = 30,
    infer_warmup: int = 5,
    infer_n_samples: int = 1000,
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    model_dir: str = "./models/BNN",
    # NEW: 保存逐点 UQ 到 Excel
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: str = "BNN_UQ",
) -> _ValidationResult:
    """
    Steps:
    1) Load & split (seeded). 2) Normalize. 3) Run MCMC. 4) Posterior predictive.
    5) Decompose UQ (ale/epi/total) and map back to ORIGINAL units. 6) (NEW) optional Excel.
    """
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_names = [r"$\dot W_{elec}$", r"$\dot m$"]

    # Seeding per run
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    pyro.set_rng_seed(split_seed)

    # 1) Load & split
    data = load_data(file, sheet)
    x, y = split_data(data, input_cols, output_cols)
    if fixed_indices is not None:
        train_idx, val_idx = fixed_indices
        x_train, x_valid = x[train_idx], x[val_idx]
        y_train, y_valid = y[train_idx], y[val_idx]
    else:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=split_seed, shuffle=True
        )

    # 2) Normalize
    x_train_norm, y_train_norm, x_scaler, y_scaler = normalize(x_train, y_train)
    x_valid_norm = x_scaler.transform(x_valid).astype(np.float64)
    y_valid_norm = y_scaler.transform(y_valid).astype(np.float64)

    # Tensors
    x_train_t = torch.from_numpy(x_train_norm).float()
    y_train_t = torch.from_numpy(y_train_norm).float()

    # Default architecture
    if layer_sizes is None:
        layer_sizes = [
            int(x_train_norm.shape[1]),
            20,
            10,
            int(y_train_norm.shape[1]),
        ]

    # 3) Build & MCMC
    bnn = BayesianNN(
        layer_sizes=layer_sizes,
        activation=activation,
        prior_scale=prior_scale,
        sigma_prior=sigma_prior,
        dropout_rate=dropout_rate,
    )
    pyro.clear_param_store()
    nuts = NUTS(bnn)  # type: ignore[call-arg]
    mcmc = MCMC(
        nuts, num_samples=int(num_samples), warmup_steps=int(warmup_steps)
    )
    mcmc.run(x_train_t, y_train_t)

    posterior = mcmc.get_samples()
    predictive = Predictive(
        bnn, posterior_samples=posterior, return_sites=["_RETURN", "sigma"]
    )
    _save_bnn_artifacts(
        model_dir=model_dir,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        posterior=posterior,
        layer_sizes=layer_sizes,
        activation=activation,
        prior_scale=prior_scale,
        sigma_prior=sigma_prior,
        dropout_rate=dropout_rate,
        input_cols=input_cols,
        output_cols=output_cols,
        output_names=output_names,
    )

    # 推理计时（ms / 1000 样本）
    infer_mean_ms, infer_std_ms = measure_inference_time_bnn(
        predictive,
        x_scaler,
        x_valid,  # 在验证集分布上计时
        repeats=infer_repeats,
        warmup=infer_warmup,
        n_samples=infer_n_samples,
    )

    # 4) Posterior predictive on validation
    sites = predictive(torch.from_numpy(x_valid_norm).float())
    mus = sites["_RETURN"].detach().cpu().numpy()  # (S, N, O)
    sigmas = sites["sigma"].detach().cpu().numpy()  # (S, O)

    # 5) UQ decomposition (NORMALIZED space)
    mu_mean_norm = mus.mean(axis=0)  # (N, O)
    epi_var_norm = mus.var(axis=0, ddof=0)  # (N, O)
    ale_var_vec_norm = (sigmas**2).mean(axis=0)  # (O,)
    ale_var_norm = np.broadcast_to(
        ale_var_vec_norm.reshape(1, -1), mu_mean_norm.shape
    )
    total_var_norm = ale_var_norm + epi_var_norm

    # Map to ORIGINAL units
    y_pred = denormalize(mu_mean_norm, y_scaler)
    scales = np.asarray(
        getattr(y_scaler, "scale_", np.ones(y.shape[1])), dtype=np.float64
    ).reshape(1, -1)
    y_std_total = np.sqrt(total_var_norm) * scales
    y_std_ale = np.sqrt(ale_var_norm) * scales
    y_std_epi = np.sqrt(epi_var_norm) * scales

    # Figure
    fig = _make_validation_figure(
        y_valid.astype(np.float64), y_pred, y_std_total, output_names
    )

    # Metrics（原尺度） + NLL（归一化）
    mse_each = [
        float(mean_squared_error(y_valid[:, j], y_pred[:, j]))
        for j in range(y_pred.shape[1])
    ]
    mape_each = [
        float(mean_absolute_percentage_error(y_valid[:, j], y_pred[:, j]))
        for j in range(y_pred.shape[1])
    ]
    r2_each = [
        float(r2_score(y_valid[:, j], y_pred[:, j]))
        for j in range(y_pred.shape[1])
    ]

    def _gaussian_nll_vec(
        y_true_norm: np.ndarray, y_mean_norm: np.ndarray, var_norm: np.ndarray
    ) -> float:
        eps = 1e-12
        v = np.maximum(var_norm, eps)
        term1 = 0.5 * np.log(v)  # match GPR/DE (no 0.5*log(2π))
        term2 = 0.5 * ((y_true_norm - y_mean_norm) ** 2) / v
        return float(np.mean(term1 + term2))

    nll_each_norm: List[float] = []
    for j in range(mu_mean_norm.shape[1]):
        nll_each_norm.append(
            _gaussian_nll_vec(
                y_valid_norm[:, j], mu_mean_norm[:, j], total_var_norm[:, j]
            )
        )

    metrics: Dict[str, Any] = {
        "MSE_each": mse_each,
        "MAPE_each": mape_each,
        "R2_each": r2_each,
        "NLL_each_norm": nll_each_norm,
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
        # 统一推理计时
        "Infer_ms_per_1000_mean": infer_mean_ms,
        "Infer_ms_per_1000_std": infer_std_ms,
        "Infer_repeats": infer_repeats,
        "Infer_warmup": infer_warmup,
        "Infer_n_samples": infer_n_samples,
    }
    settings = {
        "activation": activation,
        "prior_scale": prior_scale,
        "sigma_prior": sigma_prior,
        "dropout_rate": dropout_rate,
        "split_seed": split_seed,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "layer_sizes": list(layer_sizes),
        "model_dir": model_dir,
    }

    # -------- (NEW) save per-point UQ into Excel if requested --------
    if uq_excel_path is not None:
        uq_df = _build_uq_long_table(
            model_name="BNN (Pyro MCMC)",
            output_names=output_names,
            input_cols=input_cols,
            X_val_raw=x_valid,
            y_true=y_valid.astype(np.float64),
            y_pred=y_pred,
            y_std_total=y_std_total,
            y_std_epi=y_std_epi,
            y_std_ale=y_std_ale,
        )
        _write_df_to_excel(uq_excel_path, uq_sheet_name, uq_df)

    return _ValidationResult(
        name="BNN (Pyro MCMC)",
        y_true=y_valid.astype(np.float64),
        y_pred=y_pred,
        y_std=y_std_total,  # total
        y_std_ale=y_std_ale,
        y_std_epi=y_std_epi,
        y_lo=None,
        y_hi=None,
        output_names=output_names,
        fig=fig,
        metrics=metrics,
        settings=settings,
    )


# -------------------- Repeat runs + Excel export（保持与 GPR/DE 同步的字段） --------------------
def _col_desc_map() -> Dict[str, str]:
    return {
        "Run": "Run index starting from 1",
        "Seed": "Random seed used for train/val split",
        # Cross-output summaries
        "R2_mean": "Mean R^2 over outputs (original units) per run",
        "R2_std": "Std of R^2 over outputs (per run, ddof=1)",
        "MSE_mean": "Mean MSE over outputs (original units) per run",
        "MSE_std": "Std of MSE over outputs (per run, ddof=1)",
        "NLL_mean_norm": "Mean Gaussian NLL over outputs (normalized space) per run",
        "NLL_std_norm": "Std of Gaussian NLL over outputs (normalized, ddof=1)",
        # Inference time
        "Infer_ms_per_1000_mean": "End-to-end inference wall time over 1000 samples (mean over repeats); ms/1000 samples",
        "Infer_ms_per_1000_std": "Std of the above across repeats (ddof=1); ms/1000 samples",
        # W_dot uncertainty summary
        "W_dot_pred_mean": "Mean of predicted W_dot over validation samples",
        "W_dot_pred_std": "Std of predicted W_dot over validation samples",
        "W_dot_total_std_mean": "Mean of TOTAL predictive std over validation samples (original units)",
        "W_dot_total_std_std": "Std of TOTAL predictive std over validation samples",
        "W_dot_ale_std_mean": "Mean of ALEATORY predictive std over validation samples",
        "W_dot_ale_std_std": "Std of ALEATORY predictive std over validation samples",
        "W_dot_epi_std_mean": "Mean of EPISTEMIC predictive std over validation samples",
        "W_dot_epi_std_std": "Std of EPISTEMIC predictive std over validation samples",
        # m_dot
        "m_dot_pred_mean": "Mean of predicted m_dot over validation samples",
        "m_dot_pred_std": "Std of predicted m_dot over validation samples",
        "m_dot_total_std_mean": "Mean of TOTAL predictive std over validation samples (original units)",
        "m_dot_total_std_std": "Std of TOTAL predictive std over validation samples",
        "m_dot_ale_std_mean": "Mean of ALEATORY predictive std over validation samples",
        "m_dot_ale_std_std": "Std of ALEATORY predictive std over validation samples",
        "m_dot_epi_std_mean": "Mean of EPISTEMIC predictive std over validation samples",
        "m_dot_epi_std_std": "Std of EPISTEMIC predictive std over validation samples",
        # Per-output metrics (for per_output_metrics sheet)
        "W_dot_R2": "Per-output R^2 for W_dot on validation set (original units)",
        "W_dot_MSE": "Per-output MSE for W_dot on validation set (original units)",
        "W_dot_NLL_norm": "Per-output Gaussian NLL for W_dot on validation set (normalized space)",
        "m_dot_R2": "Per-output R^2 for m_dot on validation set (original units)",
        "m_dot_MSE": "Per-output MSE for m_dot on validation set (original units)",
        "m_dot_NLL_norm": "Per-output Gaussian NLL for m_dot on validation set (normalized space)",
    }


def _mean_std_over_samples(
    arr: NDArray[np.float_],
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    mean = np.mean(arr, axis=0)
    std = (
        np.std(arr, axis=0, ddof=1)
        if arr.shape[0] > 1
        else np.zeros(arr.shape[1], dtype=np.float64)
    )
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _build_per_output_metrics(
    per_output_acc: Dict[str, List[float]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for out_name in ["W_dot", "m_dot"]:
        r2_vals = np.asarray(
            per_output_acc.get(f"{out_name}_R2", []), dtype=np.float64
        )
        mse_vals = np.asarray(
            per_output_acc.get(f"{out_name}_MSE", []), dtype=np.float64
        )
        nll_vals = np.asarray(
            per_output_acc.get(f"{out_name}_NLL_norm", []), dtype=np.float64
        )
        rows.append(
            {
                "output": out_name,
                "R2_mean": float(np.mean(r2_vals))
                if r2_vals.size
                else float("nan"),
                "R2_std": float(np.std(r2_vals, ddof=1))
                if r2_vals.size > 1
                else float("nan"),
                "MSE_mean": float(np.mean(mse_vals))
                if mse_vals.size
                else float("nan"),
                "MSE_std": float(np.std(mse_vals, ddof=1))
                if mse_vals.size > 1
                else float("nan"),
                "NLL_norm_mean": float(np.mean(nll_vals))
                if nll_vals.size
                else float("nan"),
                "NLL_norm_std": float(np.std(nll_vals, ddof=1))
                if nll_vals.size > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def repeat_runs_to_excel(
    file: str,
    sheet: str,
    excel_filename: str,
    *,
    n_runs: int = 10,
    base_seed: int = 1234,
    layer_sizes: Optional[List[int]] = None,
    activation: str = "tanh",
    prior_scale: float = 1.0,
    sigma_prior: str = "uniform",
    dropout_rate: float = 0.0,
    num_samples: int = 1000,
    warmup_steps: int = 1000,
) -> str:
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seeds = [base_seed + i for i in range(n_runs)]

    per_output_acc: Dict[str, List[float]] = {
        "W_dot_R2": [],
        "W_dot_MSE": [],
        "W_dot_NLL_norm": [],
        "m_dot_R2": [],
        "m_dot_MSE": [],
        "m_dot_NLL_norm": [],
    }

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Repeat-BNN] Run {run_idx}/{n_runs}, seed={seed}")
        res = run_model_bnn(
            file=file,
            sheet=sheet,
            split_seed=seed,
            layer_sizes=layer_sizes,
            activation=activation,
            prior_scale=prior_scale,
            sigma_prior=sigma_prior,
            dropout_rate=dropout_rate,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            # 不写逐点表，避免重复运行文件过大
            uq_excel_path=None,
        )
        res.check()

        assert (
            res.y_std is not None
            and res.y_std_ale is not None
            and res.y_std_epi is not None
        )

        # collect per-output metrics
        mse_each: List[float] = cast(
            List[float], res.metrics.get("MSE_each", [])
        )
        r2_each: List[float] = cast(List[float], res.metrics.get("R2_each", []))
        nll_each_norm: List[float] = cast(
            List[float], res.metrics.get("NLL_each_norm", [])
        )
        if len(mse_each) >= 2 and len(r2_each) >= 2 and len(nll_each_norm) >= 2:
            per_output_acc["W_dot_MSE"].append(float(mse_each[0]))
            per_output_acc["W_dot_R2"].append(float(r2_each[0]))
            per_output_acc["W_dot_NLL_norm"].append(float(nll_each_norm[0]))
            per_output_acc["m_dot_MSE"].append(float(mse_each[1]))
            per_output_acc["m_dot_R2"].append(float(r2_each[1]))
            per_output_acc["m_dot_NLL_norm"].append(float(nll_each_norm[1]))

        # sample-wise -> per-run summary rows
        pred_mean, pred_std = _mean_std_over_samples(res.y_pred)
        total_mean, total_std = _mean_std_over_samples(res.y_std)
        ale_mean, ale_std = _mean_std_over_samples(res.y_std_ale)
        epi_mean, epi_std = _mean_std_over_samples(res.y_std_epi)

        row: Dict[str, Any] = {
            "Run": run_idx,
            "Seed": seed,
            # per-run metrics (across outputs)
            "R2_mean": float(
                res.metrics.get(
                    "R2_mean", np.mean(r2_each) if r2_each else np.nan
                )
            ),
            "R2_std": float(
                res.metrics.get(
                    "R2_std",
                    np.std(r2_each, ddof=1) if len(r2_each) > 1 else np.nan,
                )
            ),
            "MSE_mean": float(
                res.metrics.get(
                    "MSE_mean", np.mean(mse_each) if mse_each else np.nan
                )
            ),
            "MSE_std": float(
                res.metrics.get(
                    "MSE_std",
                    np.std(mse_each, ddof=1) if len(mse_each) > 1 else np.nan,
                )
            ),
            "NLL_mean_norm": float(
                res.metrics.get(
                    "NLL_mean_norm",
                    np.mean(nll_each_norm) if nll_each_norm else np.nan,
                )
            ),
            "NLL_std_norm": float(
                res.metrics.get(
                    "NLL_std_norm",
                    np.std(nll_each_norm, ddof=1)
                    if len(nll_each_norm) > 1
                    else np.nan,
                )
            ),
            # inference timing
            "Infer_ms_per_1000_mean": float(
                res.metrics.get("Infer_ms_per_1000_mean", np.nan)
            ),
            "Infer_ms_per_1000_std": float(
                res.metrics.get("Infer_ms_per_1000_std", np.nan)
            ),
            # Uncertainty summaries
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
            # per-output metrics (optional for users)
            "W_dot_R2": float(r2_each[0]) if len(r2_each) > 0 else float("nan"),
            "W_dot_MSE": float(mse_each[0])
            if len(mse_each) > 0
            else float("nan"),
            "W_dot_NLL_norm": float(nll_each_norm[0])
            if len(nll_each_norm) > 0
            else float("nan"),
            "m_dot_R2": float(r2_each[1]) if len(r2_each) > 1 else float("nan"),
            "m_dot_MSE": float(mse_each[1])
            if len(mse_each) > 1
            else float("nan"),
            "m_dot_NLL_norm": float(nll_each_norm[1])
            if len(nll_each_norm) > 1
            else float("nan"),
        }
        rows.append(row)

    per_run_df = pd.DataFrame(rows)

    # Append MEAN/STD across runs (numeric columns)
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

    # Per-output metrics sheet
    per_output_df = _build_per_output_metrics(per_output_acc)

    # README + SETTINGS
    readme_df = pd.DataFrame(
        {
            "column": list(_col_desc_map().keys()),
            "description": list(_col_desc_map().values()),
        }
    )
    settings_df = pd.DataFrame(
        {
            "param": [
                "n_runs",
                "base_seed",
                "layer_sizes",
                "activation",
                "prior_scale",
                "sigma_prior",
                "dropout_rate",
                "num_samples",
                "warmup_steps",
                "file",
                "sheet",
            ],
            "value": [
                n_runs,
                base_seed,
                layer_sizes if layer_sizes is None else list(layer_sizes),
                activation,
                prior_scale,
                sigma_prior,
                dropout_rate,
                num_samples,
                warmup_steps,
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
    print(f"\n[OK] Saved BNN Excel summary to: {abs_path}")
    return abs_path


# -------------------- Main process（保持兼容） --------------------
def main_process(
    file_path: str,
    sheet_name: str,
    *,
    repeat_runs: int = 0,
    excel_filename: Optional[str] = None,
) -> None:
    """
    If repeat_runs>0 and excel_filename provided: run repeated pipeline and export Excel.
    Else: run a single demonstration pass (training + plots).
    """
    if repeat_runs and excel_filename:
        repeat_runs_to_excel(
            file=file_path,
            sheet=sheet_name,
            excel_filename=excel_filename,
            n_runs=repeat_runs,
            base_seed=1234,
            layer_sizes=None,  # use default [in,20,10,out]
            activation="relu",
            prior_scale=1.0,
            sigma_prior="uniform",
            dropout_rate=0.0,
            num_samples=500,
            warmup_steps=500,
        )
        return

    # ---- Single run (kept for parity) ----
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_name = [r"$\dot W_{elec}$", r"$\dot m$"]

    data = load_data(file_path, sheet_name)
    x, y = split_data(data, input_cols, output_cols)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )
    x_train_norm, y_train_norm, x_scaler, y_scaler = normalize(x_train, y_train)
    x_valid_norm = x_scaler.transform(x_valid).astype(np.float64)

    x_train_t = torch.from_numpy(x_train_norm).float()
    y_train_t = torch.from_numpy(y_train_norm).float()
    x_valid_t = torch.from_numpy(x_valid_norm).float()

    print("Building and training Bayesian Neural Network...")

    bnn = BayesianNN(
        layer_sizes=[
            int(x_train_norm.shape[1]),
            20,
            10,
            int(y_train_norm.shape[1]),
        ],
        activation="tanh",
        prior_scale=1.0,
        sigma_prior="uniform",
        dropout_rate=0.0,
    )

    pyro.clear_param_store()
    nuts = NUTS(bnn)  # type: ignore[call-arg]
    mcmc = MCMC(nuts, num_samples=1000, warmup_steps=1000)
    mcmc.run(x_train_t, y_train_t)

    predictive = Predictive(
        bnn,
        posterior_samples=mcmc.get_samples(),
        return_sites=["_RETURN", "sigma"],
    )

    # Inference latency (ms / 1000 samples) — single-run printout
    mean_ms, std_ms = measure_inference_time_bnn(
        predictive, x_scaler, x_valid, repeats=20, warmup=3, n_samples=1000
    )
    print(f"Inference: {mean_ms:.1f} ± {std_ms:.1f} ms / 1000 samples")

    pred_sites = predictive(x_valid_t)
    mus = np.asarray(
        pred_sites["_RETURN"].detach().cpu().numpy(), dtype=np.float64
    )
    sigmas = np.asarray(
        pred_sites["sigma"].detach().cpu().numpy(), dtype=np.float64
    )

    mu_mean = mus.mean(0)
    epi_var = mus.var(0, ddof=0)
    ale_var = np.broadcast_to((sigmas**2).mean(0).reshape(1, -1), mu_mean.shape)
    total_var = ale_var + epi_var

    y_pred = denormalize(mu_mean, y_scaler)
    scales = np.asarray(
        getattr(y_scaler, "scale_", np.ones(y.shape[1])), dtype=np.float64
    ).reshape(1, -1)
    y_std = np.sqrt(total_var) * scales

    # Metrics (original units; RMSE printed for convenience)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    print(f"\nOriginal-scale RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAPE: {mape:.4f}")
    for i, col in enumerate(output_cols):
        rmse_i = np.sqrt(mean_squared_error(y_valid[:, i], y_pred[:, i]))
        r2_i = r2_score(y_valid[:, i], y_pred[:, i])
        mape_i = mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i])
        print(f"{col}: RMSE={rmse_i:.4f}, R^2={r2_i:.4f}, MAPE={mape_i:.4f}")

    plot_validation_group(
        y_valid.astype(np.float64), y_pred, y_std, output_name
    )


# -------------------- Script entry --------------------
if __name__ == "__main__":
    file_path = "./data/case_2_scroll.csv"
    sheet_name = "case_2_scroll"
    # file_path = "./data/case_1_rotary.csv"
    # sheet_name = "case_1_rotary"

    # 方式 A：单次运行（保持兼容）
    # main_process(file_path, sheet_name)

    # 方式 B：多次运行 + Excel 导出（默认 10 次）
    main_process(
        file_path,
        sheet_name,
        repeat_runs=10,
        excel_filename=f"outputs/BNN_repeat_{sheet_name}.xlsx",
    )

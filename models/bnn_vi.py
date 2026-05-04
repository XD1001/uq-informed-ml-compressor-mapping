"""
Bayesian Neural Network (BNN) with MCMC for VI Scroll Compressor Mapping
- Inputs:  P_suc, T_suc, T_dis, P_dis, T_inj, P_inj, frequency
- Outputs: m_inj, m_suc, W_total
- Pyro + NUTS sampling; epistemic/aleatoric decomposition
- Exports CSVs; provides a unified plotting/overlay interface identical to DE
- NEW: repeat runs (default 10) and export per-run + across-run MEAN/STD to Excel
- NEW (aligned to your first script): per_output_metrics sheet records
       R2_mean/std, MSE_mean/std, NLL_norm_mean/std across runs.
"""

from __future__ import annotations

import time
import os
import random
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

# ---------------------- Reproducibility ---------------------- #
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
pyro.set_rng_seed(1234)

# ---------------------- Canonical IO + Aliases (match DE) ---------------------- #
INPUT_ORDER: List[str] = [
    "P_suc",
    "T_suc",
    "T_dis",
    "P_dis",
    "T_inj",
    "P_inj",
    "frequency",
]
OUTPUT_ORDER: List[str] = ["m_inj", "m_suc", "W_total"]

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

# LaTeX-ready titles for plots (match DE)
OUTPUT_TITLES: List[str] = [
    r"$\dot m_{\mathrm{inj}}$",
    r"$\dot m_{\mathrm{suc}}$",
    r"$\dot W_{\mathrm{total}}$",
]


# ---------------------- Utilities ---------------------- #
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
    """Map desired canonical names to actual df columns by alias search."""
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


def _as_float1d(a: np.ndarray) -> NDArray[np.float_]:
    """Ensure numpy array is contiguous float64 1-D."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _ensure_vector_len(
    v: np.ndarray, n: int, name: str = "y_var"
) -> NDArray[np.float_]:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == n:
            return arr.astype(np.float64, copy=False)
        if arr.shape[0] == 1:
            return np.full(n, float(arr[0]), dtype=np.float64)
    raise ValueError(
        f"{name} must have length {n}, got shape {arr.shape}. "
        "If you passed a per-output vector (D,), slice the j-th element or expand to per-sample."
    )


def _prepare_1000(X_norm: np.ndarray, n: int = 1000) -> np.ndarray:
    if X_norm.shape[0] >= n:
        return X_norm[:n]
    reps = int(np.ceil(n / X_norm.shape[0]))
    return np.tile(X_norm, (reps, 1))[:n]


def time_predictive_bnn(
    model: "BayesianNN",
    mcmc: MCMC,
    X_raw: np.ndarray,
    x_scaler: StandardScaler,
    *,
    n_bench: int = 1000,
    repeats: int = 5,
) -> Tuple[float, float, float, int, int]:
    """
    Benchmark only the prediction wall time.
    Returns: (ms_per_1000, mean_ms, std_ms, n_used, posterior_S)
    """
    X_norm = x_scaler.transform(X_raw).astype(np.float64)
    Xb = _prepare_1000(X_norm, n=n_bench)

    post = mcmc.get_samples()
    some = next(iter(post.values()))
    posterior_S = int(some.shape[0])

    with torch.no_grad():
        _ = _predict_bnn(model, mcmc, Xb)

    times_ms: List[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _predict_bnn(model, mcmc, Xb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms, ddof=1)) if repeats > 1 else 0.0
    ms_per_1000 = mean_ms * (1000.0 / float(n_bench))
    return ms_per_1000, mean_ms, std_ms, int(Xb.shape[0]), posterior_S


# ---------------------- Plotting (identical behavior to DE) ---------------------- #
def _auto_grid(n_plots: int) -> Tuple[int, int]:
    cols = min(3, max(1, n_plots))
    rows = int(np.ceil(n_plots / cols))
    return rows, cols


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
            f"$R^2$={r2_v:.3f}, MSE={mse_v:.3f}, MAPE={mape_v:.3f}",
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
    return fig


# ---------------------- NLL helpers (normalized space) ---------------------- #
def gaussian_nll_vector(
    y_true: NDArray[np.float_],
    y_mean: NDArray[np.float_],
    y_var: NDArray[np.float_],
    *,
    add_const_2pi: bool = False,
) -> NDArray[np.float_]:
    yt = _as_float1d(y_true)
    ym = _as_float1d(y_mean)
    v = _ensure_vector_len(y_var, yt.shape[0], name="y_var")
    eps = 1e-12
    v = np.maximum(v, eps)
    term1 = 0.5 * np.log(v) + (
        0.5 * np.log(2 * np.pi) if add_const_2pi else 0.0
    )
    term2 = 0.5 * ((yt - ym) ** 2) / v
    return term1 + term2


# ---------------------- BNN Definition (MCMC) ---------------------- #
class BayesianNN(PyroModule):
    """Bayesian feed-forward network with Normal priors and homoscedastic noise."""

    def __init__(
        self,
        layer_sizes: Sequence[int],  # e.g., [7, 32, 16, 3]
        activation: str = "tanh",
        prior_scale: float = 5.0,
        noise_prior: str = "uniform",  # "uniform" or any other -> HalfCauchy
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(layer_sizes) >= 2, (
            "layer_sizes must include input and output"
        )

        self.activation = self._get_activation(activation)
        self.noise_prior = noise_prior
        self.dropout = nn.Dropout(dropout_rate)  # kept for API; p=0 for MCMC

        layers: List[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = int(layer_sizes[i]), int(layer_sizes[i + 1])
            lin = cast(Any, PyroModule[nn.Linear])(in_dim, out_dim)
            lin.weight = PyroSample(
                dist.Normal(0.0, prior_scale * np.sqrt(2.0 / max(1, in_dim)))
                .expand([out_dim, in_dim])  # type: ignore
                .to_event(2)
            )
            lin.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([out_dim]).to_event(1)  # type: ignore
            )
            layers.append(lin)
        self.layers = cast(Any, PyroModule[nn.ModuleList])(layers)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x
        for idx in range(len(self.layers) - 1):
            h = self.activation(self.layers[idx](h))
            if self.dropout.p > 0:
                h = self.dropout(h)
        mu = self.layers[-1](h)  # [N, D]
        pyro.deterministic("mu", mu)

        out_dim = mu.shape[-1]
        if self.noise_prior == "uniform":
            sigma = pyro.sample(
                "sigma", dist.Uniform(1e-5, 0.5).expand([out_dim]).to_event(1)
            )
        else:
            sigma = pyro.sample(
                "sigma", dist.HalfCauchy(1.0).expand([out_dim]).to_event(1)
            )

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs", dist.Normal(mu, sigma.clamp(min=1e-5)).to_event(1), obs=y
            )
        return mu

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        return {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
        }.get(name.lower(), nn.Tanh())


# ---------------------- BNN Training / Prediction ---------------------- #
def _fit_bnn_mcmc(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    *,
    layer_sizes: Sequence[int],
    activation: str = "tanh",
    prior_scale: float = 5.0,
    noise_prior: str = "uniform",
    num_samples: int = 1000,
    warmup_steps: int = 1000,
    num_chains: int = 1,
    dropout_rate: float = 0.0,
    target_accept_prob: float = 0.9,
) -> Tuple[BayesianNN, MCMC]:
    pyro.clear_param_store()
    Xtr_t = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    Ytr_t = torch.tensor(np.asarray(Ytr, dtype=np.float32))

    model = BayesianNN(
        layer_sizes=layer_sizes,
        activation=activation,
        prior_scale=prior_scale,
        noise_prior=noise_prior,
        dropout_rate=dropout_rate,
    )
    nuts = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        nuts,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )
    mcmc.run(Xtr_t, Ytr_t)
    return model, mcmc


def _collapse_mu_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < 3:
        raise ValueError(
            f"mu samples should have at least 3 dims [S,N,D], got {arr.shape}"
        )
    lead = int(np.prod(arr.shape[:-2]))
    return arr.reshape(lead, arr.shape[-2], arr.shape[-1])


def _collapse_sigma_samples(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(
            f"sigma samples should have at least 2 dims [S,D], got {arr.shape}"
        )
    lead = int(np.prod(arr.shape[:-1]))
    return arr.reshape(lead, arr.shape[-1])


def _predict_bnn(
    model: BayesianNN,
    mcmc: MCMC,
    Xte: np.ndarray,
) -> Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    """Return (y_pred_norm[N,D], epistemic_var_norm[N,D], aleatoric_var_norm[N,D])."""
    Xte_t = torch.tensor(np.asarray(Xte, dtype=np.float32))
    pred = Predictive(
        model,
        posterior_samples=mcmc.get_samples(),
        return_sites=["mu", "sigma"],
    )
    samples = pred(Xte_t)

    mu_raw = samples["mu"].detach().cpu().numpy().astype(np.float64)
    sig_raw = samples["sigma"].detach().cpu().numpy().astype(np.float64)

    mu_s = _collapse_mu_samples(mu_raw)  # (S*, N, D)
    sig_s = _collapse_sigma_samples(sig_raw)  # (S*, D)

    y_pred_norm = mu_s.mean(axis=0).astype(np.float64)  # [N, D]
    epistemic_var_norm = mu_s.var(axis=0, ddof=0).astype(np.float64)  # [N, D]

    # Aleatoric is homoscedastic per output; broadcast to [N, D]
    aleatoric_row = (
        (sig_s**2).mean(axis=0).astype(np.float64)[None, :]
    )  # [1, D]
    aleatoric_var_norm = np.repeat(
        aleatoric_row, Xte.shape[0], axis=0
    )  # [N, D]

    N, D = y_pred_norm.shape
    assert epistemic_var_norm.shape == (N, D)
    assert aleatoric_var_norm.shape == (N, D)

    return y_pred_norm, epistemic_var_norm, aleatoric_var_norm


# ---------------------- Unified overlay result (EXTENDED like GPR) ---------------------- #
class _ValidationResult:
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
        *,
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
        self.y_std_ale = y_std_ale
        self.y_std_epi = y_std_epi

    def check(self) -> None:
        assert self.y_true.shape == self.y_pred.shape, (
            f"{self.name}: y_true {self.y_true.shape} vs y_pred {self.y_pred.shape}"
        )
        if self.y_std is not None:
            assert self.y_std.shape == self.y_pred.shape, (
                f"{self.name}: y_std {self.y_std.shape} vs y_pred {self.y_pred.shape}"
            )
        if self.y_std_ale is not None:
            assert self.y_std_ale.shape == self.y_pred.shape, (
                f"{self.name}: y_std_ale {self.y_std_ale.shape} vs y_pred {self.y_pred.shape}"
            )
        if self.y_std_epi is not None:
            assert self.y_std_epi.shape == self.y_pred.shape, (
                f"{self.name}: y_std_epi {self.y_std_epi.shape} vs y_pred {self.y_pred.shape}"
            )
        if (self.y_lo is not None) ^ (self.y_hi is not None):
            raise ValueError(
                f"{self.name}: y_lo and y_hi must both be provided or both None."
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
            for _j, _name in enumerate(output_order):
                cols.append(f"{_name}_{tag}")
                mats.append(np.full((N, 1), np.nan))
        else:
            for _j, _name in enumerate(output_order):
                cols.append(f"{_name}_{tag}")
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


# ---------------------- Correct BNN inference-time measurement ---------------------- #
def _infer_time_ms_per_1000(
    model: "BayesianNN",
    mcmc: MCMC,
    x_scaler: StandardScaler,
    X_val_raw: np.ndarray,
    *,
    repeats: int = 30,
    warmup: int = 5,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    """
    Measure BNN *prediction* throughput in ms per 1000 samples.

    - Uses the trained posterior (mcmc) and Predictive to run forward passes.
    - Scales X with x_scaler, builds a batch of exactly `n_samples` (by repeat/tile).
    - Returns (mean_ms_per_1000, std_ms_per_1000) over `repeats` trials.
    """
    # Prepare exactly n_samples inputs in normalized space
    Xn = x_scaler.transform(X_val_raw).astype(np.float64)
    if Xn.shape[0] == 0:
        return float("nan"), float("nan")
    batch = _prepare_1000(Xn, n=n_samples)

    # Warmup to avoid first-call overhead
    for _ in range(max(0, warmup)):
        with torch.no_grad():
            _ = _predict_bnn(model, mcmc, batch)

    times: List[float] = []
    for _ in range(max(1, repeats)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _predict_bnn(model, mcmc, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ms_per_1000 = (dt * 1e3) * (1000.0 / float(n_samples))
        times.append(ms_per_1000)

    mean_ms_1000 = float(np.mean(times))
    std_ms_1000 = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    return mean_ms_1000, std_ms_1000


# ---------------------- High-level runner (single run; keeps CSV exports) ---------------------- #
def run_model_bnn(
    file: str,
    sheet: str,
    *,
    split_seed: int = 1234,
    layer_sizes: Sequence[int] = (7, 32, 16, 3),
    activation: str = "relu",
    prior_scale: float = 1.0,
    noise_prior: str = "half_cauchy",  # "uniform" or "half_cauchy"
    num_samples: int = 500,
    warmup_steps: int = 500,
    num_chains: int = 1,
    dropout_rate: float = 0.0,
    target_accept_prob: float = 0.9,
    model_dir: str = "./models/BNN",
    # ====== 与 GPR 对齐的可选参数：固定验证集 + 推理计时 ======
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: Optional[str] = None,
    infer_repeats: Optional[int] = None,
    infer_warmup: Optional[int] = None,
    infer_n_samples: Optional[int] = None,
) -> _ValidationResult:
    # Seeding
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    pyro.set_rng_seed(split_seed)

    # Data
    df = load_data(file, sheet)
    in_map = resolve_columns(df, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(df, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]

    X, Y = split_data(df, input_cols, output_cols)

    # 固定验证集（若提供）或常规 split
    if fixed_indices is not None:
        train_idx, val_idx = fixed_indices
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, shuffle=True
        )

    Xtr_n, Ytr_n, x_scaler, y_scaler = normalize_xy(X_train, Y_train)
    Xva_n = x_scaler.transform(X_val).astype(np.float64)
    Yva_n = y_scaler.transform(Y_val).astype(np.float64)

    # Train
    os.makedirs(model_dir, exist_ok=True)
    model, mcmc = _fit_bnn_mcmc(
        Xtr_n,
        Ytr_n,
        layer_sizes=layer_sizes,
        activation=activation,
        prior_scale=prior_scale,
        noise_prior=noise_prior,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        dropout_rate=dropout_rate,
        target_accept_prob=target_accept_prob,
    )

    # Save scalers + posterior snapshot
    joblib.dump(x_scaler, f"{model_dir}/BNN_x_scaler.pkl")
    joblib.dump(y_scaler, f"{model_dir}/BNN_y_scaler.pkl")
    torch.save(
        {k: v.cpu() for k, v in mcmc.get_samples().items()},
        f"{model_dir}/BNN_posterior_samples.pt",
    )

    # Predict + UQ (normalized)
    y_pred_n, var_epi_n, var_ale_n = _predict_bnn(model, mcmc, Xva_n)
    var_tot_n = (
        np.asarray(var_epi_n, dtype=np.float64)
        + np.asarray(var_ale_n, dtype=np.float64)
    ).astype(np.float64)

    # NLL (normalized space)
    nll_gauss_each: List[float] = []
    for j in range(Y.shape[1]):
        yt = _as_float1d(Yva_n[:, j])
        ym = _as_float1d(y_pred_n[:, j])
        yv = _as_float1d(var_tot_n[:, j])
        nll_vec = gaussian_nll_vector(yt, ym, yv, add_const_2pi=False)
        nll_gauss_each.append(float(np.mean(nll_vec)))

    # De-normalize to ORIGINAL units
    scales = cast(
        NDArray[np.float_], getattr(y_scaler, "scale_", np.ones(Y.shape[1]))
    )
    y_pred = y_scaler.inverse_transform(y_pred_n)
    y_std_epi = np.sqrt(np.maximum(var_epi_n, 1e-12)).astype(
        np.float64
    ) * scales.reshape(1, -1)
    y_std_ale = np.sqrt(np.maximum(var_ale_n, 1e-12)).astype(
        np.float64
    ) * scales.reshape(1, -1)
    y_std_tot = np.sqrt(np.maximum(var_tot_n, 1e-12)).astype(
        np.float64
    ) * scales.reshape(1, -1)

    # Figure
    fig = _make_validation_figure(
        Y_val.astype(np.float64), y_pred, y_std_tot, OUTPUT_TITLES
    )

    # Metrics (original units)
    n_out = Y.shape[1]
    mse_each = [
        mean_squared_error(Y_val[:, j], y_pred[:, j]) for j in range(n_out)
    ]
    mape_each = [
        mean_absolute_percentage_error(Y_val[:, j], y_pred[:, j])
        for j in range(n_out)
    ]
    r2_each = [r2_score(Y_val[:, j], y_pred[:, j]) for j in range(n_out)]

    metrics: Dict[str, Any] = {
        "MSE_each": mse_each,
        "MAPE_each": mape_each,
        "R2_each": r2_each,
        "MSE_mean": float(np.mean(mse_each)),
        "MAPE_mean": float(np.mean(mape_each)),
        "R2_mean": float(np.mean(r2_each)),
        "NLL_gaussian_norm_each": nll_gauss_each,
        "NLL_gaussian_norm_mean": float(np.mean(nll_gauss_each)),
        "layer_sizes": list(layer_sizes),
    }

    # === 推理计时（可选，与 GPR 对齐：返回 ms / 1000 samples 的均值/方差） ===
    if (
        (infer_repeats is not None)
        or (infer_warmup is not None)
        or (infer_n_samples is not None)
    ):
        r = infer_repeats if infer_repeats is not None else 30
        w = infer_warmup if infer_warmup is not None else 5
        n = infer_n_samples if infer_n_samples is not None else 1000
        mean_ms_1000, std_ms_1000 = _infer_time_ms_per_1000(
            model, mcmc, x_scaler, X_val, repeats=r, warmup=w, n_samples=n
        )
        metrics["Infer_ms_per_1000_mean"] = float(mean_ms_1000)
        metrics["Infer_ms_per_1000_std"] = float(std_ms_1000)

    settings = {
        "split_seed": split_seed,
        "activation": activation,
        "prior_scale": prior_scale,
        "noise_prior": noise_prior,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "num_chains": num_chains,
        "target_accept_prob": target_accept_prob,
    }

    # CSV exports (original units)
    os.makedirs("outputs", exist_ok=True)
    cols: List[str] = []
    mats: List[np.ndarray] = []
    for j, name in enumerate(OUTPUT_ORDER):
        cols.append(f"{name}_pred")
        mats.append(y_pred[:, [j]])
    for kind, arr in [
        ("std_total", y_std_tot),
        ("std_ale", y_std_ale),
        ("std_epi", y_std_epi),
    ]:
        for j, name in enumerate(OUTPUT_ORDER):
            cols.append(f"{name}_{kind}")
            mats.append(arr[:, [j]])
    out_mat = np.hstack(mats)
    pd.DataFrame(out_mat, columns=cols).to_csv(
        f"./outputs/BNN_uncertainty_decomposition_{sheet}.csv", index=False
    )
    pd.DataFrame(
        {"output": OUTPUT_ORDER, "nll_gaussian_norm": nll_gauss_each}
    ).to_csv(f"./outputs/BNN_nll_summary_{sheet}.csv", index=False)

    vr = _ValidationResult(
        name="BNN (Pyro MCMC)",
        y_true=Y_val.astype(np.float64),
        y_pred=y_pred,
        y_std=y_std_tot,
        y_lo=None,
        y_hi=None,
        output_names=OUTPUT_TITLES,
        fig=fig,
        metrics=metrics,
        settings=settings,
        y_std_ale=y_std_ale,
        y_std_epi=y_std_epi,
    )
    vr.check()

    # === 若提供 Excel 写入参数，则输出逐点 UQ（与 GPR 完全一致的列命名） ===
    if (uq_excel_path is not None) and (uq_sheet_name is not None):
        _write_uq_excel_sheet(
            uq_excel_path,
            uq_sheet_name,
            vr.y_true,
            vr.y_pred,
            vr.y_std,
            vr.y_std_ale,
            vr.y_std_epi,
            OUTPUT_ORDER,
        )
        print(
            f"[BNN] Wrote per-point UQ to '{uq_excel_path}' sheet='{uq_sheet_name}'"
        )

    return vr


# ---------------------- Repeated runs helpers ---------------------- #
def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    pyro.set_rng_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _single_run_predict(
    df: pd.DataFrame,
    *,
    split_seed: int,
    layer_sizes: Sequence[int],
    activation: str,
    prior_scale: float,
    noise_prior: str,
    num_samples: int,
    warmup_steps: int,
    num_chains: int,
    dropout_rate: float,
    target_accept_prob: float,
) -> Tuple[
    np.ndarray,  # Y_val (orig)
    np.ndarray,  # y_pred (orig)
    np.ndarray,  # y_std_tot (orig)
    np.ndarray,  # y_std_ale (orig)
    np.ndarray,  # y_std_epi (orig)
    Dict[str, List[float]],  # metrics_each
]:
    in_map = resolve_columns(df, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(df, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]

    X, Y = split_data(df, input_cols, output_cols)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=split_seed, shuffle=True
    )

    Xtr_n, Ytr_n, x_scaler, y_scaler = normalize_xy(X_train, Y_train)
    Xva_n = x_scaler.transform(X_val).astype(np.float64)
    Yva_n = y_scaler.transform(Y_val).astype(np.float64)

    model, mcmc = _fit_bnn_mcmc(
        Xtr_n,
        Ytr_n,
        layer_sizes=layer_sizes,
        activation=activation,
        prior_scale=prior_scale,
        noise_prior=noise_prior,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        dropout_rate=dropout_rate,
        target_accept_prob=target_accept_prob,
    )

    # Normalized predictions + variances
    y_pred_n, var_epi_n, var_ale_n = _predict_bnn(model, mcmc, Xva_n)
    var_tot_n = var_epi_n + var_ale_n

    # NLL in normalized space (per output)
    nll_each_norm: List[float] = []
    for j in range(Y.shape[1]):
        nll_vec = gaussian_nll_vector(
            Yva_n[:, j], y_pred_n[:, j], var_tot_n[:, j], add_const_2pi=False
        )
        nll_each_norm.append(float(np.mean(nll_vec)))

    # Map back to original units
    scales = cast(
        NDArray[np.float_], getattr(y_scaler, "scale_", np.ones(Y.shape[1]))
    )
    y_pred = y_scaler.inverse_transform(y_pred_n)
    y_std_epi = np.sqrt(np.maximum(var_epi_n, 1e-12)) * scales.reshape(1, -1)
    y_std_ale = np.sqrt(np.maximum(var_ale_n, 1e-12)) * scales.reshape(1, -1)
    y_std_tot = np.sqrt(np.maximum(var_tot_n, 1e-12)) * scales.reshape(1, -1)

    # Per-output metrics in original units
    r2_each = [
        float(r2_score(Y_val[:, j], y_pred[:, j])) for j in range(Y.shape[1])
    ]
    mse_each = [
        float(mean_squared_error(Y_val[:, j], y_pred[:, j]))
        for j in range(Y.shape[1])
    ]

    metrics_each = {
        "R2_each": r2_each,
        "MSE_each": mse_each,
        "NLL_each_norm": nll_each_norm,
    }
    return (
        Y_val.astype(np.float64),
        y_pred,
        y_std_tot,
        y_std_ale,
        y_std_epi,
        metrics_each,
    )


def _build_per_output_metrics(
    per_output_acc: Dict[str, List[float]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    label_keys = [("m_inj", "m_inj"), ("m_suc", "m_suc"), ("W_dot", "W_dot")]
    for display_name, key_prefix in label_keys:
        r2_vals = np.asarray(
            per_output_acc.get(f"{key_prefix}_R2", []), dtype=np.float64
        )
        mse_vals = np.asarray(
            per_output_acc.get(f"{key_prefix}_MSE", []), dtype=np.float64
        )
        nll_vals = np.asarray(
            per_output_acc.get(f"{key_prefix}_NLL_norm", []), dtype=np.float64
        )

        rows.append(
            {
                "output": display_name,
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


# ---------------------- Repeated runs -> Excel ---------------------- #
def repeat_runs_to_excel(
    file_path: str,
    sheet_name: str,
    *,
    excel_filename: str,
    n_runs: int = 10,
    base_seed: int = 1234,
    layer_sizes: Sequence[int] = (7, 32, 16, 3),
    activation: str = "relu",
    prior_scale: float = 1.0,
    noise_prior: str = "half_cauchy",
    num_samples: int = 500,
    warmup_steps: int = 500,
    num_chains: int = 1,
    dropout_rate: float = 0.0,
    target_accept_prob: float = 0.9,
) -> str:
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)
    df = load_data(file_path, sheet_name)

    excel_labels = ["m_suc", "m_inj", "W_dot"]
    internal_name_for_excel = {
        "m_suc": "m_suc",
        "m_inj": "m_inj",
        "W_dot": "W_total",
    }
    internal_index = {"m_inj": 0, "m_suc": 1, "W_total": 2}

    per_run_rows: List[Dict[str, Any]] = []
    per_output_acc: Dict[str, List[float]] = {
        "m_inj_R2": [],
        "m_inj_MSE": [],
        "m_inj_NLL_norm": [],
        "m_suc_R2": [],
        "m_suc_MSE": [],
        "m_suc_NLL_norm": [],
        "W_dot_R2": [],
        "W_dot_MSE": [],
        "W_dot_NLL_norm": [],
    }

    for r in range(1, n_runs + 1):
        seed = base_seed + 100 * r
        print(f"\n=== BNN repeated run {r}/{n_runs} (seed={seed}) ===")
        _seed_all(seed)

        (Y_val, y_pred, y_std_tot, y_std_ale, y_std_epi, metrics_each) = (
            _single_run_predict(
                df,
                split_seed=seed,
                layer_sizes=layer_sizes,
                activation=activation,
                prior_scale=prior_scale,
                noise_prior=noise_prior,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                num_chains=num_chains,
                dropout_rate=dropout_rate,
                target_accept_prob=target_accept_prob,
            )
        )

        r2_each = metrics_each["R2_each"]
        mse_each = metrics_each["MSE_each"]
        nll_each = metrics_each["NLL_each_norm"]

        per_output_acc["m_inj_R2"].append(float(r2_each[0]))
        per_output_acc["m_inj_MSE"].append(float(mse_each[0]))
        per_output_acc["m_inj_NLL_norm"].append(float(nll_each[0]))
        per_output_acc["m_suc_R2"].append(float(r2_each[1]))
        per_output_acc["m_suc_MSE"].append(float(mse_each[1]))
        per_output_acc["m_suc_NLL_norm"].append(float(nll_each[1]))
        per_output_acc["W_dot_R2"].append(float(r2_each[2]))
        per_output_acc["W_dot_MSE"].append(float(mse_each[2]))
        per_output_acc["W_dot_NLL_norm"].append(float(nll_each[2]))

        r2_mean = float(np.mean(r2_each))
        r2_std = float(np.std(r2_each, ddof=1)) if len(r2_each) > 1 else 0.0
        mse_mean = float(np.mean(mse_each))
        mse_std = float(np.std(mse_each, ddof=1)) if len(mse_each) > 1 else 0.0
        nll_mean = float(np.mean(nll_each))
        nll_std = float(np.std(nll_each, ddof=1)) if len(nll_each) > 1 else 0.0

        row: Dict[str, Any] = {
            "Run": r,
            "Seed": seed,
            "N_val": int(Y_val.shape[0]),
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "NLL_mean_norm": nll_mean,
            "NLL_std_norm": nll_std,
        }

        def _add_stats(label_excel: str) -> None:
            internal_key = internal_name_for_excel[label_excel]
            j = internal_index[internal_key]
            row[f"{label_excel}_pred_mean"] = float(np.mean(y_pred[:, j]))
            row[f"{label_excel}_pred_std_within_run"] = float(
                np.std(y_pred[:, j], ddof=1)
            )
            row[f"{label_excel}_std_total_mean"] = float(
                np.mean(y_std_tot[:, j])
            )
            row[f"{label_excel}_std_ale_mean"] = float(np.mean(y_std_ale[:, j]))
            row[f"{label_excel}_std_epi_mean"] = float(np.mean(y_std_epi[:, j]))

        for lbl in excel_labels:
            _add_stats(lbl)

        per_run_rows.append(row)

    per_run_df = pd.DataFrame(per_run_rows)

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

    per_output_df = _build_per_output_metrics(per_output_acc)

    summary_rows: List[Dict[str, Any]] = []
    core_df = per_run_df.iloc[:-2] if len(per_run_df) >= 2 else per_run_df
    for col in numeric_cols:
        summary_rows.append(
            {
                "metric": col,
                "MEAN_across_runs": float(core_df[col].mean()),
                "STD_across_runs": float(core_df[col].std(ddof=1)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(excel_filename) as writer:
        per_run_df.to_excel(writer, index=False, sheet_name="per-run")
        per_output_df.to_excel(
            writer, sheet_name="per_output_metrics", index=False
        )
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"\nExcel saved -> {excel_filename}")
    return excel_filename


# ---------------------- Script entry (standalone) ---------------------- #
def main_process(
    file_path: str,
    sheet_name: str,
    *,
    excel_filename: str,
    n_runs: int = 10,
    base_seed: int = 1234,
) -> None:
    repeat_runs_to_excel(
        file_path,
        sheet_name,
        excel_filename=excel_filename,
        n_runs=n_runs,
        base_seed=base_seed,
        layer_sizes=(7, 32, 16, 3),
        activation="relu",
        prior_scale=1.0,
        noise_prior="half_cauchy",
        num_samples=500,
        warmup_steps=500,
        num_chains=1,
        dropout_rate=0.0,
        target_accept_prob=0.9,
    )
    print("Done.")


if __name__ == "__main__":
    file_path = "./data/case_3_vi.csv"
    sheet_name = "case_3_vi"
    excel_file = f"./outputs/BNN_repeated_{sheet_name}.xlsx"

    # Option A: single run + CSV
    # vr = run_model_bnn(file_path, sheet_name)

    # Option B: repeated runs + Excel
    main_process(
        file_path,
        sheet_name,
        excel_filename=excel_file,
        n_runs=10,
        base_seed=1234,
    )

# ===== GPR for Vapor-Injection Scroll Compressor with repeat runs + Excel export ===== 
from __future__ import annotations

import os
import time
import random
from typing import Dict, List, Optional, Sequence, Tuple, Any, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Kernel,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

"""
Gaussian Process Regression (scikit-learn) for VI compressor mapping

Upgrades:
- Repeat runs (default 10) with per-run Excel summary (m_suc, m_inj, W_dot)
- Aleatory/Epistemic/Total UQ decomposition in ORIGINAL units
- Column alias resolver; dynamic plotting; HPO + K-fold CV; NLL metric
- (NEW) Overlay compatibility: fixed_indices + UQ Excel (per-point) writer
- (NEW) Optional inference time stats (ms/1000 samples)
"""

# ---------------------- Reproducibility ---------------------- #
random.seed(1234)
np.random.seed(1234)

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
# 内部顺序：m_inj, m_suc, W_total
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

# 可读标题（画图用）
OUTPUT_TITLES: List[str] = [
    r"$\dot m_{\mathrm{inj}}$",
    r"$\dot m_{\mathrm{suc}}$",
    r"$\dot W_{\mathrm{total}}$",
]

# Excel 对外展示的变量名与内部索引的映射（按用户要求的顺序与命名）
EXCEL_NAME_MAP = [
    ("m_suc", 1),  # index 1 in OUTPUT_ORDER
    ("m_inj", 0),  # index 0
    ("W_dot", 2),  # index 2 (内部叫 W_total)
]


# ---------------------- Data utils ---------------------- #
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
    """
    Map desired canonical names to actual df columns by alias search.
    Returns {desired_name: actual_df_col}.
    """
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
    data: pd.DataFrame, input_cols: List[str], output_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    X = data[input_cols].values
    Y = data[output_cols].values
    return X, Y


def split_train_valid(
    x: np.ndarray, y: np.ndarray, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=test_size, random_state=42, shuffle=True
    )
    return x_train, x_valid, y_train, y_valid


def normalize(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    xs, ys = StandardScaler(), StandardScaler()
    return xs.fit_transform(x), ys.fit_transform(y), xs, ys


def early_stopping(val_losses: List[float], patience: int = 10) -> bool:
    if len(val_losses) > patience:
        recent = val_losses[-patience:]
        if all(recent[i] >= recent[i - 1] for i in range(1, patience)):
            return True
    return False


# ---------------------- Plotting ---------------------- #
def _auto_grid(n_plots: int) -> Tuple[int, int]:
    cols = min(3, max(1, n_plots))
    rows = int(np.ceil(n_plots / cols))
    return rows, cols


def plot_validation_group(
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_name: List[str],
    save: bool = False,
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
        ax.errorbar(
            y_valid[:, i],
            y_pred[:, i],
            yerr=y_std[:, i],
            fmt="o",
            alpha=0.55,
            ecolor="lightgray",
        )
        lo: float = float(max(0.0, 0.95 * float(np.min(y_valid[:, i]))))
        hi: float = float(1.05 * float(np.max(y_valid[:, i])))
        ax.plot([lo, hi], [lo, hi], color="blue", lw=1)
        x = np.linspace(
            0.95 * np.min(y_valid[:, i]), 1.1 * np.max(y_valid[:, i]), 100
        )
        ax.fill_between(
            x, 0.9 * x, 1.1 * x, color="lightgray", alpha=0.35, label="$±10%$"
        )
        x_pos: float = float(0.85 * hi)
        y_pos: float = float(0.65 * hi)
        ax.text(
            float(x_pos),
            float(y_pos),
            f"$R^2$={r2_score(y_valid[:, i], y_pred[:, i]):.3f}\n"
            f"MSE={mean_squared_error(y_valid[:, i], y_pred[:, i]):.3f}\n"
            f"MAPE={mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i]):.3f}",
            fontsize=11,
            ha="right",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {y_name[i]}")
        ax.set_ylabel(f"Pred {y_name[i]}")
        ax.set_title(f"{y_name[i]} Prediction (Total UQ)")
        ax.grid(True, alpha=0.25)

    for j in range(len(y_name), len(axes_arr)):
        axes_arr[j].axis("off")

    fig.tight_layout()
    if save:
        _, parent_dir = get_current_and_parent_dir()
        os.makedirs(f"{parent_dir}/figure", exist_ok=True)
        plt.savefig(
            f"{parent_dir}/figure/sklearn_GP_multi_output.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    plt.show()


# ---------------------- NLL & noise extraction ---------------------- #
def gaussian_nll(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_var: np.ndarray,
    add_const_2pi: bool = False,
) -> float:
    eps = 1e-12
    term1 = (
        0.5 * (np.log(2 * np.pi) + np.log(y_var + eps))
        if add_const_2pi
        else 0.5 * np.log(y_var + eps)
    )
    term2 = 0.5 * (y_true - y_mean) ** 2 / (y_var + eps)
    return float(np.mean(term1 + term2))


def get_noise_level_from_kernel(kernel: Kernel) -> float:
    total = 0.0
    try:
        params = kernel.get_params(deep=True)
    except Exception:
        return 0.0
    seen = set()
    for v in params.values():
        if isinstance(v, WhiteKernel):
            oid = id(v)
            if oid in seen:
                continue
            seen.add(oid)
            noise = getattr(v, "noise_level", 0.0)
            try:
                total += float(noise)
            except Exception:
                total += float(np.asarray(noise))
    return total


# ---------------------- Hyperparameter tuning ---------------------- #
def build_kernel_init(d: int, c0: float, ls0: np.ndarray, n0: float) -> Kernel:
    return ConstantKernel(c0, (1e-3, 1e4)) * RBF(
        length_scale=ls0, length_scale_bounds=(1e-5, 1e5)
    ) + WhiteKernel(noise_level=n0, noise_level_bounds=(1e-6, 1.0))


def tune_gpr_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    tries: int = 15,
    n_restarts: int = 5,
    random_state: int = 1234,
) -> Tuple[GaussianProcessRegressor, float]:
    rng = np.random.default_rng(random_state)
    d = X_train.shape[1]
    best_model: Optional[GaussianProcessRegressor] = None
    best_nll = np.inf
    for _ in range(tries):
        c0 = float(10 ** rng.uniform(-2, 3))
        n0 = float(10 ** rng.uniform(-6, -0.3))
        ls0 = 10 ** rng.uniform(-2, 2, size=d)
        kernel = build_kernel_init(d, c0, ls0, n0)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=n_restarts,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        gpr.fit(X_train, y_train)
        mu_val, std_val = cast(
            Tuple[np.ndarray, np.ndarray], gpr.predict(X_val, return_std=True)
        )
        nll = gaussian_nll(y_val, mu_val, std_val**2, add_const_2pi=False)
        if nll < best_nll:
            best_nll = nll
            best_model = gpr
    assert best_model is not None
    return best_model, best_nll


# ---------------------- UQ decomposition ---------------------- #
def gp_predict_with_decomposition(
    models: List[GaussianProcessRegressor],
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
    """
    Returns (all in ORIGINAL units):
        Y_pred, Y_std_total, Y_std_ale, Y_std_epi, ale_var, epi_var, total_var
    """
    X_norm = x_scaler.transform(X_raw)
    n_samples, n_outputs = X_norm.shape[0], len(models)
    Y_pred_norm = np.zeros((n_samples, n_outputs))
    ale_var_norm = np.zeros((n_samples, n_outputs))
    epi_var_norm = np.zeros((n_samples, n_outputs))

    scales = getattr(y_scaler, "scale_", None)
    if scales is None:
        scales = np.ones((n_outputs,), dtype=float)
    y_scales = np.asarray(scales, dtype=float).reshape(1, n_outputs)

    for i, model in enumerate(models):
        mu_i, std_i = cast(
            Tuple[np.ndarray, np.ndarray],
            model.predict(X_norm, return_std=True),
        )
        var_i = std_i**2
        noise_lvl = get_noise_level_from_kernel(model.kernel_)  # normalized
        ale_i = np.full_like(var_i, float(noise_lvl))
        epi_i = np.clip(var_i - ale_i, a_min=0.0, a_max=None)
        Y_pred_norm[:, i] = mu_i
        ale_var_norm[:, i] = ale_i
        epi_var_norm[:, i] = epi_i

    Y_pred = y_scaler.inverse_transform(Y_pred_norm)
    ale_var = ale_var_norm * (y_scales**2)
    epi_var = epi_var_norm * (y_scales**2)
    total_var = ale_var + epi_var
    Y_std_total = np.sqrt(total_var)
    Y_std_ale = np.sqrt(ale_var)
    Y_std_epi = np.sqrt(epi_var)
    return (
        Y_pred,
        Y_std_total,
        Y_std_ale,
        Y_std_epi,
        ale_var,
        epi_var,
        total_var,
    )


# ---------------------- K-fold CV ---------------------- #
def k_fold_cross_validation(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
    patience: int = 10,
    *,
    hpo_tries: int = 15,
    hpo_restarts: int = 5,
    model_folder: str = "models/gpr",
) -> None:
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    fold = 1
    history_mse: List[float] = []
    history_mape: List[float] = []
    history_r2: List[float] = []
    history_nll: List[float] = []

    os.makedirs(model_folder, exist_ok=True)
    X_norm, Y_norm, x_scaler, y_scaler = normalize(X, Y)

    for train_index, val_index in kf.split(X):
        print(f"Fold {fold}/{k}")
        X_train_norm, X_val_norm = X_norm[train_index], X_norm[val_index]
        Y_train_norm, Y_val_norm = Y_norm[train_index], Y_norm[val_index]

        models: List[GaussianProcessRegressor] = []
        val_nll_each: List[float] = []
        for i in range(Y.shape[1]):
            best_model, best_nll = tune_gpr_hyperparams(
                X_train_norm,
                Y_train_norm[:, i],
                X_val_norm,
                Y_val_norm[:, i],
                tries=hpo_tries,
                n_restarts=hpo_restarts,
                random_state=1234 + i + 100 * fold,
            )
            models.append(best_model)
            val_nll_each.append(best_nll)

        # metrics in normalized space
        Y_pred_norm = np.zeros_like(Y_val_norm)
        for i, model in enumerate(models):
            mu_val, _ = cast(
                Tuple[np.ndarray, np.ndarray],
                model.predict(X_val_norm, return_std=True),
            )
            Y_pred_norm[:, i] = mu_val

        val_mse = mean_squared_error(Y_val_norm, Y_pred_norm)
        val_mape = mean_absolute_percentage_error(Y_val_norm, Y_pred_norm)
        val_r2 = r2_score(Y_val_norm, Y_pred_norm)
        val_nll_mean = float(np.mean(val_nll_each))

        history_mse.append(val_mse)
        history_mape.append(val_mape)
        history_r2.append(val_r2)
        history_nll.append(val_nll_mean)

        print(f"MSE:  {val_mse:.4f}")
        print(f"MAPE: {val_mape:.4f}")
        print(f"R2:   {val_r2:.4f}")
        print(f"NLL:  {val_nll_mean:.4f}  (normalized space, total variance)")

        # Save models/scalers for this fold
        for idx, model in enumerate(models):
            joblib.dump(
                model, f"{model_folder}/gpr_model_fold{fold}_output{idx}.pkl"
            )
        joblib.dump(x_scaler, f"{model_folder}/x_scaler_fold{fold}.pkl")
        joblib.dump(y_scaler, f"{model_folder}/y_scaler_fold{fold}.pkl")

        if early_stopping(history_mse, patience=patience):
            print("Early stopping triggered.")
            break
        fold += 1

    _plot_loss_history(history_mse, "MSE")
    _plot_loss_history(history_mape, "MAPE")
    _plot_loss_history(history_r2, "R2")
    _plot_loss_history(history_nll, "NLL")


def _plot_loss_history(history: List[float], loss_name: str) -> None:
    if not history:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Fold")
    plt.ylabel(loss_name)
    plt.title(f"{loss_name} History")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------- Validation container ---------------------- #
class _ValidationResult:
    """
    Container aligned with other scripts; includes optional ale/epi std.
    """

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
            "y_true vs y_pred shape mismatch"
        )
        if self.y_std is not None:
            assert self.y_std.shape == self.y_pred.shape, (
                "y_std vs y_pred shape mismatch"
            )
        if self.y_std_ale is not None:
            assert self.y_std_ale.shape == self.y_pred.shape, (
                "y_std_ale vs y_pred shape mismatch"
            )
        if self.y_std_epi is not None:
            assert self.y_std_epi.shape == self.y_pred.shape, (
                "y_std_epi vs y_pred shape mismatch"
            )
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
        x_line = np.linspace(
            0.95 * np.min(y_valid[:, i]), 1.1 * np.max(y_valid[:, i]), 200
        )
        ax.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.35, label="$±10%$"
        )
        lo: float = float(max(0.0, 0.95 * float(np.min(y_valid[:, i]))))
        hi: float = float(1.05 * float(np.max(y_valid[:, i])))
        ax.plot([lo, hi], [lo, hi], lw=1)
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
        x_pos: float = float(0.85 * hi)
        y_pos: float = float(0.65 * hi)
        ax.text(
            float(x_pos),
            float(y_pos),
            f"$R^2$={r2:.3f}\nMSE={mse:.3f}\nMAPE={mape:.3f}",
            fontsize=11,
            ha="right",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {y_name[i]}")
        ax.set_ylabel(f"Pred {y_name[i]}")
        ax.set_title(f"{y_name[i]} Prediction (Total UQ)")
        ax.grid(True, alpha=0.25)

    for j in range(len(y_name), len(axes_arr)):
        axes_arr[j].axis("off")
    fig.tight_layout()
    return fig


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
        with pd.ExcelWriter(excel_path, mode="a", if_sheet_exists="replace", engine="openpyxl") as wr:
            df.to_excel(wr, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode="w", engine="openpyxl") as wr:
            df.to_excel(wr, sheet_name=sheet_name, index=False)


# ---------------------- Optional inference-time measurement ---------------------- #
def _infer_time_ms_per_1000(
    models: List[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    X_val_raw: np.ndarray,
    *,
    repeats: int = 30,
    warmup: int = 5,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    """
    简单测量 predict 的吞吐：ms / 1000 samples（多输出逐个 predict）。
    """
    Xn = x_scaler.transform(X_val_raw)
    if Xn.shape[0] == 0:
        return float("nan"), float("nan")

    # 组一个长度为 n_samples 的 batch（循环/重复取样）
    idx = np.arange(Xn.shape[0])
    batch = Xn[np.mod(np.arange(n_samples), len(idx))]

    times: List[float] = []

    # warmup
    for _ in range(max(0, warmup)):
        for m in models:
            _ = m.predict(batch, return_std=True)

    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for m in models:
            _ = m.predict(batch, return_std=True)
        dt = time.perf_counter() - t0
        # 转换为 ms/1000 样本
        ms_per_1000 = (dt * 1e3) * (1000.0 / float(n_samples))
        times.append(ms_per_1000)

    return float(np.mean(times)), (float(np.std(times, ddof=1)) if len(times) > 1 else 0.0)


# ---------------------- Main flows ---------------------- #
def run_model_gpr(
    file: str,
    sheet: str,
    *,
    split_seed: int = 1234,
    k_folds: int = 5,
    patience: int = 5,
    hpo_tries: int = 15,
    hpo_restarts: int = 5,
    model_folder: str = "models/gpr",
    # === overlay 对接的新增可选参数 ===
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: Optional[str] = None,
    infer_repeats: Optional[int] = None,
    infer_warmup: Optional[int] = None,
    infer_n_samples: Optional[int] = None,
) -> _ValidationResult:
    """
    1) train with K-fold + HPO (on train split)
    2) load fold-1 models & scalers
    3) predict + UQ decomposition on held-out val set (ORIGINAL units)
    4) (optional) write per-point UQ to Excel (uq_excel_path + uq_sheet_name)
    5) (optional) measure inference time (ms/1000 samples)
    """
    data = load_data(file, sheet)
    in_map = resolve_columns(data, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(data, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]

    X, Y = split_data(data, input_cols, output_cols)

    # —— 固定验证集或按 split_seed 切分 —— #
    if fixed_indices is not None:
        train_idx, val_idx = fixed_indices
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, shuffle=True
        )

    # 仅在 train 部分做 K-fold + HPO
    k_fold_cross_validation(
        x_train,
        y_train,
        k=k_folds,
        patience=patience,
        hpo_tries=hpo_tries,
        hpo_restarts=hpo_restarts,
        model_folder=model_folder,
    )

    models: List[GaussianProcessRegressor] = []
    for i in range(len(OUTPUT_ORDER)):
        models.append(
            joblib.load(f"{model_folder}/gpr_model_fold1_output{i}.pkl")
        )
    x_scaler: StandardScaler = joblib.load(f"{model_folder}/x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_folder}/y_scaler_fold1.pkl")

    # —— 这里传入 x_val —— #
    Y_pred, Y_std_total, Y_std_ale, Y_std_epi, *_ = (
        gp_predict_with_decomposition(models, x_scaler, y_scaler, x_val)
    )

    fig = _make_validation_figure(y_val, Y_pred, Y_std_total, OUTPUT_TITLES)

    # Metrics (ORIGINAL units, per output)
    mse_each = [
        mean_squared_error(y_val[:, j], Y_pred[:, j])
        for j in range(Y_pred.shape[1])
    ]
    mape_each = [
        mean_absolute_percentage_error(y_val[:, j], Y_pred[:, j])
        for j in range(Y_pred.shape[1])
    ]
    r2_each = [
        r2_score(y_val[:, j], Y_pred[:, j]) for j in range(Y_pred.shape[1])
    ]

    # NLL (normalized space)
    X_val_norm = x_scaler.transform(x_val)
    Y_val_norm = y_scaler.transform(y_val)
    nll_each = []
    for i, model in enumerate(models):
        mu_val, std_val = cast(
            Tuple[np.ndarray, np.ndarray],
            model.predict(X_val_norm, return_std=True),
        )
        nll_each.append(
            gaussian_nll(
                Y_val_norm[:, i], mu_val, std_val**2, add_const_2pi=False
            )
        )

    # Inference-time (optional)
    infer_mean_ms_1000: Optional[float] = None
    infer_std_ms_1000: Optional[float] = None
    if (infer_repeats is not None) or (infer_warmup is not None) or (infer_n_samples is not None):
        r = infer_repeats if infer_repeats is not None else 30
        w = infer_warmup if infer_warmup is not None else 5
        n = infer_n_samples if infer_n_samples is not None else 1000
        m, s = _infer_time_ms_per_1000(models, x_scaler, x_val, repeats=r, warmup=w, n_samples=n)
        infer_mean_ms_1000, infer_std_ms_1000 = m, s

    # Add mean/std across outputs (ddof=1 for std)
    mse_mean = float(np.mean(mse_each))
    mse_std = float(np.std(mse_each, ddof=1))
    r2_mean = float(np.mean(r2_each))
    r2_std = float(np.std(r2_each, ddof=1))
    nll_mean_norm = float(np.mean(nll_each))
    nll_std_norm = float(np.std(nll_each, ddof=1))

    metrics: Dict[str, Any] = {
        "MSE_each": mse_each,
        "MAPE_each": mape_each,
        "R2_each": r2_each,
        "MSE_mean": mse_mean,
        "MSE_std": mse_std,
        "MAPE_mean": float(np.mean(mape_each)),
        "R2_mean": r2_mean,
        "R2_std": r2_std,
        "NLL_each_norm": nll_each,
        "NLL_mean_norm": nll_mean_norm,
        "NLL_std_norm": nll_std_norm,
    }
    if infer_mean_ms_1000 is not None:
        metrics["Infer_ms_per_1000_mean"] = float(infer_mean_ms_1000)
        metrics["Infer_ms_per_1000_std"] = float(infer_std_ms_1000)   # type: ignore

    settings = {
        "k_folds": k_folds,
        "patience": patience,
        "hpo_tries": hpo_tries,
        "hpo_restarts": hpo_restarts,
        "split_seed": split_seed,
        "model_folder": model_folder,
    }

    vr = _ValidationResult(
        name="GPR (sklearn, VI)",
        y_true=y_val,
        y_pred=Y_pred,
        y_std=Y_std_total,
        y_std_ale=Y_std_ale,
        y_std_epi=Y_std_epi,
        y_lo=None,
        y_hi=None,
        output_names=OUTPUT_TITLES,
        fig=fig,
        metrics=metrics,
        settings=settings,
    )
    vr.check()

    # —— 若 overlay 传入 Excel 写入参数，则在此写入逐点 UQ —— #
    if (uq_excel_path is not None) and (uq_sheet_name is not None):
        _write_uq_excel_sheet(
            uq_excel_path,
            uq_sheet_name,
            vr.y_true,
            vr.y_pred,
            vr.y_std,
            vr.y_std_ale,
            vr.y_std_epi,
            OUTPUT_ORDER,  # 列名按内部顺序（m_inj, m_suc, W_total）
        )
        print(f"[GPR] Wrote per-point UQ to '{uq_excel_path}' sheet='{uq_sheet_name}'")

    return vr


# ---------------------- Repeat runs + Excel export ---------------------- #
def _col_desc_map() -> Dict[str, str]:
    # 为 Excel README 提供释义
    desc = {
        "Run": "Run index starting from 1",
        "Seed": "Random seed used for train/val split",
        # Per-run metric summaries (across outputs)
        "R2_mean": "Mean R^2 over outputs in ORIGINAL units",
        "R2_std": "Std of R^2 over outputs (ddof=1)",
        "MSE_mean": "Mean MSE over outputs in ORIGINAL units",
        "MSE_std": "Std of MSE over outputs (ddof=1)",
        "NLL_mean_norm": "Mean Gaussian NLL over outputs in NORMALIZED space",
        "NLL_std_norm": "Std of Gaussian NLL over outputs in NORMALIZED space (ddof=1)",
    }
    # 逐输出的不确定性统计（样本维度 mean/std）+ 逐输出性能指标
    for pretty_name, _idx in EXCEL_NAME_MAP:
        prefix = pretty_name  # m_suc / m_inj / W_dot
        desc.update(
            {
                f"{prefix}_pred_mean": f"Mean of predicted {prefix} over validation samples (ORIGINAL units)",
                f"{prefix}_pred_std": f"Std of predicted {prefix} over validation samples (ORIGINAL units)",
                f"{prefix}_total_std_mean": f"Mean of TOTAL predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_total_std_std": f"Std of TOTAL predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_ale_std_mean": f"Mean of ALEATORY predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_ale_std_std": f"Std of ALEATORY predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_epi_std_mean": f"Mean of EPISTEMIC predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_epi_std_std": f"Std of EPISTEMIC predictive std of {prefix} (ORIGINAL units)",
                f"{prefix}_R2": f"Per-output R^2 for {prefix} on validation set (ORIGINAL units)",
                f"{prefix}_MSE": f"Per-output MSE for {prefix} on validation set (ORIGINAL units)",
                f"{prefix}_NLL_norm": f"Per-output Gaussian NLL for {prefix} on validation set (NORMALIZED space)",
            }
        )
    return desc


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
) -> str:
    """
    重复运行 n 次，每次训练+预测+UQ 分解；把验证集上样本维度的统计（mean±std）写入 Excel。
    另外记录每次运行的 R^2 / MSE / NLL（跨输出的 mean 与 std），以及各输出变量分别的 R^2 / MSE / NLL_norm。
    额外新增：在 per_output_metrics 工作表中，统计各输出变量 R^2 / MSE / NLL_norm 的 mean 与 std（跨运行）。
    """
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for run_idx in range(1, n_runs + 1):
        seed = base_seed + (run_idx - 1)
        print(f"\n[GPR-VI repeat] Run {run_idx}/{n_runs}, seed={seed}")
        res = run_model_gpr(file=file, sheet=sheet, split_seed=seed)
        res.check()
        assert (
            res.y_std is not None
            and res.y_std_ale is not None
            and res.y_std_epi is not None
        )

        # 逐输出统计（在验证样本维度上）
        pred_mean, pred_std = _mean_std_over_samples(res.y_pred)
        total_mean, total_std = _mean_std_over_samples(res.y_std)
        ale_mean, ale_std = _mean_std_over_samples(res.y_std_ale)
        epi_mean, epi_std = _mean_std_over_samples(res.y_std_epi)

        # 该次运行的指标
        metrics: Dict[str, Any] = res.metrics or {}
        r2_each = list(metrics.get("R2_each", []))
        mse_each = list(metrics.get("MSE_each", []))
        nll_each_norm = list(metrics.get("NLL_each_norm", []))

        # 汇总（跨输出）
        r2_mean = float(
            metrics.get("R2_mean", np.mean(r2_each) if len(r2_each) else np.nan)
        )
        r2_std = float(
            metrics.get(
                "R2_std",
                np.std(r2_each, ddof=1) if len(r2_each) > 1 else np.nan,
            )
        )
        mse_mean = float(
            metrics.get(
                "MSE_mean", np.mean(mse_each) if len(mse_each) else np.nan
            )
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
                np.mean(nll_each_norm) if len(nll_each_norm) else np.nan,
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

        row: Dict[str, Any] = {
            "Run": run_idx,
            "Seed": seed,
            # Per-run metric summaries (across outputs)
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "NLL_mean_norm": nll_mean_norm,
            "NLL_std_norm": nll_std_norm,
        }

        # Excel 列以 m_suc, m_inj, W_dot 顺序输出（每输出在验证样本维度的 mean/std）
        for pretty_name, j in EXCEL_NAME_MAP:
            # 不确定性相关（样本维度 mean/std）
            row[f"{pretty_name}_pred_mean"] = float(pred_mean[j])
            row[f"{pretty_name}_pred_std"] = float(pred_std[j])
            row[f"{pretty_name}_total_std_mean"] = float(total_mean[j])
            row[f"{pretty_name}_total_std_std"] = float(total_std[j])
            row[f"{pretty_name}_ale_std_mean"] = float(ale_mean[j])
            row[f"{pretty_name}_ale_std_std"] = float(ale_std[j])
            row[f"{pretty_name}_epi_std_mean"] = float(epi_mean[j])
            row[f"{pretty_name}_epi_std_std"] = float(epi_std[j])

            # 逐输出的 R2/MSE/NLL_norm（单次 run 的单值）
            r2_val = float(r2_each[j]) if len(r2_each) > j else float("nan")
            mse_val = float(mse_each[j]) if len(mse_each) > j else float("nan")
            nll_val = (
                float(nll_each_norm[j])
                if len(nll_each_norm) > j
                else float("nan")
            )
            row[f"{pretty_name}_R2"] = r2_val
            row[f"{pretty_name}_MSE"] = mse_val
            row[f"{pretty_name}_NLL_norm"] = nll_val

        rows.append(row)

    per_run_df = pd.DataFrame(rows)

    # 跨运行的 MEAN/STD（仅数值列）
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

    # === 新增：各输出变量的 R2/MSE/NLL_norm 跨运行 mean/std 的整洁汇总 ===
    per_output_rows: List[Dict[str, Any]] = []
    for pretty_name, _j in EXCEL_NAME_MAP:
        r2_col = f"{pretty_name}_R2"
        mse_col = f"{pretty_name}_MSE"
        nll_col = f"{pretty_name}_NLL_norm"

        # 去除 NaN 后计算
        r2_series = pd.to_numeric(
            per_run_df.loc[
                per_run_df["Run"].apply(
                    lambda v: isinstance(v, (int, np.integer))
                ),
                r2_col,
            ],
            errors="coerce",
        ).dropna()
        mse_series = pd.to_numeric(
            per_run_df.loc[
                per_run_df["Run"].apply(
                    lambda v: isinstance(v, (int, np.integer))
                ),
                mse_col,
            ],
            errors="coerce",
        ).dropna()
        nll_series = pd.to_numeric(
            per_run_df.loc[
                per_run_df["Run"].apply(
                    lambda v: isinstance(v, (int, np.integer))
                ),
                nll_col,
            ],
            errors="coerce",
        ).dropna()

        per_output_rows.append(
            {
                "output": pretty_name,  # m_suc / m_inj / W_dot
                "R2_mean": float(r2_series.mean())
                if len(r2_series)
                else float("nan"),
                "R2_std": float(r2_series.std(ddof=1))
                if len(r2_series) > 1
                else float("nan"),
                "MSE_mean": float(mse_series.mean())
                if len(mse_series)
                else float("nan"),
                "MSE_std": float(mse_series.std(ddof=1))
                if len(mse_series) > 1
                else float("nan"),
                "NLL_norm_mean": float(nll_series.mean())
                if len(nll_series)
                else float("nan"),
                "NLL_norm_std": float(nll_series.std(ddof=1))
                if len(nll_series) > 1
                else float("nan"),
            }
        )
    per_output_df = pd.DataFrame(per_output_rows)

    # README + SETTINGS
    readme_kv = _col_desc_map()
    readme_df = pd.DataFrame(
        {
            "column": list(readme_kv.keys()),
            "description": list(readme_kv.values()),
        }
    )
    settings_df = pd.DataFrame(
        {
            "param": ["n_runs", "base_seed", "file", "sheet"],
            "value": [n_runs, base_seed, file, sheet],
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
    print(f"\n[OK] Saved GPR-VI Excel summary to: {abs_path}")
    return abs_path


def main_process(
    file_path: str,
    sheet_name: str,
    *,
    repeat_runs: int = 10,
    excel_filename: Optional[str] = None,
    model_folder: str = "models/gpr",
) -> None:
    """
    - 当提供 excel_filename 时：执行 repeat_runs_to_excel（默认 10 次）
    - 否则：执行单次流程并导出详细 CSV（与原脚本一致）
    """
    if excel_filename:
        repeat_runs_to_excel(
            file=file_path,
            sheet=sheet_name,
            excel_filename=excel_filename,
            n_runs=repeat_runs,
        )
        return

    # ---- 单次流程（保持原有导出 CSV 的功能） ----
    data = load_data(file_path, sheet_name)
    in_map = resolve_columns(data, INPUT_ALIASES, strict=True)
    out_map = resolve_columns(data, OUTPUT_ALIASES, strict=True)
    input_cols = [in_map[k] for k in INPUT_ORDER]
    output_cols = [out_map[k] for k in OUTPUT_ORDER]
    print("[columns] inputs  ->", dict(zip(INPUT_ORDER, input_cols)))
    print("[columns] outputs ->", dict(zip(OUTPUT_ORDER, output_cols)))

    X, Y = split_data(data, input_cols, output_cols)
    x_train, x_val, y_train, y_val = split_train_valid(X, Y)

    k_fold_cross_validation(
        x_train, y_train, k=5, patience=5, hpo_tries=15, hpo_restarts=5
    )

    models: List[GaussianProcessRegressor] = []
    for i in range(len(OUTPUT_ORDER)):
        models.append(
            joblib.load(f"{model_folder}/gpr_model_fold1_output{i}.pkl")
        )
    x_scaler: StandardScaler = joblib.load(f"{model_folder}/x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_folder}/y_scaler_fold1.pkl")

    Y_pred, Y_std_total, Y_std_ale, Y_std_epi, ale_var, epi_var, total_var = (
        gp_predict_with_decomposition(models, x_scaler, y_scaler, x_val)
    )

    plot_validation_group(y_val, Y_pred, Y_std_total, OUTPUT_TITLES)

    X_val_norm = x_scaler.transform(x_val)
    Y_val_norm = y_scaler.transform(y_val)
    nll_per_output: List[float] = []
    for i, model in enumerate(models):
        mu_val, std_val = cast(
            Tuple[np.ndarray, np.ndarray],
            model.predict(X_val_norm, return_std=True),
        )
        nll_per_output.append(
            gaussian_nll(
                Y_val_norm[:, i], mu_val, std_val**2, add_const_2pi=False
            )
        )

    print("\n=== Validation NLL (normalized space; total variance) ===")
    for j, name in enumerate(OUTPUT_TITLES):
        print(f"{name}: NLL = {nll_per_output[j]:.6f}")
    print(f"Mean NLL: {np.mean(nll_per_output):.6f}")

    # 详细分解导出（ORIGINAL units）
    os.makedirs("outputs", exist_ok=True)
    cols, mats = [], []
    for j, name in enumerate(OUTPUT_ORDER):
        cols.append(f"{name}_pred")
        mats.append(Y_pred[:, [j]])
    for kind, arr in [
        ("std_total", Y_std_total),
        ("std_ale", Y_std_ale),
        ("std_epi", Y_std_epi),
    ]:
        for j, name in enumerate(OUTPUT_ORDER):
            cols.append(f"{name}_{kind}")
            mats.append(arr[:, [j]])
    for kind, arr in [
        ("var_ale", ale_var),
        ("var_epi", epi_var),
        ("var_total", total_var),
    ]:
        for j, name in enumerate(OUTPUT_ORDER):
            cols.append(f"{name}_{kind}")
            mats.append(arr[:, [j]])
    out_df = pd.DataFrame(np.hstack(mats), columns=cols)
    out_path = f"outputs/gpr_UQ_decomp_{sheet_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed decomposition to {out_path}")


# ---------------------- Script entry ---------------------- #
if __name__ == "__main__":
    # 示例：请按需修改文件/工作表名与导出文件名
    file_path = "./data/case_3_vi.csv"
    sheet_name = "case_3_vi"

    # 方式 A：多次运行 + Excel 导出（推荐；默认重复 10 次）
    main_process(
        file_path,
        sheet_name,
        repeat_runs=10,
        excel_filename=f"./outputs/GPR_repeat_{sheet_name}.xlsx",
    )

    # 方式 B：只跑一次并导出详细 CSV
    # main_process(file_path, sheet_name, excel_filename=None)

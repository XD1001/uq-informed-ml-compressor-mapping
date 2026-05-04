"""
Import necessary libraries
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple, Dict, Any, cast

import time
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
Build Gaussian Process Regression model using scikit-learn

Upgrades:
1) StandardScaler
2) K-fold cross-validation
3) Aleatory & Epistemic uncertainty decomposition
4) Hyperparameter optimization (multi-start random search)
5) Negative Log-Likelihood (NLL) metric for UQ quality
6) Repeat runs (default 10) and export per-run stats to Excel
7) Inference timing (ms/1000 samples): mean + std
8) (NEW) Save per-point UQ decomposition to a user-specified Excel (sheet append)
"""

# fix random seeds for reproducibility (base seed; runs will offset this)
random.seed(1234)
np.random.seed(1234)


# --------------------- Utilities & plotting ---------------------
def get_current_and_parent_dir() -> Tuple[str, str]:
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    return current_dir, parent_dir


def load_data(file: str, sheet: str) -> pd.DataFrame:
    if file.lower().endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file, sheet_name=sheet)
    if data.empty:
        raise ValueError("Loaded data is empty")
    return data.dropna()


def split_data(
    data: pd.DataFrame, input_cols: List[str], output_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    x = data[input_cols].values
    y = data[output_cols].values
    return x, y


def split_train_valid(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=test_size, random_state=42, shuffle=True
    )
    return x_train, x_valid, y_train, y_valid


def normalize(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    return (
        x_scaler.fit_transform(x),
        y_scaler.fit_transform(y),
        x_scaler,
        y_scaler,
    )


def early_stopping(val_losses: List[float], patience: int = 10) -> bool:
    if len(val_losses) > patience:
        recent_losses = val_losses[-patience:]
        if all(
            recent_losses[i] >= recent_losses[i - 1] for i in range(1, patience)
        ):
            return True
    return False


def plot_loss_history(history: List[float], loss_name: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(history) + 1), history, marker="o", label="Validation Loss"
    )
    plt.xticks(ticks=range(1, len(history) + 1))
    plt.xlabel("Epoch/Fold")
    plt.ylabel(loss_name)
    plt.title(f"{loss_name} History")
    plt.grid()
    plt.show()


def plot_validation_group(
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_name: List[str],
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
        plt.plot(
            [0.95 * y_valid[:, i].min(), 1.05 * y_valid[:, i].max()],
            [0.95 * y_valid[:, i].min(), 1.05 * y_valid[:, i].max()],
            color="blue",
            lw=1,
        )
        x = np.linspace(
            0.95 * y_valid[:, i].min(), 1.1 * y_valid[:, i].max(), 100
        )
        plt.fill_between(
            x, 0.9 * x, 1.1 * x, color="lightgray", alpha=0.4, label="$±10%$"
        )
        plt.text(
            0.65 * y_valid[:, i].max(),
            0.9 * y_valid[:, i].max(),
            r"$\pm$ 10 %",
            fontsize=12,
        )
        plt.text(
            0.85 * y_valid[:, i].max(),
            0.65 * y_valid[:, i].max(),
            f"$R^2$={r2_score(y_valid[:, i], y_pred[:, i]):.3f} \n"
            + f"MSE={mean_squared_error(y_valid[:, i], y_pred[:, i]):.3f} \n"
            + f"MAPE={mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i]):.3f} \n",
            fontsize=12,
        )
        plt.xlim(max(0, 0.95 * y_valid[:, i].min()), 1.05 * y_valid[:, i].max())
        plt.ylim(max(0, 0.95 * y_valid[:, i].min()), 1.05 * y_valid[:, i].max())
        plt.xlabel(f"True {y_name[i]}")
        plt.ylabel(f"Pred {y_name[i]}")
        plt.title(f"{y_name[i]} Prediction with Uncertainty (total)")
    plt.tight_layout()
    if save:
        _, parent_dir = get_current_and_parent_dir()
        safe_name = (
            y_name[-1]
            .replace("$", "")
            .replace("\\", "")
            .replace("{", "_")
            .replace("}", "_")
            .replace(" ", "_")
            .replace("^", "")
        )[:50]
        plt.savefig(
            f"{parent_dir}/figure/sklearn_GP_{safe_name}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()
    plt.show()


# --------------------- NLL & noise extraction ---------------------
def gaussian_nll(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_var: np.ndarray,
    add_const_2pi: bool = False,
) -> float:
    eps = 1e-12
    if add_const_2pi:
        term1 = 0.5 * (np.log(2 * np.pi) + np.log(y_var + eps))
    else:
        term1 = 0.5 * np.log(y_var + eps)
    term2 = 0.5 * (y_true - y_mean) ** 2 / (y_var + eps)
    return float(np.mean(term1 + term2))


def get_noise_level_from_kernel(kernel: Kernel) -> float:
    total = 0.0
    try:
        params = kernel.get_params(deep=True)
    except Exception:
        return 0.0

    seen_ids = set()
    for v in params.values():
        if isinstance(v, WhiteKernel):
            oid = id(v)
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            noise = getattr(v, "noise_level", 0.0)
            try:
                total += float(noise)
            except Exception:
                total += float(np.asarray(noise))
    return total


# --------------------- HPO ---------------------
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
        c0 = float(10 ** rng.uniform(-2, 3))  # 1e-2 ~ 1e3
        n0 = float(10 ** rng.uniform(-6, -0.3))  # ~1e-6 ~ 0.5
        ls0 = 10 ** rng.uniform(-2, 2, size=d)  # per-dim ARD

        kernel = build_kernel_init(d, c0, ls0, n0)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=n_restarts,
            random_state=int(rng.integers(0, 1_000_000_000)),
        ).fit(X_train, y_train)

        mu_val, std_val = cast(
            Tuple[np.ndarray, np.ndarray], gpr.predict(X_val, return_std=True)
        )
        nll = gaussian_nll(y_val, mu_val, std_val**2, add_const_2pi=False)

        if nll < best_nll:
            best_nll = nll
            best_model = gpr

    assert best_model is not None
    return best_model, best_nll


# --------------------- UQ decomposition ---------------------
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
    X_norm = x_scaler.transform(X_raw)
    n_samples = X_norm.shape[0]
    n_outputs = len(models)

    Y_pred_norm = np.zeros((n_samples, n_outputs))
    Y_std_norm = np.zeros((n_samples, n_outputs))
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
        noise_lvl = get_noise_level_from_kernel(
            model.kernel_
        )  # normalized space
        ale_i = np.full_like(var_i, float(noise_lvl))
        epi_i = np.clip(var_i - ale_i, a_min=0.0, a_max=None)

        Y_pred_norm[:, i] = mu_i
        Y_std_norm[:, i] = std_i
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


# --------------------- Inference timing helpers ---------------------
def _make_1000_samples(
    X: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Build a batch of exactly n_samples from X (bootstrap if needed)."""
    if rng is None:
        rng = np.random.default_rng(2025)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Empty X provided for inference timing.")
    if n >= n_samples:
        idx = rng.choice(n, size=n_samples, replace=False)
    else:
        idx = rng.choice(n, size=n_samples, replace=True)
    return X[idx]


def measure_inference_time(
    models: List[GaussianProcessRegressor],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    X_ref: np.ndarray,
    *,
    repeats: int = 30,
    warmup: int = 5,
    n_samples: int = 1000,
) -> Tuple[float, float]:
    """
    Measure end-to-end wall time of gp_predict_with_decomposition on a batch
    of n_samples. Returns (mean_ms, std_ms) over `repeats` (excl. warmup).
    """
    rng = np.random.default_rng(42)
    X_batch = _make_1000_samples(X_ref, n_samples=n_samples, rng=rng)

    # Warmup runs (excluded from stats)
    for _ in range(warmup):
        _ = gp_predict_with_decomposition(models, x_scaler, y_scaler, X_batch)

    times_ms: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = gp_predict_with_decomposition(models, x_scaler, y_scaler, X_batch)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms


# --------------------- K-fold ---------------------
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
        val_losses_nll: List[float] = []

        # HPO per output
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
            val_losses_nll.append(best_nll)

        # metrics on normalized space
        Y_pred_norm = np.zeros_like(Y_val_norm)
        for i, model in enumerate(models):
            mu_val, std_val = cast(
                Tuple[np.ndarray, np.ndarray],
                model.predict(X_val_norm, return_std=True),
            )
            Y_pred_norm[:, i] = mu_val

        val_loss_mse = mean_squared_error(Y_val_norm, Y_pred_norm)
        val_loss_mape = mean_absolute_percentage_error(Y_val_norm, Y_pred_norm)
        val_loss_r2 = r2_score(Y_val_norm, Y_pred_norm)
        val_loss_nll_mean = float(np.mean(val_losses_nll))

        history_mse.append(val_loss_mse)
        history_mape.append(val_loss_mape)
        history_r2.append(val_loss_r2)
        history_nll.append(val_loss_nll_mean)

        print(f"MSE:  {val_loss_mse:.4f}")
        print(f"MAPE: {val_loss_mape:.4f}")
        print(f"R2:   {val_loss_r2:.4f}")
        print(
            f"NLL:  {val_loss_nll_mean:.4f}  (normalized space, total variance)"
        )

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

    plot_loss_history(history_mse, "MSE")
    plot_loss_history(history_mape, "MAPE")
    plot_loss_history(history_r2, "R2")
    plot_loss_history(history_nll, "NLL")


# --------------------- Validation container & figure ---------------------
class _ValidationResult:
    def __init__(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_names: List[str],
        y_std: Optional[np.ndarray] = None,  # total std
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
        self.y_std_ale = y_std_ale
        self.y_std_epi = y_std_epi
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.fig = fig
        self.metrics = metrics or {}
        self.settings = settings or {}

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
    """
    Create validation figure and return Figure without showing.
    FIX: use ax.set_ylabel(...) (was ax.set.ylabel(...))
    """
    fig, axes = plt.subplots(1, len(y_name), figsize=(10, 5))
    if len(y_name) == 1:
        axes = [axes]

    for i, _ in enumerate(y_name):
        ax = axes[i]
        # +/-10% band
        x_line = np.linspace(
            0.95 * np.min(y_valid[:, i]), 1.1 * np.max(y_valid[:, i]), 200
        )
        ax.fill_between(
            x_line, 0.9 * x_line, 1.1 * x_line, alpha=0.4, label="$±10%$"
        )

        # 45° reference
        lo = max(0.0, 0.95 * float(np.min(y_valid[:, i])))
        hi = 1.05 * float(np.max(y_valid[:, i]))
        ax.plot([lo, hi], [lo, hi], lw=1)

        # points / errorbars
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

        # metrics
        r2 = r2_score(y_valid[:, i], y_pred[:, i])
        mse = mean_squared_error(y_valid[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_valid[:, i], y_pred[:, i])
        ax.text(
            0.85 * hi,
            0.65 * hi,
            f"$R^2$={r2:.3f}\nMSE={mse:.3f}\nMAPE={mape:.3f}",
            fontsize=11,
            ha="right",
        )

        # labels
        ax.set_xlabel(f"True {y_name[i]}", fontsize=14)
        ax.set_ylabel(f"Pred {y_name[i]}", fontsize=14)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(
            f"{y_name[i]} Prediction with Uncertainty (total)", fontsize=14
        )
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# --------------------- Excel writing helpers (NEW) ---------------------
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
            # attach raw inputs for plotting flexibility
            for k, col in enumerate(input_cols):
                row[col] = float(X_val_raw[i, k])
            rows.append(row)
    return pd.DataFrame(rows)


def _write_df_to_excel(
    excel_path: str, sheet_name: str, df: pd.DataFrame
) -> None:
    """
    Append/replace a sheet if the file exists (requires openpyxl).
    If openpyxl is unavailable, the file will be overwritten with a new one.
    """
    excel_dir = os.path.dirname(excel_path) or "."
    os.makedirs(excel_dir, exist_ok=True)

    if os.path.exists(excel_path):
        try:
            # append + replace sheet if exists
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
            print(
                f"[WARN] openpyxl append failed ({e}). Will overwrite the file using xlsxwriter."
            )

    # create new or overwrite
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"[OK] Wrote new Excel file: {os.path.abspath(excel_path)}")


# --------------------- One-run interface ---------------------
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
    infer_repeats: int = 30,
    infer_warmup: int = 5,
    infer_n_samples: int = 1000,
    fixed_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # NEW: save per-point UQ to Excel if provided
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: str = "GPR_UQ",
) -> _ValidationResult:
    # NOTE: keep your original IO schema here
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_names = [r"$\dot W_{elec}$", r"$\dot m$"]

    data = load_data(file, sheet)
    X, Y = split_data(data, input_cols, output_cols)
    if fixed_indices is not None:
        train_idx, val_idx = fixed_indices
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
    else:
        x_train, x_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.2, random_state=split_seed, shuffle=True
        )

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
    for i in range(len(output_names)):
        models.append(
            joblib.load(f"{model_folder}/gpr_model_fold1_output{i}.pkl")
        )
    x_scaler: StandardScaler = joblib.load(f"{model_folder}/x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_folder}/y_scaler_fold1.pkl")

    Y_pred, Y_std_total, Y_std_ale, Y_std_epi, _, _, _ = (
        gp_predict_with_decomposition(models, x_scaler, y_scaler, x_val)
    )

    fig = _make_validation_figure(y_val, Y_pred, Y_std_total, output_names)

    # Per-output metrics in ORIGINAL units
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

    # NLL in NORMALIZED space
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

    # Inference timing (ms / 1000 samples)
    infer_mean_ms, infer_std_ms = measure_inference_time(
        models,
        x_scaler,
        y_scaler,
        x_val,
        repeats=infer_repeats,
        warmup=infer_warmup,
        n_samples=infer_n_samples,
    )
    print(
        f"[Inference] {infer_mean_ms:.1f} ± {infer_std_ms:.1f} ms / "
        f"{infer_n_samples} samples (repeats={infer_repeats}, warmup={infer_warmup})"
    )

    # Mean & Std (across outputs)
    mse_mean = float(np.mean(mse_each))
    mse_std = float(np.std(mse_each, ddof=1)) if len(mse_each) > 1 else 0.0
    r2_mean = float(np.mean(r2_each))
    r2_std = float(np.std(r2_each, ddof=1)) if len(r2_each) > 1 else 0.0
    nll_mean_norm = float(np.mean(nll_each))
    nll_std_norm = float(np.std(nll_each, ddof=1)) if len(nll_each) > 1 else 0.0

    metrics = {
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
        # NEW: inference timing
        "Infer_ms_per_1000_mean": infer_mean_ms,
        "Infer_ms_per_1000_std": infer_std_ms,
        "Infer_repeats": infer_repeats,
        "Infer_warmup": infer_warmup,
        "Infer_n_samples": infer_n_samples,
    }

    settings = {
        "k_folds": k_folds,
        "patience": patience,
        "hpo_tries": hpo_tries,
        "hpo_restarts": hpo_restarts,
        "split_seed": split_seed,
        "model_folder": model_folder,
    }

    # -------- NEW: save per-point UQ into Excel if requested --------
    if uq_excel_path is not None:
        uq_df = _build_uq_long_table(
            model_name="GPR (sklearn)",
            output_names=output_names,
            input_cols=input_cols,
            X_val_raw=x_val,
            y_true=y_val,
            y_pred=Y_pred,
            y_std_total=Y_std_total,
            y_std_epi=Y_std_epi,
            y_std_ale=Y_std_ale,
        )
        _write_df_to_excel(uq_excel_path, uq_sheet_name, uq_df)

    return _ValidationResult(
        name="GPR (sklearn)",
        y_true=y_val,
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


# --------------------- Repeat runs + Excel export ---------------------
def _col_desc_map() -> Dict[str, str]:
    # Add metric columns + uncertainty columns for README
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
        # NEW: inference timing per run
        "Infer_ms_per_1000_mean": "End-to-end inference wall time over 1000 samples (mean over repeats); ms/1000 samples",
        "Infer_ms_per_1000_std": "Std of the above across repeats (ddof=1); ms/1000 samples",
        # W_dot uncertainty summary (per-run, over validation samples)
        "W_dot_pred_mean": "Mean of predicted W_dot over validation samples",
        "W_dot_pred_std": "Std of predicted W_dot over validation samples",
        "W_dot_total_std_mean": "Mean of TOTAL predictive std over validation samples (original units)",
        "W_dot_total_std_std": "Std of TOTAL predictive std over validation samples",
        "W_dot_ale_std_mean": "Mean of ALEATORY predictive std over validation samples",
        "W_dot_ale_std_std": "Std of ALEATORY predictive std over validation samples",
        "W_dot_epi_std_mean": "Mean of EPISTEMIC predictive std over validation samples",
        "W_dot_epi_std_std": "Std of EPISTEMIC predictive std over validation samples",
        # m_dot uncertainty summary
        "m_dot_pred_mean": "Mean of predicted m_dot over validation samples",
        "m_dot_pred_std": "Std of predicted m_dot over validation samples",
        "m_dot_total_std_mean": "Mean of TOTAL predictive std over validation samples (original units)",
        "m_dot_total_std_std": "Std of TOTAL predictive std over validation samples",
        "m_dot_ale_std_mean": "Mean of ALEATORY predictive std over validation samples",
        "m_dot_ale_std_std": "Std of ALEATORY predictive std over validation samples",
        "m_dot_epi_std_mean": "Mean of EPISTEMIC predictive std over validation samples",
        "m_dot_epi_std_std": "Std of EPISTEMIC predictive std over validation samples",
        # NEW: per-output performance (single-run values; cross-run mean/std in per_output_metrics)
        "W_dot_R2": "Per-output R^2 for W_dot on validation set (ORIGINAL units)",
        "W_dot_MSE": "Per-output MSE for W_dot on validation set (ORIGINAL units)",
        "W_dot_NLL_norm": "Per-output Gaussian NLL for W_dot on validation set (NORMALIZED space)",
        "m_dot_R2": "Per-output R^2 for m_dot on validation set (ORIGINAL units)",
        "m_dot_MSE": "Per-output MSE for m_dot on validation set (ORIGINAL units)",
        "m_dot_NLL_norm": "Per-output Gaussian NLL for m_dot on validation set (NORMALIZED space)",
    }
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
    k_folds: int = 5,
    patience: int = 5,
    hpo_tries: int = 15,
    hpo_restarts: int = 5,
    model_folder: str = "models/gpr",
    infer_repeats: int = 30,
    infer_warmup: int = 5,
    infer_n_samples: int = 1000,
) -> str:
    os.makedirs(os.path.dirname(excel_filename) or ".", exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seeds = [base_seed + i for i in range(n_runs)]

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Repeat] Run {run_idx}/{n_runs}, seed={seed}")
        res = run_model_gpr(
            file,
            sheet,
            split_seed=seed,
            k_folds=k_folds,
            patience=patience,
            hpo_tries=hpo_tries,
            hpo_restarts=hpo_restarts,
            model_folder=model_folder,
            infer_repeats=infer_repeats,
            infer_warmup=infer_warmup,
            infer_n_samples=infer_n_samples,
            # Do not write per-point UQ during repeats to avoid huge files
            uq_excel_path=None,
        )
        res.check()

        assert (
            res.y_std is not None
            and res.y_std_ale is not None
            and res.y_std_epi is not None
        )

        # Uncertainty stats over validation samples (per output)
        pred_mean, pred_std = _mean_std_over_samples(res.y_pred)
        total_mean, total_std = _mean_std_over_samples(res.y_std)
        ale_mean, ale_std = _mean_std_over_samples(res.y_std_ale)
        epi_mean, epi_std = _mean_std_over_samples(res.y_std_epi)

        # Per-run metric summaries across outputs
        metrics: Dict[str, Any] = res.metrics or {}
        r2_each = list(metrics.get("R2_each", []))
        mse_each = list(metrics.get("MSE_each", []))
        nll_each_norm = list(metrics.get("NLL_each_norm", []))

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

        infer_mean_ms = float(metrics.get("Infer_ms_per_1000_mean", np.nan))
        infer_std_ms = float(metrics.get("Infer_ms_per_1000_std", np.nan))

        row: Dict[str, Any] = {
            "Run": run_idx,
            "Seed": seed,
            # Per-run metrics (across outputs)
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "NLL_mean_norm": nll_mean_norm,
            "NLL_std_norm": nll_std_norm,
            # NEW: inference timing
            "Infer_ms_per_1000_mean": infer_mean_ms,
            "Infer_ms_per_1000_std": infer_std_ms,
            # W_dot (index 0)
            "W_dot_pred_mean": float(pred_mean[0]),
            "W_dot_pred_std": float(pred_std[0]),
            "W_dot_total_std_mean": float(total_mean[0]),
            "W_dot_total_std_std": float(total_std[0]),
            "W_dot_ale_std_mean": float(ale_mean[0]),
            "W_dot_ale_std_std": float(ale_std[0]),
            "W_dot_epi_std_mean": float(epi_mean[0]),
            "W_dot_epi_std_std": float(epi_std[0]),
            # m_dot (index 1)
            "m_dot_pred_mean": float(pred_mean[1]),
            "m_dot_pred_std": float(pred_std[1]),
            "m_dot_total_std_mean": float(total_mean[1]),
            "m_dot_total_std_std": float(total_std[1]),
            "m_dot_ale_std_mean": float(ale_mean[1]),
            "m_dot_ale_std_std": float(ale_std[1]),
            "m_dot_epi_std_mean": float(epi_mean[1]),
            "m_dot_epi_std_std": float(epi_std[1]),
            # NEW: store single-run per-output metrics for later cross-run aggregation
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

    # Overall mean/std across runs for numeric columns
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

    # === NEW: per-output R2/MSE/NLL_norm cross-run mean/std in a tidy sheet ===
    def _series_clean(col: str) -> pd.Series:
        # consider only numeric Run rows to exclude MEAN/STD rows
        mask_numeric_run = per_run_df["Run"].apply(
            lambda v: isinstance(v, (int, np.integer))
        )
        return pd.to_numeric(
            per_run_df.loc[mask_numeric_run, col], errors="coerce"
        ).dropna()

    per_output_rows: List[Dict[str, Any]] = []
    for out_name in ["W_dot", "m_dot"]:
        r2_s = _series_clean(f"{out_name}_R2")
        mse_s = _series_clean(f"{out_name}_MSE")
        nll_s = _series_clean(f"{out_name}_NLL_norm")
        per_output_rows.append(
            {
                "output": out_name,
                "R2_mean": float(r2_s.mean()) if len(r2_s) else float("nan"),
                "R2_std": float(r2_s.std(ddof=1))
                if len(r2_s) > 1
                else float("nan"),
                "MSE_mean": float(mse_s.mean()) if len(mse_s) else float("nan"),
                "MSE_std": float(mse_s.std(ddof=1))
                if len(mse_s) > 1
                else float("nan"),
                "NLL_norm_mean": float(nll_s.mean())
                if len(nll_s)
                else float("nan"),
                "NLL_norm_std": float(nll_s.std(ddof=1))
                if len(nll_s) > 1
                else float("nan"),
            }
        )
    per_output_df = pd.DataFrame(per_output_rows)

    # README + SETTINGS
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
                "hpo_tries",
                "hpo_restarts",
                "model_folder",
                "file",
                "sheet",
                # NEW timing params
                "infer_repeats",
                "infer_warmup",
                "infer_n_samples",
            ],
            "value": [
                n_runs,
                base_seed,
                k_folds,
                patience,
                hpo_tries,
                hpo_restarts,
                model_folder,
                file,
                sheet,
                infer_repeats,
                infer_warmup,
                infer_n_samples,
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
    print(f"\n[OK] Saved Excel summary to: {abs_path}")
    return abs_path


# --------------------- Main process ---------------------
def main_process(
    file_path: str,
    sheet_name: str,
    model_folder: str = "models/gpr",
    *,
    repeat_runs: int = 0,
    excel_filename: Optional[str] = None,
    infer_repeats: int = 30,
    infer_warmup: int = 5,
    infer_n_samples: int = 1000,
    # NEW: per-point UQ excel for single-run path
    uq_excel_path: Optional[str] = None,
    uq_sheet_name: str = "GPR_UQ",
) -> None:
    if repeat_runs and excel_filename:
        repeat_runs_to_excel(
            file=file_path,
            sheet=sheet_name,
            excel_filename=excel_filename,
            n_runs=repeat_runs,
            base_seed=1234,
            k_folds=5,
            patience=5,
            hpo_tries=15,
            hpo_restarts=5,
            model_folder=model_folder,
            infer_repeats=infer_repeats,
            infer_warmup=infer_warmup,
            infer_n_samples=infer_n_samples,
        )
        return

    # Original single-run path
    data = load_data(file_path, sheet_name)
    input_cols = ["T_suc", "P_suc", "T_dis", "P_dis"]
    output_cols = ["W_dot", "m_dot"]
    output_names = [r"$\dot W_{elec}$", r"$\dot m$"]

    X, Y = split_data(data, input_cols, output_cols)
    x_train, x_val, y_train, y_val = split_train_valid(X, Y)

    t_start = time.time()
    k_fold_cross_validation(
        x_train, y_train, k=5, patience=5, hpo_tries=15, hpo_restarts=5
    )
    t_end = time.time()
    print(f"\n[Timing] Total training time: {t_end - t_start:.1f} seconds")

    models: List[GaussianProcessRegressor] = []
    for i in range(len(output_names)):
        models.append(
            joblib.load(f"{model_folder}/gpr_model_fold1_output{i}.pkl")
        )
    x_scaler: StandardScaler = joblib.load(f"{model_folder}/x_scaler_fold1.pkl")
    y_scaler: StandardScaler = joblib.load(f"{model_folder}/y_scaler_fold1.pkl")

    Y_pred, Y_std_total, Y_std_ale, Y_std_epi, ale_var, epi_var, total_var = (
        gp_predict_with_decomposition(models, x_scaler, y_scaler, x_val)
    )

    # Print single-run inference timing as well
    infer_mean_ms, infer_std_ms = measure_inference_time(
        models,
        x_scaler,
        y_scaler,
        x_val,
        repeats=infer_repeats,
        warmup=infer_warmup,
        n_samples=infer_n_samples,
    )
    print(
        f"[Inference] {infer_mean_ms:.1f} ± {infer_std_ms:.1f} ms / "
        f"{infer_n_samples} samples (repeats={infer_repeats}, warmup={infer_warmup})"
    )

    plot_validation_group(y_val, Y_pred, Y_std_total, output_names)

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
    for j, name in enumerate(output_names):
        print(f"{name}: NLL = {nll_per_output[j]:.6f}")
    print(f"Mean NLL: {np.mean(nll_per_output):.6f}")

    for j, name in enumerate(output_names):
        print(
            f"\n=== Uncertainty decomposition for {name} (ORIGINAL units) ==="
        )
        print(f"Mean total std     : {Y_std_total[:, j].mean():.6g}")
        print(f"Mean aleatory std  : {Y_std_ale[:, j].mean():.6g}")
        print(f"Mean epistemic std : {Y_std_epi[:, j].mean():.6g}")
        denom = total_var[:, j].mean()
        frac_ale = (ale_var[:, j].mean() / denom) if denom > 0 else np.nan
        print(f"Aleatory fraction of variance (mean): {frac_ale:.3f}")

    # NEW: build and optionally write per-point UQ table for this single run
    if uq_excel_path is not None:
        uq_df = _build_uq_long_table(
            model_name="GPR (sklearn)",
            output_names=output_names,
            input_cols=input_cols,
            X_val_raw=x_val,
            y_true=y_val,
            y_pred=Y_pred,
            y_std_total=Y_std_total,
            y_std_epi=Y_std_epi,
            y_std_ale=Y_std_ale,
        )
        _write_df_to_excel(uq_excel_path, uq_sheet_name, uq_df)

    os.makedirs("outputs", exist_ok=True)
    out_df = pd.DataFrame(
        np.hstack(
            [
                Y_pred,
                Y_std_total,
                Y_std_ale,
                Y_std_epi,
                ale_var,
                epi_var,
                total_var,
            ]
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
    out_df.to_csv(f"outputs/gpr_UQ_decomp_{sheet_name}.csv", index=False)
    print(
        f"\nSaved detailed decomposition to outputs/gpr_UQ_decomp_{sheet_name}.csv"
    )


if __name__ == "__main__":
    file_path = "./data/case_2_scroll.csv"
    sheet_name = "case_2_scroll"
    # file_path = "./data/case_1_rotary.csv"
    # sheet_name = "case_1_rotary"

    # Option A: original single run + (NEW) per-point UQ Excel
    main_process(
        file_path,
        sheet_name,
        uq_excel_path=f"outputs/UQ_decomp_{sheet_name}.xlsx",  # <- 由你指定路径/文件名
        uq_sheet_name="GPR_UQ",  # <- 表名你也可自定
    )

    # Option B: repeated runs + Excel export (summary only; no per-point UQ)
    # main_process(
    #     file_path,
    #     sheet_name,
    #     repeat_runs=10,
    #     excel_filename=f"outputs/GPR_repeat_{sheet_name}.xlsx",
    #     infer_repeats=30,
    #     infer_warmup=5,
    #     infer_n_samples=1000,
    # )

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()
plt.show = lambda *args, **kwargs: None

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.de_standard import run_model_del as run_model_del_standard
from models.de_vi import run_model_del as run_model_del_vi


def build_case_config() -> dict[str, dict[str, object]]:
    return {
        "case_1_rotary": {
            "data_file": PACKAGE_ROOT / "data" / "case_1_rotary.csv",
            "sheet": "case_1_rotary",
            "runner": run_model_del_standard,
            "kwargs": {
                "k_folds": 3,
                "patience": 15,
                "n_estimators": 3,
                "hidden_layers": (32, 16),
                "lr": 1e-3,
                "batch": 16,
                "epochs": 60,
                "dropout": 0.0,
                "activation": "relu",
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "de" / "case_1_rotary"),
                "infer_repeats": 3,
                "infer_warmup": 1,
                "infer_n_samples": 200,
            },
        },
        "case_2_scroll": {
            "data_file": PACKAGE_ROOT / "data" / "case_2_scroll.csv",
            "sheet": "case_2_scroll",
            "runner": run_model_del_standard,
            "kwargs": {
                "k_folds": 3,
                "patience": 15,
                "n_estimators": 3,
                "hidden_layers": (32, 16),
                "lr": 1e-3,
                "batch": 16,
                "epochs": 60,
                "dropout": 0.0,
                "activation": "relu",
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "de" / "case_2_scroll"),
                "infer_repeats": 3,
                "infer_warmup": 1,
                "infer_n_samples": 200,
            },
        },
        "case_3_vi": {
            "data_file": PACKAGE_ROOT / "data" / "case_3_vi.csv",
            "sheet": "case_3_vi",
            "runner": run_model_del_vi,
            "kwargs": {
                "k_folds": 3,
                "patience": 15,
                "n_estimators": 3,
                "hidden_layers": (32, 16),
                "lr": 1e-3,
                "batch": 16,
                "epochs": 60,
                "dropout": 0.0,
                "activation": "relu",
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "de_vi" / "case_3_vi"),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DE reviewer example.")
    parser.add_argument(
        "--case",
        choices=["case_1_rotary", "case_2_scroll", "case_3_vi"],
        default="case_2_scroll",
        help="Select which packaged dataset to run.",
    )
    args = parser.parse_args()

    config = build_case_config()[args.case]
    runner = config["runner"]
    data_file = Path(config["data_file"])
    kwargs = dict(config["kwargs"])

    result = runner(str(data_file), str(config["sheet"]), **kwargs)
    print(f"Completed DE example for {args.case}")
    print(f"Data file: {data_file.name}")
    print(f"Outputs: {result.output_names}")
    if result.metrics is not None:
        pprint(result.metrics)


if __name__ == "__main__":
    main()

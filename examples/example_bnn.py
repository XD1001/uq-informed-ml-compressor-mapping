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

from models.bnn_standard import run_model_bnn as run_model_bnn_standard
from models.bnn_vi import run_model_bnn as run_model_bnn_vi


def build_case_config() -> dict[str, dict[str, object]]:
    return {
        "case_1_rotary": {
            "data_file": PACKAGE_ROOT / "data" / "case_1_rotary.csv",
            "sheet": "case_1_rotary",
            "runner": run_model_bnn_standard,
            "kwargs": {
                "layer_sizes": [4, 16, 8, 2],
                "activation": "relu",
                "prior_scale": 1.0,
                "sigma_prior": "uniform",
                "dropout_rate": 0.0,
                "num_samples": 40,
                "warmup_steps": 40,
                "num_chains": 1,
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "bnn" / "case_1_rotary"),
                "infer_repeats": 2,
                "infer_warmup": 1,
                "infer_n_samples": 100,
            },
        },
        "case_2_scroll": {
            "data_file": PACKAGE_ROOT / "data" / "case_2_scroll.csv",
            "sheet": "case_2_scroll",
            "runner": run_model_bnn_standard,
            "kwargs": {
                "layer_sizes": [4, 16, 8, 2],
                "activation": "relu",
                "prior_scale": 1.0,
                "sigma_prior": "uniform",
                "dropout_rate": 0.0,
                "num_samples": 40,
                "warmup_steps": 40,
                "num_chains": 1,
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "bnn" / "case_2_scroll"),
                "infer_repeats": 2,
                "infer_warmup": 1,
                "infer_n_samples": 100,
            },
        },
        "case_3_vi": {
            "data_file": PACKAGE_ROOT / "data" / "case_3_vi.csv",
            "sheet": "case_3_vi",
            "runner": run_model_bnn_vi,
            "kwargs": {
                "layer_sizes": (7, 16, 8, 3),
                "activation": "relu",
                "prior_scale": 1.0,
                "noise_prior": "half_cauchy",
                "num_samples": 40,
                "warmup_steps": 40,
                "num_chains": 1,
                "dropout_rate": 0.0,
                "target_accept_prob": 0.9,
                "model_dir": str(PACKAGE_ROOT / "artifacts" / "bnn_vi" / "case_3_vi"),
                "infer_repeats": 2,
                "infer_warmup": 1,
                "infer_n_samples": 100,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BNN reviewer example.")
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
    print(f"Completed BNN example for {args.case}")
    print(f"Data file: {data_file.name}")
    print(f"Outputs: {result.output_names}")
    if result.metrics is not None:
        pprint(result.metrics)


if __name__ == "__main__":
    main()

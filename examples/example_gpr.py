from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from pprint import pprint

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

plt.ioff()
plt.show = lambda *args, **kwargs: None
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models.gpr_standard import run_model_gpr as run_model_gpr_standard
from models.gpr_vi import run_model_gpr as run_model_gpr_vi


def build_case_config() -> dict[str, dict[str, object]]:
    return {
        "case_1_rotary": {
            "data_file": PACKAGE_ROOT / "data" / "case_1_rotary.csv",
            "sheet": "case_1_rotary",
            "runner": run_model_gpr_standard,
            "kwargs": {
                "k_folds": 3,
                "patience": 3,
                "hpo_tries": 4,
                "hpo_restarts": 1,
                "model_folder": str(PACKAGE_ROOT / "artifacts" / "gpr" / "case_1_rotary"),
                "infer_repeats": 3,
                "infer_warmup": 1,
                "infer_n_samples": 200,
            },
        },
        "case_2_scroll": {
            "data_file": PACKAGE_ROOT / "data" / "case_2_scroll.csv",
            "sheet": "case_2_scroll",
            "runner": run_model_gpr_standard,
            "kwargs": {
                "k_folds": 3,
                "patience": 3,
                "hpo_tries": 4,
                "hpo_restarts": 1,
                "model_folder": str(PACKAGE_ROOT / "artifacts" / "gpr" / "case_2_scroll"),
                "infer_repeats": 3,
                "infer_warmup": 1,
                "infer_n_samples": 200,
            },
        },
        "case_3_vi": {
            "data_file": PACKAGE_ROOT / "data" / "case_3_vi.csv",
            "sheet": "case_3_vi",
            "runner": run_model_gpr_vi,
            "kwargs": {
                "k_folds": 3,
                "patience": 3,
                "hpo_tries": 4,
                "hpo_restarts": 1,
                "model_folder": str(PACKAGE_ROOT / "artifacts" / "gpr_vi" / "case_3_vi"),
                "infer_repeats": 3,
                "infer_warmup": 1,
                "infer_n_samples": 200,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPR reviewer example.")
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
    print(f"Completed GPR example for {args.case}")
    print(f"Data file: {data_file.name}")
    print(f"Outputs: {result.output_names}")
    if result.metrics is not None:
        pprint(result.metrics)


if __name__ == "__main__":
    main()

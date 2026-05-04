# UQ-Informed Machine Learning for Compressor Mapping

This repository contains cleaned reproducibility materials for the accepted Applied Thermal Engineering article:

> Xin Ding and Davide Ziviani, "Uncertainty Quantification-Informed Machine Learning Models For Compressor Performance Mapping", Applied Thermal Engineering, accepted for publication.

The repository provides representative Python implementations and assembled CSV data files used to demonstrate uncertainty-quantification-informed machine-learning models for compressor performance mapping.

## Repository Contents

- `data/`: cleaned CSV exports for the three manuscript case studies.
- `models/`: source-code modules for Gaussian Process Regression (GPR), Deep Ensembles (DE), and Bayesian Neural Networks (BNN).
- `examples/`: lightweight example scripts that run the model workflows on the packaged datasets.
- `requirements.txt`: Python package dependencies.
- `environment.yml`: Conda environment specification.

Generated cache files, trained pickle artifacts, and intermediate model outputs are intentionally excluded from the repository. Running the examples may create an `artifacts/` folder locally; this folder is ignored by Git.

## Case Mapping

- `case_1_rotary.csv`: constant-speed R410A dual-cylinder rolling-piston compressor.
- `case_2_scroll.csv`: constant-speed R454C scroll compressor.
- `case_3_vi.csv`: variable-speed R410A vapor-injection scroll compressor.

The assembled datasets were compiled from previously published and publicly accessible literature sources cited in the article.

## Quick Start

Create an environment:

```bash
conda env create -f environment.yml
conda activate uq-compressor-mapping
```

Alternatively, install with pip:

```bash
python -m pip install -r requirements.txt
```

Run an example from the repository root:

```bash
python examples/example_gpr.py --case case_2_scroll
python examples/example_de.py --case case_2_scroll
python examples/example_bnn.py --case case_2_scroll
```

Other supported cases are:

```bash
python examples/example_gpr.py --case case_1_rotary
python examples/example_de.py --case case_3_vi
python examples/example_bnn.py --case case_3_vi
```

The example settings are intentionally lighter than the full manuscript runs so that users can inspect the workflow quickly.

## Reproducibility Notes

- The manuscript used Python 3.12 and the package versions listed in `requirements.txt`.
- Some model-training routines use random splits, stochastic optimization, or sampling-based inference, so exact numerical values may vary slightly between runs.
- The examples write generated outputs under `artifacts/`; these files are not required as input and are not tracked.
- No `.pyc`, `__pycache__`, or trained `.pkl` files are part of the public source package.

## License

The final reuse license is pending author/institution confirmation. See `LICENSE.md`.

## Citation

Please cite the associated journal article if you use this repository. A machine-readable citation file is provided in `CITATION.cff`.

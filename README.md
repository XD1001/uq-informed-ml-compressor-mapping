# UQ-Informed Machine Learning for Compressor Mapping

This repository contains cleaned reproducibility materials for the accepted Applied Thermal Engineering article:

> Xin Ding and Davide Ziviani, "Uncertainty Quantification-Informed Machine Learning Models For Compressor Performance Mapping", Applied Thermal Engineering, accepted for publication.

The repository provides representative Python implementations and assembled CSV data files used to demonstrate uncertainty-quantification-informed machine-learning models for compressor performance mapping.

## Repository Contents

- `data/`: cleaned CSV exports for the three manuscript case studies.
- `models/`: source-code modules for Gaussian Process Regression (GPR), Deep Ensembles (DE), and Bayesian Neural Networks (BNN).
- `examples/`: lightweight example scripts that run the model workflows on the packaged datasets.
- `pyproject.toml`: primary Python dependency specification for `uv`.
- `uv.lock`: locked `uv` dependency resolution for reproducible setup.
- `requirements.txt`: pip-compatible dependency list.
- `environment.yml`: Conda environment specification for users who prefer Conda.

Generated cache files, trained pickle artifacts, and intermediate model outputs are intentionally excluded from the repository. Running the examples may create an `artifacts/` folder locally; this folder is ignored by Git.

## Case Mapping

- `case_1_rotary.csv`: constant-speed R410A dual-cylinder rolling-piston compressor.
- `case_2_scroll.csv`: constant-speed R454C scroll compressor.
- `case_3_vi.csv`: variable-speed R410A vapor-injection scroll compressor.

The assembled datasets were compiled from previously published and publicly accessible literature sources cited in the article. The original source for each case is:

- `case_1_rotary.csv`: Ma, J., Ding, X., Horton, W. T., and Ziviani, D. (2020), "Development of an automated compressor performance mapping using artificial neural network and multiple compressor technologies", *International Journal of Refrigeration*, 120, 66-80. https://doi.org/10.1016/j.ijrefrig.2020.08.001
- `case_2_scroll.csv`: Hou, W., Raeisi Fard, H., Burns, L., Groll, E. A., Ziviani, D., and Braun, J. E. (2022), "Experimental Investigation of R454C as a Replacement for R410A in a Residential Heat Pump Split System", *International Refrigeration and Air Conditioning Conference at Purdue*, Paper 2482. https://docs.lib.purdue.edu/iracc/2482/
- `case_3_vi.csv`: Dardenne, L., Fraccari, E., Maggioni, A., Molinaroli, L., Proserpio, L., and Winandy, E. (2015), "Semi-empirical modelling of a variable speed scroll compressor with vapour injection", *International Journal of Refrigeration*, 54, 76-87. https://doi.org/10.1016/j.ijrefrig.2015.03.004

## Quick Start

The recommended setup uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run an example from the repository root:

```bash
uv run python examples/example_gpr.py --case case_2_scroll
uv run python examples/example_de.py --case case_2_scroll
uv run python examples/example_bnn.py --case case_2_scroll
```

Other supported cases are:

```bash
uv run python examples/example_gpr.py --case case_1_rotary
uv run python examples/example_de.py --case case_3_vi
uv run python examples/example_bnn.py --case case_3_vi
```

Conda remains supported as an alternative:

```bash
conda env create -f environment.yml
conda activate uq-compressor-mapping
```

Pip can also be used directly:

```bash
python -m pip install -r requirements.txt
python examples/example_gpr.py --case case_2_scroll
```

The example settings are intentionally lighter than the full manuscript runs so that users can inspect the workflow quickly.

## Reproducibility Notes

- The manuscript used Python 3.12 and the package versions listed in `requirements.txt`.
- `uv.lock` captures the dependency resolution used for the `uv` workflow.
- Some model-training routines use random splits, stochastic optimization, or sampling-based inference, so exact numerical values may vary slightly between runs.
- The examples write generated outputs under `artifacts/`; these files are not required as input and are not tracked.
- No `.pyc`, `__pycache__`, or trained `.pkl` files are part of the public source package.

## Release

The accepted-paper repository snapshot is versioned as `v1.0.0`.

## License

This repository uses separate licenses for code and data:

- Source code and documentation are licensed under the MIT License. See `LICENSE.md`.
- CSV data files in `data/` are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `DATA_LICENSE.md`.

The assembled datasets were compiled from previously published and publicly accessible literature sources cited in the article and listed in the Case Mapping section above. Users should also cite the original data sources where appropriate.

## Citation

Please cite the associated journal article if you use this repository. A machine-readable citation file is provided in `CITATION.cff`.

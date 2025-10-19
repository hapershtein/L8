# Documentation for simulation6

This folder contains artifacts and instructions for `simulation6.py`, a multivariate linear regression simulation.

## Purpose

`simulation6.py` generates a synthetic dataset (1000 samples, 50 features), creates a noisy linear target y using randomized true coefficients, fits a multiple linear regression using `np.linalg.lstsq`, and saves a diagnostic scatter plot and a results summary.

## Files produced

- `simulation6.png` — scatter plot showing Actual (y) vs Predicted (y_predicted).
- `Results.md` — markdown summary with the generated plot, input parameters, excerpts of original and regression equations, first few coefficients, and R² metric.
- `simulation6.py` — the simulation script (one directory up at repository root).
- `requirements.txt` — Python package requirements for running the scripts.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- reportlab (optional, for PRD PDF generation)

Install with:

```powershell
python -m pip install -r requirements.txt
```

## How to run

From repository root (`c:\25D\L8`):

PowerShell:

```powershell
python .\simulation6.py
```

CMD:

```cmd
python simulation6.py
```

Running the script will create (or overwrite) `Documentation/simulation6.png` and `Documentation/Results.md`.

## Quick verification

- Open `Documentation/Results.md` to view the embedded plot and summary.
- Check R² and the first few estimated coefficients to verify model behavior.

## Next steps / suggestions

- Generate a formal PRD PDF (`simulation6_PRD.pdf`) summarizing the script and metrics.
- Add normalization and compare metrics before/after normalization.
- Add a small unit test verifying shapes and reproducibility with the fixed random seed.
- Consider adding Ridge/Lasso and feature importance analysis.

---

Created automatically to accompany `simulation6.py`. If you want a richer README (badges, CI instructions), tell me what to include.

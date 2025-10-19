# Tasks for `simulation6.py`

## Overview

`simulation6.py` generates a synthetic multivariate dataset (1000 samples, 50 features), uses linear algebra to fit a multiple linear regression (via `np.linalg.lstsq`), and saves a scatter plot `Documentation/simulation6.png` that shows actual vs predicted values with a regression reference line. This `tasks.md` captures immediate tasks, how to run the script, dependencies, and acceptance criteria.

## Immediate tasks

- [ ] Run `simulation6.py` and verify `Documentation/simulation6.png` is created and looks reasonable.
- [x] Create this `Documentation/tasks.md` (current)
- [ ] Generate a PRD PDF (`Documentation/simulation6_PRD.pdf`) summarizing the script, inputs, outputs, algorithm, and metrics (R²). A helper script using ReportLab or similar is recommended.
- [ ] Add `requirements.txt` listing required packages: `numpy`, `matplotlib`, `reportlab` (if generating PRD), pin versions if desired.
- [ ] Add a short `Documentation/README.md` with purpose and run instructions.
- [ ] Improve inline comments and add minimal unit tests for the data generation and regression steps.

## How to run

From the repository root (`c:\25D\L8`):

PowerShell:

```powershell
python .\simulation6.py
```

CMD:

```cmd
python simulation6.py
```

If you plan to generate the PRD PDF with a helper script `Documentation/generate_prd.py` (not included by default), run:

```powershell
python .\Documentation\generate_prd.py
```

## Dependencies

- Python 3.8+ (recommended)
- numpy
- matplotlib
- reportlab (optional, for PDF generation)

You can install quickly with:

```powershell
python -m pip install numpy matplotlib reportlab
```

Or, create `requirements.txt` and run `pip install -r requirements.txt`.

## Outputs

- `Documentation/simulation6.png` — scatter plot of actual vs predicted
- `Documentation/simulation6_PRD.pdf` — (optional) product requirements document for this simulation

## Acceptance criteria

- `simulation6.py` runs without uncaught exceptions and writes `Documentation/simulation6.png`.
- PRD PDF (if generated) contains script summary, inputs, outputs, algorithm steps, and at least one quick metric such as R².

## Next improvements / backlog

- Normalize features and compare model performance (R², RMSE).
- Add covariance / correlation analysis for feature importance.
- Replace `np.linalg.lstsq` with regularized solvers (Ridge/Lasso) and compare.
- Add automated test(s) for reproducibility (seeded RNG) and a CI job to run small subset tests.

---

Created from `simulation6.py` (located at repository root).
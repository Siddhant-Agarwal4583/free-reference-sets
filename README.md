# Free Reference Sets — code, data and paper source

Label-free accuracy estimation under distribution shift for bagged ensembles, using out-of-bag
(OOB) predictions as a free labeled reference set, with size-matched 1/m extrapolation.

## Layout
| Path | What it is |
|---|---|
| `estimators.py` | All estimators: naive OOB, size-matched (-M), size-extrapolated (-X), holdout baselines, OOB domain classifier |
| `shifts.py` | Covariate (sub-population), corruption and label-shift generators |
| `bench.py` | Experiment runner: `python bench.py {main,tsweep,nsweep,real,et} [n_jobs]` |
| `mech.py` | Mechanism experiment (matched-size gap, 1/m fits) |
| `bench_openml.py` | **OpenML-CC18 runner — not yet run; run this on a machine with internet** |
| `analyze.py`, `paper_stats.py` | All tables, statistics and figures |
| `HYPOTHESES.md` | Pre-registered hypotheses (written before the main runs) |
| `results/*.csv` | Raw per-task results for all 3,700 tasks |
| `tables/`, `figs/` | Generated tables and figures |
| `paper/` | LaTeX source (`main.tex`), bibliography, compiled `main.pdf` |

## Reproduce everything (≈3 h on 2 CPU cores)
```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib
for e in main tsweep nsweep real et; do python bench.py $e 2; done
python mech.py
python analyze.py all && python paper_stats.py
cd paper && latexmk -pdf main.tex
```

## Strongly recommended before submission
```bash
pip install openml
python bench_openml.py --jobs 2 --seeds 3   # 72 datasets x 7 shifts x 3 seeds, resumable
# or open run_openml_colab.ipynb in Google Colab (no installation needed)
python analyze.py openml
```
Then add the OpenML results as a main real-data section (the current real-data evidence uses
only 3 small scikit-learn datasets, which reviewers will flag).

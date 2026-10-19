"""Run the full benchmark on the OpenML-CC18 suite (72 real classification datasets).

Usage (needs internet access to openml.org):
    pip install openml scikit-learn pandas joblib
    python bench_openml.py --jobs 2 --seeds 3

Resumable: finished jobs are recorded in results/openml_done.txt, so if the run is
interrupted, simply run the same command again and it continues where it stopped.
Output: results/openml.csv (same format as bench.py); analyse with  python analyze.py openml
"""
import sys, os, argparse, traceback
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from estimators import run_one
from shifts import prep
from bench import SHIFTS, make_task

MAX_N = 20000          # subsample very large datasets for tractability
MIN_N = 500            # skip datasets too small to split into source/target
CACHE = "openml_cache"


def fetch(did):
    """Download one OpenML dataset and store a preprocessed copy as .npz."""
    import openml
    f = os.path.join(CACHE, f"{did}.npz")
    if os.path.exists(f):
        return f
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    X, y, cat, _ = ds.get_data(target=ds.default_target_attribute)
    X = X.copy()
    for c, is_cat in zip(X.columns, cat):
        if is_cat or X[c].dtype == object:
            X[c] = OrdinalEncoder().fit_transform(X[[c]].astype(str))[:, 0]
    X = SimpleImputer().fit_transform(X.astype(float))
    y = pd.factorize(y)[0]
    if len(y) > MAX_N:
        idx = np.random.default_rng(0).choice(len(y), MAX_N, replace=False)
        X, y = X[idx], y[idx]
    np.savez_compressed(f, X=X, y=y)
    return f


def load(did):
    d = np.load(os.path.join(CACHE, f"{did}.npz"))
    return prep(d["X"], d["y"])


def job(did, shift, level, seed, T=200, loader=load):
    try:
        X, y = loader(did)
        if len(y) < MIN_N:
            return "skip", []
        rng = np.random.default_rng(seed * 7919 + did)
        n_s = min(2000, len(y) // 2); n_t = min(2000, len(y) // 2)
        Xs, ys, Xt, yt = make_task(X, y, shift, level, rng, n_s, n_t)
        keep = np.isin(yt, np.unique(ys)); Xt, yt = Xt[keep], yt[keep]
        if len(np.unique(ys)) < 2 or len(yt) < 50:
            return "skip", []
        ys_map = {c: i for i, c in enumerate(np.unique(ys))}
        ys = np.array([ys_map[c] for c in ys]); yt = np.array([ys_map[c] for c in yt])
        rows, diag = run_one(Xs, ys, Xt, yt, T=T, seed=seed)
        return "ok", [dict(exp="openml", dataset=f"oml{did}", shift=shift, level=level, n_s=len(ys),
                           n_t=len(yt), T=T, kind="rf", seed=seed, estimator=k, estimate=v, truth=t,
                           abs_err=abs(v - t), err=v - t, acc_full=diag["acc_full"],
                           acc_ho_model=diag["acc_holdout_model"]) for k, (v, t) in rows.items()]
    except Exception:
        return "error: " + traceback.format_exc(limit=2).replace("\n", " | "), []


def run(dids, nj, seeds, loader=load, out="results/openml.csv"):
    os.makedirs("results", exist_ok=True)
    done_path = out.replace(".csv", "_done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    J = [(d, s, l, seed) for d in dids for s, l in SHIFTS for seed in range(seeds)]
    key = lambda a: f"{a[0]}|{a[1]}|{a[2]}|{a[3]}"
    todo = [a for a in J if key(a) not in done]
    print(f"{len(J)} jobs in total, {len(J) - len(todo)} already done, {len(todo)} to run", flush=True)
    for b in range(0, len(todo), 40):
        batch = todo[b:b + 40]
        res = Parallel(n_jobs=nj)(delayed(job)(*a, loader=loader) for a in batch)
        rows = [r for _, rr in res for r in rr]
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        with open(done_path, "a") as fh:
            for a, (status, _) in zip(batch, res):
                if not status.startswith("error"):
                    fh.write(key(a) + "\n")
                else:
                    print("ERROR", key(a), status[:300], flush=True)
        print(f"{min(b + 40, len(todo))}/{len(todo)} done", flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import openml
    os.makedirs(CACHE, exist_ok=True)
    dids = list(openml.study.get_suite(99).data)  # OpenML-CC18
    print(f"Downloading {len(dids)} datasets (cached after the first run)...", flush=True)
    ok = []
    for i, d in enumerate(dids):
        try:
            fetch(d); ok.append(d)
        except Exception as e:
            print(f"  could not download dataset {d}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(dids)} downloaded", flush=True)
    run(ok, args.jobs, args.seeds)

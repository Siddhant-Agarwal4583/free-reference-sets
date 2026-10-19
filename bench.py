"""Benchmark runner.  Usage: python bench.py <experiment> [n_jobs]
Experiments: main | tsweep | nsweep | real | et
Every job writes one row per estimator: estimate, true accuracy of that route's deployed
model, absolute error.  Results are appended to results/<experiment>.csv."""
import sys, os, time, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import make_classification, load_breast_cancer, load_digits, load_wine
from estimators import run_one
from shifts import prep, split_covariate, split_iid, corrupt, label_shift

os.makedirs("results", exist_ok=True)

SHIFTS = [("iid", 0), ("cov", 1.0), ("cov", 2.0), ("cov", 3.0),
          ("corrupt", 0.5), ("corrupt", 1.0), ("label", 0.7)]


def synth_config(cid):
    r = np.random.default_rng(1000 + cid)
    d = int(r.integers(10, 41))
    k = int(r.integers(2, 6))
    inf = max(int(np.ceil(np.log2(k * 2))) + 1, int(r.uniform(0.2, 0.6) * d))
    return dict(n_features=d, n_informative=min(inf, d), n_redundant=int(r.integers(0, max(1, (d - inf) // 2))),
                n_classes=k, n_clusters_per_class=int(r.integers(1, 3)),
                class_sep=float(r.uniform(0.6, 1.6)), flip_y=float(r.uniform(0, 0.08)))


def synth_pool(cid, n=12000):
    cfg = synth_config(cid)
    X, y = make_classification(n_samples=n, random_state=cid, **cfg)
    return prep(X, y), cfg


def real_pool(name):
    loader = {"breast_cancer": load_breast_cancer, "digits": load_digits, "wine": load_wine}[name]
    X, y = loader(return_X_y=True)
    return prep(X, y)


def make_task(X, y, shift, level, rng, n_s, n_t):
    if shift == "cov":
        return split_covariate(X, y, level, rng, n_s=n_s, n_t=n_t)
    Xs, ys, Xt, yt = split_iid(X, y, rng, frac_s=0.5, n_s=n_s, n_t=n_t)
    if shift == "corrupt":
        Xt = corrupt(Xt, level, rng)
    elif shift == "label":
        Xt, yt = label_shift(Xt, yt, level, rng)
    return Xs, ys, Xt, yt


def job(exp, dataset, shift, level, n_s, n_t, T, seed, kind="rf"):
    t0 = time.time()
    if dataset.startswith("syn"):
        (X, y), _ = synth_pool(int(dataset[3:]))
    else:
        X, y = real_pool(dataset)
    rng = np.random.default_rng(seed * 7919 + hash((dataset, shift, level)) % 10007)
    Xs, ys, Xt, yt = make_task(X, y, shift, level, rng, n_s, n_t)
    if len(np.unique(ys)) < 2:
        return []
    rows, diag = run_one(Xs, ys, Xt, yt, kind=kind, T=T, seed=seed)
    out = []
    for est, (v, truth) in rows.items():
        out.append(dict(exp=exp, dataset=dataset, shift=shift, level=level, n_s=len(ys), n_t=len(yt),
                        T=T, kind=kind, seed=seed, estimator=est, estimate=v, truth=truth,
                        abs_err=abs(v - truth), err=v - truth, acc_full=diag["acc_full"],
                        acc_ho_model=diag["acc_holdout_model"], secs=time.time() - t0))
    return out


def jobs_for(exp):
    J = []
    if exp == "main":
        for cid in range(40):
            for sh, lv in SHIFTS:
                for n_s in (200, 1000):
                    for seed in range(3):
                        J.append((exp, f"syn{cid}", sh, lv, n_s, 2000, 200, seed))
    elif exp == "tsweep":
        for cid in range(20):
            for sh, lv in [("iid", 0), ("cov", 2.0), ("corrupt", 1.0)]:
                for T in (25, 50, 100, 200, 500):
                    for seed in range(2):
                        J.append((exp, f"syn{cid}", sh, lv, 500, 2000, T, seed))
    elif exp == "nsweep":
        for cid in range(20):
            for sh, lv in [("iid", 0), ("cov", 2.0), ("corrupt", 1.0)]:
                for n_s in (100, 200, 400, 800, 1600, 3200):
                    for seed in range(2):
                        J.append((exp, f"syn{cid}", sh, lv, n_s, 2000, 200, seed))
    elif exp == "real":
        for ds in ("breast_cancer", "digits", "wine"):
            for sh, lv in SHIFTS:
                for seed in range(20):
                    J.append((exp, ds, sh, lv, None, None, 200, seed))
    elif exp == "et":
        for cid in range(20):
            for sh, lv in SHIFTS:
                for seed in range(2):
                    J.append((exp, f"syn{cid}", sh, lv, 1000, 2000, 200, seed, "et"))
    return J


if __name__ == "__main__":
    exp = sys.argv[1]
    nj = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    J = jobs_for(exp)
    print(exp, len(J), "jobs", flush=True)
    path = f"results/{exp}.csv"
    if os.path.exists(path): os.remove(path)
    t0 = time.time()
    B = 40
    for b in range(0, len(J), B):
        res = Parallel(n_jobs=nj)(delayed(job)(*a) for a in J[b:b + B])
        rows = [r for rr in res for r in rr]
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"{b + len(J[b:b+B])}/{len(J)} done, {time.time() - t0:.0f}s", flush=True)
    print("FINISHED", flush=True)

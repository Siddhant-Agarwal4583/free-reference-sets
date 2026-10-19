"""Finer-grained statistics from results/main.csv used in the paper."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score

df = pd.read_csv("results/main.csv")
EST = ["HO", "HO-DoC", "HO-ATC", "HO-IW", "OOB-X", "OOB-DoC-X", "OOB-ATC-X", "OOB-IW-X"]
out = {}

# MAE by n and shift level
t = df[df.estimator.isin(EST)].groupby(["estimator", "n_s"]).abs_err.mean().mul(100).unstack()
out["mae_by_n"] = t.round(2).to_dict()
t2 = df[df.estimator.isin(EST)].assign(cond=df["shift"] + "@" + df.level.astype(str)) \
    .groupby(["estimator", "cond"]).abs_err.mean().mul(100).unstack()
t2.round(2).to_csv("tables/main_mae_by_level.csv")
out["mae_by_level"] = t2.round(2).to_dict()
print(t.round(2)); print(t2.round(2))

# paired tests within each n
pt = []
for n in sorted(df.n_s.unique()):
    d = df[df.n_s == n]
    W = d.pivot_table(index=["dataset", "shift", "level", "seed"], columns="estimator", values="abs_err")
    for a, b in [("HO", "OOB-X"), ("HO-DoC", "OOB-DoC-X"), ("HO-ATC", "OOB-ATC-X"), ("HO-IW", "OOB-IW-X")]:
        pt.append(dict(n=int(n), base=a, ours=b, mae_base=W[a].mean() * 100, mae_ours=W[b].mean() * 100,
                       win=(W[b] < W[a]).mean(), p=wilcoxon(W[a], W[b]).pvalue))
P = pd.DataFrame(pt); P.to_csv("tables/main_paired_by_n.csv", index=False); print(P.round(4))

# Monitoring: does the estimator flag a real accuracy drop of >5pp relative to the
# in-distribution (iid) accuracy of the same dataset/n/seed?  AUROC of (est drop) vs (true drop>5)
base = df[(df["shift"] == "iid")].set_index(["dataset", "n_s", "seed", "estimator"])
mon = []
for est in EST:
    d = df[(df.estimator == est) & (df["shift"] != "iid")].copy()
    key = list(zip(d.dataset, d.n_s, d.seed, [est] * len(d)))
    b = base.reindex(key)
    d["true_drop"] = b.truth.values - d.truth.values
    d["est_drop"] = b.estimate.values - d.estimate.values
    d = d.dropna(subset=["true_drop", "est_drop"])
    y = (d.true_drop > 0.05).astype(int)
    mon.append(dict(estimator=est, auroc=roc_auc_score(y, d.est_drop), prevalence=y.mean(),
                    spearman=spearmanr(d.true_drop, d.est_drop).correlation))
M = pd.DataFrame(mon); M.to_csv("tables/main_monitoring.csv", index=False); print(M.round(3))
out["monitoring"] = M.round(4).to_dict(orient="records")

# fraction of tasks where estimate within 2pp / 5pp
within = df[df.estimator.isin(EST)].groupby("estimator").abs_err.agg(
    w2=lambda x: (x <= .02).mean(), w5=lambda x: (x <= .05).mean())
within.round(3).to_csv("tables/main_within.csv"); print(within.round(3))
out["within"] = within.round(4).to_dict()
json.dump(out, open("tables/main_extra.json", "w"), indent=1, default=float)

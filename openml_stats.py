"""Dataset-level and breakdown statistics for the OpenML-CC18 experiment."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
from analyze import holm
df = pd.read_csv("results/openml.csv")
PAIRS = [("HO", "OOB-X"), ("HO-DoC", "OOB-DoC-X"), ("HO-ATC", "OOB-ATC-X"), ("HO-IW", "OOB-IW-X")]
XP = [("OOB", "OOB-X"), ("OOB-DoC", "OOB-DoC-X"), ("OOB-ATC", "OOB-ATC-X"), ("OOB-IW", "OOB-IW-X")]
out = {}
# 1) dataset-level: mean abs error per dataset, Wilcoxon over 72 datasets
D = df.groupby(["dataset", "estimator"]).abs_err.mean().unstack()
rows = []
for a, b in PAIRS + XP:
    rows.append(dict(base=a, ours=b, mae_base=D[a].mean()*100, mae_ours=D[b].mean()*100,
                     rel=1-D[b].mean()/D[a].mean(), ds_win=(D[b] < D[a]).mean(),
                     n_win=int((D[b] < D[a]).sum()), p=wilcoxon(D[a], D[b]).pvalue))
R = pd.DataFrame(rows); R["p_holm"] = holm(R.p.values); R.to_csv("tables/openml_dataset_level.csv", index=False)
print("DATASET LEVEL (72 datasets)"); print(R.round(4).to_string(index=False)); out["dataset_level"] = R.to_dict("records")
# 2) by shift: paired tests per shift type (task level) and dataset-level win counts
rows = []
for sh in ["iid", "cov", "corrupt", "label"]:
    d = df[df["shift"] == sh]
    W = d.pivot_table(index=["dataset", "level", "seed"], columns="estimator", values="abs_err")
    Dd = d.groupby(["dataset", "estimator"]).abs_err.mean().unstack()
    for a, b in PAIRS:
        rows.append(dict(shift=sh, base=a, ours=b, mae_base=W[a].mean()*100, mae_ours=W[b].mean()*100,
                         rel=1-W[b].mean()/W[a].mean(), p=wilcoxon(W[a], W[b]).pvalue,
                         ds_wins=int((Dd[b] < Dd[a]).sum())))
S = pd.DataFrame(rows); S["p_holm"] = holm(S.p.values); S.to_csv("tables/openml_by_shift.csv", index=False)
print("\nBY SHIFT"); print(S.round(4).to_string(index=False))
# 3) by dataset size (n_s tercile)
ns = df.groupby("dataset").n_s.median(); bins = pd.cut(ns, [0, 700, 1999, 10**6], labels=["small (n<700)", "medium (700-1999)", "large (n=2000 cap)"])
df["size"] = df.dataset.map(bins)
rows = []
for g, d in df.groupby("size", observed=True):
    W = d.pivot_table(index=["dataset", "shift", "level", "seed"], columns="estimator", values="abs_err")
    u = d.drop_duplicates(["dataset", "shift", "level", "seed"])
    for a, b in PAIRS:
        rows.append(dict(size=g, n_range=f"{int(ns[bins==g].min())}-{int(ns[bins==g].max())}", base=a, ours=b,
                         rel=1-W[b].mean()/W[a].mean(), p=wilcoxon(W[a], W[b]).pvalue,
                         acc_gain_pp=((u.acc_full-u.acc_ho_model)*100).mean()))
Z = pd.DataFrame(rows); Z.to_csv("tables/openml_by_size.csv", index=False); print("\nBY SIZE"); print(Z.round(4).to_string(index=False))
# 4) monitoring AUROC (drop >5pp vs iid of same dataset/seed)
base = df[df["shift"] == "iid"].set_index(["dataset", "seed", "estimator"])
mon = []
for a, b in PAIRS:
    for est in (a, b):
        d = df[(df.estimator == est) & (df["shift"] != "iid")].copy()
        bb = base.reindex(list(zip(d.dataset, d.seed, [est]*len(d))))
        td = bb.truth.values - d.truth.values; ed = bb.estimate.values - d.estimate.values
        y = (td > 0.05).astype(int)
        mon.append(dict(estimator=est, auroc=roc_auc_score(y, ed), prevalence=y.mean(),
                        spearman=spearmanr(td, ed).correlation))
M = pd.DataFrame(mon); M.to_csv("tables/openml_monitoring.csv", index=False); print("\nMONITORING"); print(M.round(3).to_string(index=False))
# 5) within 2/5 pp, accuracy gain, ranks
E = [x for p in PAIRS for x in p]
w = df[df.estimator.isin(E)].groupby("estimator").abs_err.agg(w2=lambda x: (x <= .02).mean(), w5=lambda x: (x <= .05).mean())
print("\nWITHIN"); print(w.round(3))
u = df.drop_duplicates(["dataset", "shift", "level", "seed"]); g = (u.acc_full-u.acc_ho_model)*100
gd = (u.assign(g=g).groupby("dataset").g.mean())
out["accgain"] = dict(task_mean=g.mean(), task_frac_pos=(g > 0).mean(), ds_frac_pos=(gd > 0).mean(),
                      ds_p=wilcoxon(gd).pvalue, n_datasets=len(gd))
print("\nACC GAIN", out["accgain"])
print("\nNUM CLASSES / sizes:", df.drop_duplicates("dataset").n_s.describe().round(0).to_dict())
json.dump(out, open("tables/openml_extra.json", "w"), indent=1, default=float)

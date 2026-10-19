"""Statistical analysis + figures.  python analyze.py [main|real|et|tsweep|nsweep|openml|all]"""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True); os.makedirs("tables", exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

FAMILY = {"shift": {"iid": "IID", "cov": "Covariate", "corrupt": "Corruption", "label": "Label"}}
ORDER = ["AC", "HO", "HO-DoC", "HO-ATC", "HO-IW", "OOB", "OOB-DoC", "OOB-ATC", "OOB-IW",
         "OOB-X", "OOB-DoC-X", "OOB-ATC-X", "OOB-IW-X"]
PAIRS = [("HO", "OOB-X"), ("HO-DoC", "OOB-DoC-X"), ("HO-ATC", "OOB-ATC-X"), ("HO-IW", "OOB-IW-X"),
         ("OOB", "OOB-X"), ("OOB-DoC", "OOB-DoC-X"), ("OOB-ATC", "OOB-ATC-X"), ("OOB-IW", "OOB-IW-X")]


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def cond_key(df):
    return df["shift"] + "_" + df["level"].astype(str)


def mae_table(df, name):
    d = df[df.estimator.isin(ORDER)].copy()
    d["cond"] = d["shift"].map(FAMILY["shift"])
    t = d.groupby(["estimator", "cond"]).abs_err.agg(["mean", "sem"]).mul(100)
    piv = t["mean"].unstack().reindex(ORDER)
    se = t["sem"].unstack().reindex(ORDER)
    allm = d.groupby("estimator").abs_err.agg(["mean", "sem"]).mul(100).reindex(ORDER)
    piv["All"] = allm["mean"]; se["All"] = allm["sem"]
    cols = [c for c in ["IID", "Covariate", "Corruption", "Label", "All"] if c in piv.columns]
    piv, se = piv[cols], se[cols]
    piv.round(2).to_csv(f"tables/{name}_mae.csv")
    # latex with best bolded
    lines = []
    for est in piv.index:
        cells = []
        for c in cols:
            v, s = piv.loc[est, c], se.loc[est, c]
            txt = f"{v:.2f}\\,\\scriptsize{{$\\pm${s:.2f}}}"
            if np.isclose(v, piv[c].min()): txt = "\\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(est.replace("-", "\\mbox{-}") + " & " + " & ".join(cells) + r" \\")
    open(f"tables/{name}_mae.tex", "w").write("\n".join(lines))
    return piv


def bias_table(df, name):
    d = df[df.estimator.isin(ORDER)].copy()
    t = d.groupby(["estimator", "shift"]).err.mean().mul(100).unstack().reindex(ORDER)
    t.round(2).to_csv(f"tables/{name}_bias.csv"); return t


def paired_tests(df, name):
    d = df.copy(); d["task"] = d.dataset + "|" + cond_key(d) + "|" + d.n_s.astype(str) + "|" + d.seed.astype(str) + "|" + d["T"].astype(str)
    W = d.pivot_table(index="task", columns="estimator", values="abs_err")
    out = []
    for a, b in PAIRS:
        if a not in W or b not in W: continue
        x = W[[a, b]].dropna()
        diff = x[a] - x[b]
        p = wilcoxon(x[a], x[b]).pvalue if (diff != 0).any() else 1.0
        out.append(dict(baseline=a, ours=b, n=len(x), mae_base=x[a].mean() * 100, mae_ours=x[b].mean() * 100,
                        rel_reduction=1 - x[b].mean() / x[a].mean(), win_rate=(x[b] < x[a]).mean(),
                        p=p))
    T = pd.DataFrame(out)
    if len(T): T["p_holm"] = holm(T.p.values)
    T.to_csv(f"tables/{name}_paired.csv", index=False)
    # Friedman + average ranks over all estimators
    Wc = W[[c for c in ORDER if c in W]].dropna()
    ranks = pd.DataFrame(np.apply_along_axis(rankdata, 1, Wc.values), columns=Wc.columns).mean()
    fr = friedmanchisquare(*[Wc[c] for c in Wc.columns])
    ranks.sort_values().to_csv(f"tables/{name}_ranks.csv")
    json.dump({"friedman_stat": fr.statistic, "friedman_p": fr.pvalue, "n_tasks": len(Wc)},
              open(f"tables/{name}_friedman.json", "w"), indent=1)
    return T, ranks.sort_values()


def acc_gain(df, name):
    d = df.drop_duplicates(["dataset", "shift", "level", "n_s", "seed", "T"])
    g = (d.acc_full - d.acc_ho_model) * 100
    res = {"mean_gain_pp": g.mean(), "sem": g.sem(), "frac_positive": (g > 0).mean(),
           "p": wilcoxon(d.acc_full, d.acc_ho_model).pvalue}
    if "n_s" in d:
        res["by_n"] = (d.assign(g=g).groupby("n_s").g.mean()).round(3).to_dict() if d.n_s.nunique() < 10 else None
    json.dump(res, open(f"tables/{name}_accgain.json", "w"), indent=1, default=float)
    return res


def analyze(name):
    df = pd.read_csv(f"results/{name}.csv")
    print(f"== {name}: {len(df)} rows, {df.groupby(['dataset','shift','level','n_s','seed','T']).ngroups} tasks")
    piv = mae_table(df, name); print(piv.round(2))
    print(bias_table(df, name).round(2))
    T, ranks = paired_tests(df, name); print(T.round(4)); print(ranks.round(2))
    print(acc_gain(df, name))
    return df


COL = {"HO": "#7f7f7f", "OOB": "#1f77b4", "OOB-X": "#d62728"}


def fig_tsweep():
    df = pd.read_csv("results/tsweep.csv")
    fams = [("plain", ["HO", "OOB", "OOB-X"]), ("DoC", ["HO-DoC", "OOB-DoC", "OOB-DoC-X"]),
            ("ATC", ["HO-ATC", "OOB-ATC", "OOB-ATC-X"]), ("IW", ["HO-IW", "OOB-IW", "OOB-IW-X"])]
    Ts = [25, 50, 100, 200, 500]
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.5), sharex=True)
    for j, (fam, ests) in enumerate(fams):
        for e, key, lab in zip(ests, ["HO", "OOB", "OOB-X"], ["holdout", "OOB naive", "OOB-X (ours)"]):
            d0 = df[(df.estimator == e) & (df["shift"] == "iid")].groupby("T").err.mean() * 100
            d1 = df[(df.estimator == e) & (df["shift"] == "cov")].groupby("T").abs_err.mean() * 100
            ax[0, j].plot(d0.index, d0.values, "o-", ms=3, color=COL[key], label=lab)
            ax[1, j].plot(d1.index, d1.values, "o-", ms=3, color=COL[key])
        ax[0, j].axhline(0, color="k", lw=.5); ax[0, j].set_title(fam)
        for r in range(2):
            ax[r, j].set_xscale("log"); ax[r, j].set_xticks(Ts); ax[r, j].set_xticklabels([str(t) for t in Ts], fontsize=7)
            ax[r, j].minorticks_off()
        ax[1, j].set_xlabel("ensemble size $T$")
    ax[0, 0].set_ylabel("bias, IID (pp)"); ax[1, 0].set_ylabel("MAE, covariate (pp)")
    ax[0, 0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig("figs/tsweep.pdf"); fig.savefig("figs/tsweep.png")


def fig_nsweep():
    df = pd.read_csv("results/nsweep.csv")
    pairs = [("HO", "OOB-X", "plain"), ("HO-DoC", "OOB-DoC-X", "DoC"), ("HO-ATC", "OOB-ATC-X", "ATC"),
             ("HO-IW", "OOB-IW-X", "IW")]
    fig, ax = plt.subplots(1, 5, figsize=(7.2, 1.9))
    for j, (a, b, fam) in enumerate(pairs):
        for e, key in [(a, "HO"), (b, "OOB-X")]:
            g = df[df.estimator == e].groupby("n_s").abs_err.mean() * 100
            ax[j].plot(g.index, g.values, "o-", ms=3, color=COL[key], label="holdout" if key == "HO" else "OOB-X")
        ax[j].set_xscale("log"); ax[j].set_title(fam); ax[j].set_xlabel("$n$ (source)"); ax[j].minorticks_off()
    ax[0].set_ylabel("MAE (pp)"); ax[0].legend(frameon=False, fontsize=7)
    d = df.drop_duplicates(["dataset", "shift", "level", "n_s", "seed"])
    g = ((d.acc_full - d.acc_ho_model) * 100).groupby(d.n_s)
    ax[4].errorbar(g.mean().index, g.mean(), yerr=g.sem(), fmt="o-", ms=3, color="#2ca02c")
    ax[4].axhline(0, color="k", lw=.5); ax[4].set_xscale("log"); ax[4].set_title("accuracy gain")
    ax[4].set_xlabel("$n$ (source)"); ax[4].set_ylabel("pp")
    fig.tight_layout(); fig.savefig("figs/nsweep.pdf"); fig.savefig("figs/nsweep.png")


def fig_mech():
    df = pd.read_csv("results/mech.csv")
    df["gap"] = (df.acc_oob_m - df.acc_target_m) * 100
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for sh, c, lab in [("iid", "#1f77b4", "IID"), ("cov", "#ff7f0e", "covariate")]:
        g = df[df["shift"] == sh].groupby("m").gap.agg(["mean", "sem"])
        ax[0].errorbar(g.index, g["mean"], yerr=g["sem"], fmt="o-", ms=3, color=c, label=lab)
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_xscale("log"); ax[0].minorticks_off()
    ax[0].set_xlabel("sub-ensemble size $m$"); ax[0].set_ylabel("OOB $-$ target acc. (pp)")
    ax[0].set_title("(a) matched-size gap"); ax[0].legend(frameon=False, fontsize=7)
    for k, (cid, sh) in enumerate([(0, "iid"), (3, "cov")]):
        a = ax[k + 1]
        g = df[(df.cid == cid) & (df["shift"] == sh) & (df.m >= 5)]
        a.plot(1 / g.m, g.acc_target_m * 100, "o", ms=3, color="#1f77b4", label="target, size $m$")
        a.plot(1 / g.m, g.acc_oob_m * 100, "s", ms=3, color="#ff7f0e", label="OOB, size $m$")
        for col, cc in [("acc_target_m", "#1f77b4"), ("acc_oob_m", "#ff7f0e")]:
            h = g[(g.m >= 8) & (g.m <= 60)]
            A = np.c_[np.ones(len(h)), 1 / h.m.values]; coef = np.linalg.lstsq(A, h[col].values * 100, rcond=None)[0]
            xx = np.linspace(0, 0.2, 50); a.plot(xx, coef[0] + coef[1] * xx, "-", lw=.8, color=cc)
        a.axhline(g.acc_full500.iloc[0] * 100, color="k", ls="--", lw=.7, label="full $T{=}500$")
        a.set_xlabel("$1/m$"); a.set_xlim(-0.005, 0.21)
        a.set_title(f"({'bc'[k]}) task {cid}, {'IID' if sh=='iid' else 'covariate'}")
    ax[1].set_ylabel("accuracy (%)"); ax[1].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig("figs/mech.pdf"); fig.savefig("figs/mech.png")


def fig_main_bars(name="main"):
    df = pd.read_csv(f"results/{name}.csv")
    fams = [("plain", "HO", "OOB", "OOB-X"), ("DoC", "HO-DoC", "OOB-DoC", "OOB-DoC-X"),
            ("ATC", "HO-ATC", "OOB-ATC", "OOB-ATC-X"), ("IW", "HO-IW", "OOB-IW", "OOB-IW-X")]
    shifts = ["iid", "cov", "corrupt", "label"]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=False)
    for j, sh in enumerate(shifts):
        d = df[df["shift"] == sh]
        x = np.arange(4)
        for i, key in enumerate(["HO", "OOB", "OOB-X"]):
            vals = [d[d.estimator == f[i + 1]].abs_err.mean() * 100 for f in fams]
            ax[j].bar(x + (i - 1) * 0.27, vals, 0.27, color=COL[key],
                      label=["holdout", "OOB naive", "OOB-X (ours)"][i])
        ax[j].set_xticks(x, [f[0] for f in fams], fontsize=7); ax[j].set_title(FAMILY["shift"][sh])
    ax[0].set_ylabel("MAE (pp)"); ax[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(f"figs/{name}_bars.pdf"); fig.savefig(f"figs/{name}_bars.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = ["main", "real", "et", "tsweep", "nsweep"] if which == "all" else [which]
    for n in names:
        if os.path.exists(f"results/{n}.csv"): analyze(n)
    if which in ("all", "tsweep") and os.path.exists("results/tsweep.csv"): fig_tsweep()
    if which in ("all", "nsweep") and os.path.exists("results/nsweep.csv"): fig_nsweep()
    if which in ("all", "mech") and os.path.exists("results/mech.csv"): fig_mech()
    if which in ("all", "main") and os.path.exists("results/main.csv"): fig_main_bars("main")
    if which in ("all", "real") and os.path.exists("results/real.csv"): fig_main_bars("real")


def fig_openml():
    df = pd.read_csv("results/openml.csv")
    D = df.groupby(["dataset", "estimator"]).abs_err.mean().unstack() * 100
    pairs = [("HO", "OOB-X", "plain"), ("HO-DoC", "OOB-DoC-X", "DoC"), ("HO-ATC", "OOB-ATC-X", "ATC"),
             ("HO-IW", "OOB-IW-X", "IW")]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for j, (a, b, fam) in enumerate(pairs):
        x, y = D[a], D[b]
        win = y < x
        ax[j].scatter(x[win], y[win], s=7, color="#d62728", lw=0, label="OOB-X better")
        ax[j].scatter(x[~win], y[~win], s=7, color="#7f7f7f", lw=0, label="holdout better")
        m = max(x.max(), y.max()) * 1.05
        ax[j].plot([0, m], [0, m], "k-", lw=.5); ax[j].set_xlim(0, m); ax[j].set_ylim(0, m)
        ax[j].set_aspect("equal"); ax[j].set_title(f"{fam}: {int(win.sum())}/72", fontsize=8)
        ax[j].set_xlabel("holdout MAE (pp)", fontsize=7); ax[j].tick_params(labelsize=6)
    ax[0].set_ylabel("OOB-X MAE (pp)", fontsize=7); ax[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout(); fig.savefig("figs/openml_scatter.pdf"); fig.savefig("figs/openml_scatter.png")

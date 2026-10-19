"""Mechanism check: (i) does sub-ensemble accuracy follow acc(m) = a + b/m, and
(ii) how large is the OOB vs full-ensemble gap as a function of T?
Uses labels on the target purely for diagnosis."""
import numpy as np, pandas as pd, json
from estimators import make_forest, per_tree_proba, oob_mask, _sizem_source, _sizem_target
from bench import synth_pool, make_task

rows = []
for cid in range(12):
    (X, y), _ = synth_pool(cid)
    for shift, lv in [("iid", 0), ("cov", 2.0)]:
        rng = np.random.default_rng(cid)
        Xs, ys, Xt, yt = make_task(X, y, shift, lv, rng, 500, 2000)
        K = int(max(ys.max(), yt.max()) + 1)
        f = make_forest("rf", 500, cid).fit(Xs, ys)
        Ps, Pt, M = per_tree_proba(f, Xs, K), per_tree_proba(f, Xt, K), oob_mask(f, len(Xs))
        full = (Pt.mean(0).argmax(1) == yt).mean()
        for m in [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 160]:
            at = np.mean([(_sizem_target(Pt, m, rng).argmax(1) == yt).mean() for _ in range(6)])
            src = []
            for _ in range(6):
                ps, keep = _sizem_source(Ps, M, m, rng)
                if keep.sum() > 50: src.append((ps.argmax(1) == ys[keep]).mean())
            rows.append(dict(cid=cid, shift=shift, m=m, acc_target_m=at,
                             acc_oob_m=np.mean(src) if src else np.nan, acc_full500=full))
df = pd.DataFrame(rows)
df.to_csv("results/mech.csv", index=False)

# goodness of fit of a + b/m on m in [m_lo, m_hi] and extrapolation error to T=500
fits = []
for (cid, sh), g in df.groupby(["cid", "shift"]):
    g = g.dropna()
    for lo, hi in [(8, 60), (20, 160)]:
        h = g[(g.m >= lo) & (g.m <= hi)]
        A = np.c_[np.ones(len(h)), 1 / h.m.values]
        for col in ["acc_target_m", "acc_oob_m"]:
            c, res, *_ = np.linalg.lstsq(A, h[col].values, rcond=None)
            pred = A @ c
            r2 = 1 - np.sum((h[col] - pred) ** 2) / max(np.sum((h[col] - h[col].mean()) ** 2), 1e-12)
            fits.append(dict(cid=cid, shift=sh, window=f"{lo}-{hi}", series=col, r2=r2,
                             extrap500=c[0] + c[1] / 500, raw_at_hi=h[col].values[-1],
                             full500=g.acc_full500.iloc[0]))
F = pd.DataFrame(fits)
F.to_csv("results/mech_fits.csv", index=False)
t = F[F.series == "acc_target_m"]
summary = {
    "median_R2_target": float(t.r2.median()),
    "frac_R2_gt_0.9": float((t.r2 > 0.9).mean()),
    "MAE_extrap_vs_full_pp": float((t.extrap500 - t.full500).abs().mean() * 100),
    "MAE_raw_at_hi_vs_full_pp": float((t.raw_at_hi - t.full500).abs().mean() * 100),
}
print(json.dumps(summary, indent=1))
json.dump(summary, open("results/mech_summary.json", "w"), indent=1)

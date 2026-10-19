"""Label-free accuracy estimation under distribution shift for bagged ensembles.

Core idea: a bagged ensemble's out-of-bag (OOB) predictions are a free, labeled,
held-out-like sample from the source distribution.  Every reference-set based
estimator (DoC, ATC, importance weighting) can therefore be run without
sacrificing training data.  A subtlety: an OOB prediction is made by a random
sub-ensemble (~e^-1 of the trees), whereas target predictions are made by the full
ensemble, so confidence scores on the two sides are not exchangeable.  We correct
this with *size-matched (Bernoulli-masked) scoring* of target points.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------- utils
def per_tree_proba(forest, X, n_classes):
    """(T, n, K) array of per-tree class probabilities aligned to forest.classes_."""
    T = len(forest.estimators_)
    out = np.zeros((T, X.shape[0], n_classes), dtype=np.float32)
    for t, est in enumerate(forest.estimators_):
        p = est.predict_proba(X)
        # trees store classes_ as indices into forest.classes_ (all classes seen in bootstrap
        # are not guaranteed); map columns safely
        cls = est.classes_.astype(int)
        out[t][:, cls] = p
    return out


def oob_mask(forest, n):
    T = len(forest.estimators_)
    M = np.ones((T, n), dtype=bool)
    for t, idx in enumerate(forest.estimators_samples_):
        M[t, idx] = False
    return M


_JRNG = np.random.default_rng(777)


def jit(c):
    """Randomized tie-breaking: tree-vote confidences are discrete (multiples of 1/m)."""
    return c + 1e-7 * _JRNG.random(np.shape(c))


def atc_threshold(conf, correct):
    """ATC-MC threshold t: fraction of source scores above t equals source accuracy."""
    acc = correct.mean()
    # choose t as the (1-acc) quantile of conf
    return np.quantile(jit(conf), 1.0 - acc, method="linear")


def domain_weights(Xs, Xt, seed, clip=20.0, oob=True):
    """Density ratio w(x)=p_t(x)/p_s(x) on source points via a domain classifier.
    Probabilities for source points are OOB (no cross-fitting split needed)."""
    X = np.vstack([Xs, Xt])
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, oob_score=True,
                                 n_jobs=1, random_state=seed).fit(X, d)
    p = clf.oob_decision_function_[: len(Xs), 1]
    p = np.nan_to_num(p, nan=0.5)
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = (p / (1 - p)) * (len(Xs) / len(Xt))
    w = np.minimum(w, clip)
    return w / w.mean()


# ----------------------------------------------------------------------------- main
def oob_estimates(forest, Xs, ys, Xt, R=20, seed=0, n_classes=None, want_iw=True,
                  extrapolate=True):
    """All OOB-based estimates of the full ensemble's accuracy on unlabeled Xt."""
    rng = np.random.default_rng(seed)
    K = n_classes
    Ps = per_tree_proba(forest, Xs, K)
    Pt = per_tree_proba(forest, Xt, K)
    M = oob_mask(forest, len(Xs))
    cnt = M.sum(0)
    ok = cnt > 0
    p_oob = np.einsum("tn,tnk->nk", M.astype(np.float32), Ps) / np.maximum(cnt, 1)[:, None]
    p_oob, ys_ok = p_oob[ok], ys[ok]
    conf_s = p_oob.max(1)
    corr_s = (p_oob.argmax(1) == ys_ok).astype(float)
    acc_oob = corr_s.mean()

    p_full = Pt.mean(0)
    conf_t_full = p_full.max(1)

    # size-matched scoring: each tree kept w.p. q (empirical OOB inclusion rate)
    q = M.mean()
    T = Pt.shape[0]
    conf_t_m = np.zeros((R, len(Xt)))
    for r in range(R):
        B = rng.random((T, len(Xt))) < q
        B[rng.integers(0, T, len(Xt)), np.arange(len(Xt))] = True  # ensure >=1 tree
        pm = np.einsum("tn,tnk->nk", B.astype(np.float32), Pt) / B.sum(0)[:, None]
        conf_t_m[r] = pm.max(1)

    thr = atc_threshold(conf_s, corr_s)
    est = {
        "OOB": acc_oob,
        "AC": conf_t_full.mean(),
        "OOB-DoC": acc_oob - (conf_s.mean() - conf_t_full.mean()),
        "OOB-DoC-M": acc_oob - (conf_s.mean() - conf_t_m.mean()),
        "OOB-ATC": (jit(conf_t_full) > thr).mean(),
        "OOB-ATC-M": (jit(conf_t_m) > thr).mean(),
    }
    w = domain_weights(Xs, Xt, seed) if want_iw else None
    if want_iw:
        wk = w[ok]
        est["OOB-IW"] = float(np.sum(wk * corr_s) / np.sum(wk))
    if extrapolate:
        ex, grid = oob_extrapolated(forest, Xs, ys, Xt, K, seed=seed, w=w, Ps=Ps, Pt=Pt, M=M)
        est.update(ex)
    diag = {"q": float(q), "mean_conf_oob": float(conf_s.mean()),
            "mean_conf_full_t": float(conf_t_full.mean()),
            "mean_conf_matched_t": float(conf_t_m.mean())}
    return est, diag


def _sizem_source(Ps, M, m, rng):
    """Per source point: average m randomly chosen OOB trees. Returns probs and kept idx."""
    T, n, K = Ps.shape
    u = rng.random((T, n))
    u[~M] = np.inf
    order = np.argsort(u, axis=0)[:m]          # (m, n) tree indices
    keep = M.sum(0) >= m
    p = Ps[order, np.arange(n)[None, :], :].mean(0)
    return p[keep], keep


def _sizem_target(Pt, m, rng):
    T, n, K = Pt.shape
    order = np.argsort(rng.random((T, n)), axis=0)[:m]
    return Pt[order, np.arange(n)[None, :], :].mean(0)


def oob_extrapolated(forest, Xs, ys, Xt, n_classes, seed=0, R=8, n_grid=5, w=None,
                     Ps=None, Pt=None, M=None):
    """Size-matched estimators computed on a grid of ensemble sizes m and extrapolated
    to the full size T with est(m) = a + b/m (O(1/m) convergence of randomized ensembles).
    Returns dict of estimates for {OOB, DoC, ATC, IW} with suffix -X."""
    rng = np.random.default_rng(seed + 12345)
    if Ps is None: Ps = per_tree_proba(forest, Xs, n_classes)
    if Pt is None: Pt = per_tree_proba(forest, Xt, n_classes)
    if M is None: M = oob_mask(forest, len(Xs))
    T = Ps.shape[0]
    cnt = M.sum(0)
    m_max = max(1, int(np.quantile(cnt, 0.10)))
    grid = np.unique(np.round(np.linspace(max(1, m_max / 4), m_max, n_grid)).astype(int))
    fams = {"OOB-X": [], "OOB-DoC-X": [], "OOB-ATC-X": [], "OOB-IW-X": []}
    for m in grid:
        vals = {k: [] for k in fams}
        for r in range(R):
            ps, keep = _sizem_source(Ps, M, m, rng)
            ys_k = ys[keep]
            conf_s = ps.max(1); corr_s = (ps.argmax(1) == ys_k).astype(float)
            pt = _sizem_target(Pt, m, rng); conf_t = pt.max(1)
            acc = corr_s.mean()
            thr = atc_threshold(conf_s, corr_s)
            vals["OOB-X"].append(acc)
            vals["OOB-DoC-X"].append(acc - (conf_s.mean() - conf_t.mean()))
            vals["OOB-ATC-X"].append((jit(conf_t) > thr).mean())
            if w is not None:
                wk = w[keep]
                vals["OOB-IW-X"].append(np.sum(wk * corr_s) / np.sum(wk))
        for k in fams:
            fams[k].append(np.mean(vals[k]) if vals[k] else np.nan)
    out = {}
    A = np.c_[np.ones(len(grid)), 1.0 / grid]
    for k, v in fams.items():
        v = np.asarray(v)
        if np.isnan(v).any():
            continue
        if len(grid) >= 2:
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            out[k] = float(np.clip(coef[0] + coef[1] / T, 0, 1))
        else:
            out[k] = float(v[-1])
    return out, grid


def holdout_estimates(model, Xh, yh, Xt, seed=0, want_iw=True):
    """Classic reference-set estimators using a labeled held-out split."""
    ph = model.predict_proba(Xh)
    pt = model.predict_proba(Xt)
    cls = model.classes_
    conf_h, conf_t = ph.max(1), pt.max(1)
    corr_h = (cls[ph.argmax(1)] == yh).astype(float)
    acc_h = corr_h.mean()
    thr = atc_threshold(conf_h, corr_h)
    est = {
        "HO": acc_h,
        "HO-DoC": acc_h - (conf_h.mean() - conf_t.mean()),
        "HO-ATC": (jit(conf_t) > thr).mean(),
    }
    if want_iw:
        w = domain_weights(Xh, Xt, seed)
        est["HO-IW"] = float(np.sum(w * corr_h) / np.sum(w))
    return est


def make_forest(kind, T, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=T, n_jobs=1, random_state=seed)
    if kind == "et":
        return ExtraTreesClassifier(n_estimators=T, bootstrap=True, n_jobs=1, random_state=seed)
    raise ValueError(kind)


def run_one(Xs, ys, Xt, yt, kind="rf", T=200, seed=0, holdout_frac=0.2, want_iw=True):
    """Fit OOB-route model on all of S and holdout-route model on (1-f) of S, return
    estimates and errors against each route's own true target accuracy."""
    K = int(max(ys.max(), yt.max()) + 1)
    f_all = make_forest(kind, T, seed).fit(Xs, ys)
    acc_all = (f_all.predict(Xt) == yt).mean()
    e_oob, diag = oob_estimates(f_all, Xs, ys, Xt, seed=seed, n_classes=K, want_iw=want_iw)

    strat = ys if np.bincount(ys).min() >= 2 else None
    Xtr, Xh, ytr, yh = train_test_split(Xs, ys, test_size=holdout_frac, random_state=seed,
                                        stratify=strat)
    f_ho = make_forest(kind, T, seed).fit(Xtr, ytr)
    acc_ho = (f_ho.predict(Xt) == yt).mean()
    e_ho = holdout_estimates(f_ho, Xh, yh, Xt, seed=seed, want_iw=want_iw)

    rows = {}
    for k, v in e_oob.items():
        rows[k] = (float(v), float(acc_all))
    for k, v in e_ho.items():
        rows[k] = (float(v), float(acc_ho))
    return rows, {"acc_full": float(acc_all), "acc_holdout_model": float(acc_ho), **diag}

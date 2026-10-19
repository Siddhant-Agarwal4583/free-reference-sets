"""Controlled distribution shifts applied to a labeled pool (X, y)."""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def prep(X, y):
    X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    y = LabelEncoder().fit_transform(y)
    return X, y


def split_covariate(X, y, alpha, rng, n_s=None, n_t=None, direction="pc1"):
    """Subpopulation covariate shift: points are routed to source with prob
    sigmoid(-alpha*z) and to target otherwise, z = standardized projection.
    alpha=0 gives an iid split.  p(y|x) is unchanged."""
    if direction == "pc1":
        z = PCA(n_components=1, random_state=0).fit_transform(X)[:, 0]
    else:
        v = rng.normal(size=X.shape[1]); v /= np.linalg.norm(v)
        z = X @ v
    z = (z - z.mean()) / (z.std() + 1e-12)
    to_s = rng.random(len(X)) < _sigmoid(-alpha * z)
    S, Tg = np.where(to_s)[0], np.where(~to_s)[0]
    if n_s is not None and len(S) > n_s: S = rng.choice(S, n_s, replace=False)
    if n_t is not None and len(Tg) > n_t: Tg = rng.choice(Tg, n_t, replace=False)
    return X[S], y[S], X[Tg], y[Tg]


def split_iid(X, y, rng, frac_s=0.5, n_s=None, n_t=None):
    idx = rng.permutation(len(X)); k = int(frac_s * len(X))
    S, Tg = idx[:k], idx[k:]
    if n_s is not None: S = S[:n_s]
    if n_t is not None: Tg = Tg[:n_t]
    return X[S], y[S], X[Tg], y[Tg]


def corrupt(Xt, sigma, rng, kind="gauss"):
    """Covariate corruption of the target (support shift, analogous to CIFAR-C)."""
    if kind == "gauss":
        return Xt + sigma * rng.normal(size=Xt.shape)
    if kind == "dropout":  # feature values replaced by the mean (0 after scaling)
        m = rng.random(Xt.shape) < sigma
        Xc = Xt.copy(); Xc[m] = 0.0
        return Xc
    raise ValueError(kind)


def label_shift(Xt, yt, strength, rng):
    """Resample target so class priors follow a Dirichlet-perturbed distribution.
    strength in [0,1]: 0 = original priors, 1 = fully Dirichlet(1) priors."""
    K = yt.max() + 1
    p0 = np.bincount(yt, minlength=K) / len(yt)
    pd = rng.dirichlet(np.ones(K))
    p = (1 - strength) * p0 + strength * pd
    n = len(yt)
    idx = []
    for k in range(K):
        pool = np.where(yt == k)[0]
        m = int(round(p[k] * n))
        if len(pool) == 0 or m == 0: continue
        idx.append(rng.choice(pool, m, replace=True))
    idx = np.concatenate(idx)
    return Xt[idx], yt[idx]

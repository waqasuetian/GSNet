# data/scripts/features_extended.py
import numpy as np
from scipy.stats import skew, kurtosis

def time_features(x):
    """
    x: (N, L) node-by-time for a 1s window (e.g., 200 samples).
    Returns: (N, K_time)
    """
    N, L = x.shape
    dx = np.diff(x, axis=1)
    eps = 1e-12

    line_len = np.sum(np.abs(dx), axis=1, keepdims=True)
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True))
    var = np.var(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    zcr = ((x[:, 1:] * x[:, :-1]) < 0).mean(axis=1, keepdims=True)

    # Teager–Kaiser energy
    psi = x[:, 1:-1]**2 - x[:, :-2] * x[:, 2:]
    tkeo = np.mean(np.abs(psi), axis=1, keepdims=True)

    # Hjorth parameters
    activity = var
    mobility = np.sqrt(np.var(dx, axis=1, keepdims=True) / (var + eps))
    complexity = (
        np.sqrt(np.var(np.diff(dx, axis=1), axis=1, keepdims=True) / (np.var(dx, axis=1, keepdims=True) + eps))
        / (mobility + eps)
    )

    sk = skew(x, axis=1).reshape(-1, 1)
    ku = kurtosis(x, axis=1, fisher=True).reshape(-1, 1)

    return np.concatenate(
        [line_len, rms, var, std, zcr, tkeo, activity, mobility, complexity, sk, ku], axis=1
    )

def spectral_shapes(log_bins, fs=200):
    """
    log_bins: (N, F) log amplitude (F≈100 after dropping DC).
    Returns: (N, 6) = [centroid, spread, SEF90, slope, offset, flatness]
    """
    N, F = log_bins.shape
    p = np.exp(log_bins)
    p = np.maximum(p, 1e-12)
    freqs = np.linspace(1, F, F)[None, :]  # approx 1..F Hz

    # Centroid & spread
    centroid = (p * freqs).sum(axis=1, keepdims=True) / p.sum(axis=1, keepdims=True)
    spread = np.sqrt(((freqs - centroid) ** 2 * p).sum(axis=1, keepdims=True) / p.sum(axis=1, keepdims=True))

    # Spectral edge frequency (90%)
    cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    sef_idx = (cdf >= 0.90).argmax(axis=1).reshape(-1, 1) + 1

    # 1/f slope & offset (log-power vs log-freq)
    f_safe = np.maximum(np.arange(1, F + 1), 1)
    X = np.vstack([np.log(f_safe), np.ones(F)]).T  # (F,2)
    coef = np.linalg.lstsq(X, log_bins.T, rcond=None)[0]  # (2,N)
    slope = coef[0, :].reshape(-1, 1)
    offset = coef[1, :].reshape(-1, 1)

    # Spectral flatness = geometric mean / arithmetic mean
    geo = np.exp(np.mean(log_bins, axis=1, keepdims=True))
    arith = np.mean(p, axis=1, keepdims=True)
    flat = geo / np.maximum(arith, 1e-12)

    return np.concatenate([centroid, spread, sef_idx.astype(float), slope, offset, flat], axis=1)

def permutation_entropy(x, m=3, delay=1):
    """
    x: (N, L), returns (N, 1) normalized permutation entropy.
    """
    N, L = x.shape
    if L < m * delay:
        return np.zeros((N, 1), dtype=float)
    out = np.zeros((N, 1), dtype=float)
    for i in range(N):
        cnt = {}
        for j in range(L - (m - 1) * delay):
            window = x[i, j:(j + m * delay):delay]
            key = tuple(np.argsort(window))
            cnt[key] = cnt.get(key, 0) + 1
        p = np.array(list(cnt.values()), dtype=float)
        p = p / p.sum()
        out[i, 0] = -np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(np.math.factorial(m))
    return out

def node_coherence(log_bins):
    """
    Approximate node-node coherence by cosine similarity in spectral (log-bin) space.
    Input: (N, F) log amplitude. Output: (N, N) in [0,1], diagonal=0.
    """
    X = (log_bins - log_bins.mean(axis=1, keepdims=True))
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    W = X @ X.T
    W = (W + 1.0) / 2.0  # [-1,1] -> [0,1]
    np.fill_diagonal(W, 0.0)
    return W

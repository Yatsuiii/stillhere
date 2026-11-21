"""Cheap baselines: word rate and the sound envelope, as HRF-convolved regressors read at each volume. Frozen with core.py."""
import numpy as np
from scipy import stats

FS = 10  # the convolution grid, samples per second


def spm_hrf(dt: float, length: float = 32.0) -> np.ndarray:
    """SPM's canonical double-gamma HRF sampled every dt seconds (peak 6 s, undershoot 16 s, ratio 1/6), summing to 1."""
    t = np.arange(0, length, dt)
    h = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def word_rate_per_second(onsets, n_sec: int) -> np.ndarray:
    """Words starting in each whole second of the stimulus."""
    on = np.asarray(onsets, float)
    on = on[(on >= 0) & (on < n_sec)]
    return np.bincount(on.astype(int), minlength=n_sec).astype(float)


def envelope_per_second(x, sr: int) -> np.ndarray:
    """Mean absolute amplitude of a mono (or channel-mean) signal in each whole second."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(1)
    m = np.abs(x)
    n = int(len(m) // sr)
    return m[: n * sr].reshape(n, sr).mean(1)


def hrf_regressor(x_per_s, n_vols: int, tr: float, lag: float = 0.0, start: float = 0.0) -> np.ndarray:
    """x_per_s[k] covers stimulus second [k, k+1). Convolved with the HRF on a 0.1 s grid and read at stimulus time
    start + j * tr - lag for volume j (lag > 0 reads an earlier stimulus time)."""
    xf = np.repeat(np.asarray(x_per_s, float), FS)
    r = np.convolve(xf, spm_hrf(1.0 / FS))[: len(xf)]
    t = start + np.arange(n_vols) * tr - lag
    return np.interp(t, np.arange(len(xf)) / FS + 0.5 / FS, r, left=r[0], right=r[-1])

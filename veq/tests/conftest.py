import numpy as np
import pytest


def smooth(rng, shape, width=6):
    """Slow random series: white noise smoothed by a moving average, z-scored along the last axis."""
    x = rng.standard_normal(shape)
    k = np.ones(width) / width
    x = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), -1, x)
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


@pytest.fixture(scope="session")
def world():
    """A synthetic story: n people, p areas, T volumes. Each person's BOLD is a shared story signal S plus some word rate W and sound
    envelope E, plus that person's own noise. The answer to every question the scorer asks is known by construction."""
    rng = np.random.default_rng(7)
    n, p, T = 12, 4, 300
    S = smooth(rng, (p, T))
    W = smooth(rng, (T,), 10)
    E = smooth(rng, (T,), 3)
    bold = {f"sub-{i:02d}": S + 0.5 * W + 0.3 * E + 3.0 * smooth(rng, (p, T)) for i in range(n)}
    return {"rng": rng, "S": S, "W": W, "E": E, "bold": bold, "baselines": {"word_rate": W, "envelope": E}, "n": n, "p": p, "T": T}

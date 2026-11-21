"""veq: how many real volunteers is a brain-encoding model worth, per person, beside cheap baselines."""
from .core import clean, gauge, score
from .regressors import envelope_per_second, hrf_regressor, word_rate_per_second

__all__ = ["score", "clean", "gauge", "hrf_regressor", "word_rate_per_second", "envelope_per_second"]
__version__ = "0.1.0"

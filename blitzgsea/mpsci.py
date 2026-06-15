# this is directly copied from the fantastic project https://github.com/WarrenWeckesser/mpsci

import math

from mpmath import mp, exp, log
from scipy.stats import gamma, norm


def gammacdf(x: float, k: float, theta: float, dps: int = 100) -> float:
    """
    Gamma distribution CDF using mpmath for high-precision computation.
    k is the shape parameter, theta is the scale parameter.
    """
    mp.dps = dps
    with mp.extradps(mp.dps):
        x = mp.mpf(x)
        if x < 0:
            return mp.zero
        return mp.gammainc(k, 0, x / theta, regularized=True)


def gammacdf1(x: float, k: float, theta: float) -> float:
    log_sf_value = gamma.logsf(x, k, scale=theta)
    sf_value = mp.exp(log_sf_value)
    return 1 - sf_value


def gammacdf2(x: float, k: float, theta: float) -> float:
    log_sf_value = mp.log(1 - mp.gammainc(k, a=x / theta, regularized=True))
    sf_value = exp(log_sf_value)
    return mp.mpf(1) - sf_value


def invcdf_old(p: float, mu: float = 0, sigma: float = 1) -> float:
    """Normal distribution inverse CDF (quantile function)."""
    if math.isnan(p):
        p = 1
    p = min(max(p, 0), 1)
    with mp.extradps(mp.dps):
        mu = mp.mpf(mu)
        sigma = mp.mpf(sigma)
        try:
            a = mp.erfinv(2 * p - 1)
            x = mp.sqrt(2) * sigma * a + mu
        except Exception:
            print("The problem value is: ", p)
            quit()
        return x


def invcdf(p: float, mu: float = 0, sigma: float = 1) -> float:
    orig_p = p
    if p > 0.5:
        p = 1 - p
    p = float(p)
    if math.isnan(p):
        p = 1
    p = min(max(p, 0), 1)
    n = norm.isf(p)
    if orig_p > 0.5:
        return -n
    return n

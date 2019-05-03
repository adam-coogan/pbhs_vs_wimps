import numpy as np
from scipy.optimize import minimize

from src.constants import n_u_0
from src.distributions import Distribution_N_gamma


@np.vectorize
def point_source_limit(m_dm,
                       m_pbh,
                       n_pbh,
                       f,
                       merger_rate_prior="LF",
                       alpha=0.95):
    """Computes the point source constraint on <sigma v>.

    Parameters
    ----------
    m_dm : float or numpy.array
        DM mass.
    m_pbh : float
        PBH mass.
    n_pbh : int
        Number of PBH detections by LIGO O3, ET or SKA.
    merger_rate_prior : str
        Prior on merger rate (for GW scenarios) or event rate (for SKA).
    alpha : float
        Level for upper limit. Must be between 0 and 1.
    f_percentile : float
        f will be set to this percentile for the corresponding p(f|n_pbh)
        distribution. Must be between 0 and 1.

    Returns
    -------
    numpy.array
        The upper limit on <sigma v> at the alpha level.
    """
    dist_n_gamma = Distribution_N_gamma(m_pbh)
    p_value = 1 - alpha

    @np.vectorize
    def func(log10_sv):  # computes Pr(N_gamma >= 19) = 1 - Pr(N_gamma < 19)
        cdf_val = 1 - dist_n_gamma.cdf(n_u_0 - 1, 10**log10_sv, f, m_dm).sum()
        return (cdf_val - p_value)**2

    # Find good bracketing interval for the root finder
    log10_svs = np.linspace(-44, -23, 100)
    idx_a, idx_b = np.where(func(log10_svs) / p_value**2 < 0.9999)[0][[0, -1]]

    res = minimize(
        func,
        x0=np.mean(log10_svs[[idx_a, idx_b]]),
        bounds=[[log10_svs[idx_a], log10_svs[idx_b]]])
    assert res.success
    return 10**res.x[0]

import numpy as np
from constants import fs_0, n_u_0, flux_type_0, flux_thresh_0, b_cut_0
from constants import n_mw_pbhs
from scipy.integrate import quad, dblquad, trapz
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import root_scalar
from scipy.stats import binom, norm, poisson
from scipy import special
from pbhhalosim import PBHHaloSim

post_f_dir = "data/posteriors_f/"
post_sv_dir = "data/posteriors_sv/"
p_gamma_dir = "data/p_gammas/"
ligo_masses = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_p_f_gw(m_pbh, n_pbh, post_f_dir=post_f_dir):
    """Loads p(f_PBH | N_PBH) for gravitational wave detectors.

    Parameters
    ----------
    m_pbh : float
        PBH mass
    n_pbh : int
        The number of detections via gravitational waves.

    Returns
    -------
    p_f : float, int -> float
    """
    if m_pbh == 10:  # Einstein telescope
        f_pts, p_f_1_pts, p_f_10_pts, p_f_100_pts = np.loadtxt(
            "%sPosterior_f_ET_M=%.1f.txt" % (post_f_dir, m_pbh)).T
    elif m_pbh in ligo_masses:  # LIGO 03
        f_pts, p_f_1_pts, p_f_10_pts, p_f_100_pts = np.loadtxt(
            "%sPosterior_f_M=%.1f.txt" % (post_f_dir, m_pbh)).T
    else:
        raise ValueError("Invalid PBH mass")

    def p_f(f, n_pbh):
        if n_pbh == 1:
            p_f_pts = p_f_1_pts
        elif n_pbh == 10:
            p_f_pts = p_f_10_pts
        elif n_pbh == 100:
            p_f_pts = p_f_100_pts
        else:
            raise ValueError("Invalid number of GW detections")
        return np.interp(f, f_pts, p_f_pts)

    return p_f


def p_sv(sv, log_uniform=False):
    """p(<sigma v>), the prior on <sigma v>.

    Parameters
    ----------
    log_uniform : bool
        If true, uses a log-uniform prior. Otherwise, uses a flat prior.
    """
    def _p_sv(sv):
        if log_uniform:
            return 1 / sv
        else:
            return 1
    return np.vectorize(_p_sv)(sv)


def get_p_gamma_val(m_pbh, m_dm, sv, fs=fs_0, flux_type=flux_type_0, b_cut=b_cut_0,
                    flux_thresh=flux_thresh_0, n_samples=50000):
    """Computes p_gamma(M_pbh, m_dm, <sigma v>), the probability that a PBH
    passes the Fermi point source selection cuts.

    To-do
    -----
    Vectorize this over m_dm.
    """
    def _get_p_gamma(m_dm, sv):
        sim = PBHHaloSim(mass_dist=m_pbh, f_pbh=1, m_dm=m_dm, sv=sv,
                         flux_type=flux_type, b_cut=b_cut,
                         flux_thresh=flux_thresh,
                         n_samples=n_samples)
        sim.run()
        return sim.pr_det

    return np.vectorize(_get_p_gamma)(m_dm, sv)


def save_p_gamma_table(m_pbh, m_dms, svs, fs=fs_0, flux_type=flux_type_0, b_cut=b_cut_0,
                       flux_thresh=flux_thresh_0, n_samples=50000):
    """Generates a table containing p_gamma.

    Parameters
    ----------
    m_pbh : float
    m_dms : np.array
        Must contain more than one element.
    svs : np.array
        Must contain more than one element.
    """
    if m_dms.size <= 1:
        raise ValueError("m_dms must have more than one element")
    if svs.size <= 1:
        raise ValueError("svs must have more than one element")
    m_dm_col = np.repeat(m_dms, svs.size)
    sv_col = np.tile(svs, m_dms.size)
    # Compute the table values
    p_gamma_vals = get_p_gamma_val(m_pbh, m_dm_col, sv_col, fs=fs_0, flux_type=flux_type_0,
                                   b_cut=b_cut_0, flux_thresh=flux_thresh_0, n_samples=50000)
    # Save the data table
    p_gamma_path = "%sp_gamma_M=%.1f.csv" % (p_gamma_dir, m_pbh)
    p_gamma_tab = np.stack([m_dm_col, sv_col, p_gamma_vals]).T
    np.savetxt(p_gamma_path, p_gamma_tab,
               header=("p_gamma for M_PBH = %.1f M_sun.\n"
                       "Columns are: m_DM (GeV), <sigma v> (cm^3/s), p_gamma.") % m_pbh)


def load_p_gamma(m_pbh):
    """Loads an interpolator for p_gamma(<sigma v>, M_PBH, m_DM).

    Returns
    -------
    A vectorized function mapping (m_DM, <sigma v>) to p_gamma.
    """
    m_dm_col, sv_col, p_gamma_col = np.loadtxt("%sp_gamma_M=%.1f.csv" % (p_gamma_dir, m_pbh)).T
    m_dms = np.unique(m_dm_col)
    svs = np.unique(sv_col)
    p_gammas = p_gamma_col.reshape([m_dms.size, svs.size])
    p_gamma_rg = RegularGridInterpolator((m_dms, svs), p_gammas)

    def p_gamma(m_dm, sv):
        # Wrap the interpolator, since its interface is horrible
        m_dm = np.asarray(m_dm)
        sv = np.asarray(sv)

        if m_dm.size > 1 and sv.size > 1:
            return p_gamma_rg(np.array([m_dm, sv]).T)
        elif m_dm.size > 1:
            return p_gamma_rg(np.array([m_dm, sv*np.ones_like(m_dm)]).T)
        elif sv.size > 1:
            return p_gamma_rg(np.array([m_dm*np.ones_like(sv), sv]).T)
        else:
            return p_gamma_rg(np.array([m_dm, sv]))

    return m_dms, svs, p_gamma


def p_n_gamma(n_gamma, sv, f, p_gamma, m_pbh, m_dm):
    """p(N_gamma | <sigma v>, f), the probability of detecting N_gamma PBH
    halos as gamma-ray point sources.

    Parameters
    ----------
    n_gamma : int
    sv : float
    f : float
    p_gamma : float, float -> float
        Probability for a PBH to pass the gamma-ray point source cuts, as a
        function of m_DM and <sigma v>.
    m_pbh : float
    """
    return binom.pmf(n_gamma, n=np.floor(n_mw_pbhs(f, m_pbh)), p=p_gamma(m_dm, sv))


def p_u(n_gamma, n_u, prior="jeffreys"):
    """p(N_U | N_gamma), the probability of having a point source catalogue of
    size N_U given N_gamma PBHs passing the gamma-ray point source cuts.

    Parameters
    ----------
    n_u : int
    n_gamma : int
    prior : "jeffreys", "uniform"
        Specifies either a Jeffreys' prior (lambda^{-1/2}) or uniform prior for
        lambda.
    """
    if prior not in ["jeffreys", "uniform"]:
        raise ValueError("Invalid prior on lambda")

    def _p_u(n_gamma):
        if n_u > n_gamma:
            if prior == "jeffreys":
                n_a = n_u - n_gamma
                return special.gamma(n_a + 1/2) / special.gamma(n_a + 1)
            elif prior == "uniform":
                return 1
        else:
            return 0

    return np.vectorize(_p_u)(n_gamma)


def posterior_integrand(sv, n_gamma, f, n_pbh, p_f, p_gamma, m_pbh, m_dm, n_u=n_u_0):
    """Computes the value of the integrand/summand in the expression for
    p(<sigma v> | N_PBH, N_U, M_PBH, m_DM).

    Parameters
    ----------
    sv : float
    n_gamma : int
    f : float
    n_pbh : int
    n_u : int
    p_f : float, int -> float
        p(f_PBH | N_PBH)
    p_gamma : float, float -> float
        p_gamma as a function of m_DM and <sigma v>.
    """
    return p_sv(sv) * p_f(f, n_pbh) * p_n_gamma(n_gamma, sv, f, p_gamma, m_pbh, m_dm) * p_u(n_gamma, n_u)


def get_posterior_val(sv, n_pbh, p_f, p_gamma, m_pbh, m_dm, n_u=n_u_0):
    """Computes the posterior for <sigma v>. Supports broadcasting over sv and
    m_dm. See documentation for `posterior_integrand`.
    """
    def get_posterior_val_(m_dm, sv):
        post_val = 0
        for n_gamma in np.arange(0, n_u+1, 1):
            post_val += quad(
                lambda f: posterior_integrand(sv, n_gamma, f, n_pbh, p_f, p_gamma, m_pbh, m_dm, n_u),
                1e-6, 1, epsabs=1e-99)[0]
        return post_val

    return np.vectorize(get_posterior_val_)(m_dm, sv)


def save_posterior_table(svs, n_pbh, p_f, p_gamma, m_pbh, m_dms, n_u=n_u_0):
    """Generates a table containing p_gamma.

    Parameters
    ----------
    svs : np.array
        Must contain more than one element.
    m_dms : np.array
        Must contain more than one element.
    """
    if svs.size <= 1:
        raise ValueError("svs must have more than one element")
    if m_dms.size <= 1:
        raise ValueError("m_dms must have more than one element")
    m_dm_col = np.repeat(m_dms, svs.size)
    sv_col = np.tile(svs, m_dms.size)
    # Compute the table values
    post_vals = get_posterior_val(sv_col, n_pbh, p_f, p_gamma, m_pbh, m_dm_col, n_u)
    # Save the data table
    post_path = "%sposterior_sv_M=%.1f_N=%i.csv" % (post_sv_dir, m_pbh, n_pbh)
    post_tab = np.stack([m_dm_col, sv_col, post_vals]).T
    np.savetxt(post_path, post_tab,
               header=("p(<sigma v> | N_PBH, M_PBH, m_DM, U) for M_PBH = %.1f M_sun.\n"
                       "Columns are: m_DM (GeV), <sigma v> (cm^3/s), p(sv | ...).") % m_pbh)


def load_posterior(m_pbh, n_pbh):
    """Loads a table of posterior values for <sigma v>.

    Returns
    -------
    m_dms, svs, post_vals
        post_vals is defined so that:
            post_vals[i, j] = posterior(m_dms[i], svs[j]).
    """
    m_dm_col, sv_col, post_col = np.loadtxt("%sposterior_sv_M=%.1f_N=%i.csv" %
                                            (post_sv_dir, m_pbh, n_pbh)).T
    m_dms = np.unique(m_dm_col)
    svs = np.unique(sv_col)
    post_vals = post_col.reshape([m_dms.size, svs.size])
    return m_dms, svs, post_vals


def credible_interval(posterior, alpha, x_max, x_guess):
    """Computes the credible interval for p(x). At the level alpha, this is
    x_alpha such that
        int_0^{x_alpha} dx p(x) = alpha.

    Parameters
    ----------
    posterior : float -> float
    alpha : float
        Level of the interval.
    x_max : float
        Largest possible value for x.
    x_guess : float
        Point at which to start the root finder.

    Returns
    -------
    x_alpha : float
    """
    def objective(x):
        return quad(posterior, 0, x, epsabs=1e-99, limit=100)[0] - alpha

    return root_scalar(objective, bracket=[0, x_max], x0=x_guess, xtol=1e-99).root

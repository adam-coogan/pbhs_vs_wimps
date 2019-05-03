import numpy as np
from scipy.integrate import trapz, quad
from scipy.stats import chi2

from src.constants import kpc_to_cm, exp_tau
from src.constants import hubble, dnde_interps, speed_of_light, rho_dm_avg_0
from src.constants import e_egb, err_high_egb, r_s_mw, rho_einasto
from src.constants import d_earth, pbh_ann_rate, fs_0


"""
Functions for computing diffuse galactic and Extragalactic fluxes from PBHs.
"""


# Integral of density along line-of-sight away from the galactic center.
rho_los_int = quad(rho_einasto, d_earth, 1000*r_s_mw)[0]


def phi_gal(e, m_dm, sv, m_pbh, f, fs=fs_0):
    """Diffuse gamma-ray flux from PBHs along line-of-sight away from the
    galactic center.

    Parameters
    ----------
    e : float
        Gamma-ray energy.
    m_dm : float
        DM mass.
    sv : float
        Self-annihilation cross section.
    m_pbh : float
        PBH mass.
    f : float
        Relative PBH abundance.
    fs : str
        DM annihilation final state.

    Returns
    -------
    float
        phi, (GeV cm^2 sr s)^{-1}.
    """
    return (f * pbh_ann_rate(m_dm, sv, m_pbh) / m_pbh * rho_los_int /
            (4 * np.pi) * dnde_interps[fs](e, m_dm).flatten()) / (kpc_to_cm)**2


def phi_ex(e, m_dm, sv, m_pbh, f, fs=fs_0):
    """Extragalactic flux from PBHs, obtained by integrating over redshifts.

    Parameters
    ----------
    e : float
        Gamma-ray energy.
    m_dm : float
        DM mass.
    sv : float
        Self-annihilation cross section.
    m_pbh : float
        PBH mass.
    f : float
        Relative PBH abundance.
    fs : str
        DM annihilation final state.

    Returns
    -------
    float
        phi, (GeV cm^2 sr s)^{-1}.
    """
    @np.vectorize
    def _phi_ex(e):
        def integrand(zp):
            return (exp_tau(e, zp)[0] / hubble(zp) *
                    dnde_interps[fs]((1 + zp) * e, m_dm))

        # Much faster than quad
        zps = np.geomspace(0.01, 1e3, 1000)
        integral = trapz(integrand(zps), zps)

        # 10^3 converts Mpc in 1/H from integral to kpc
        return (speed_of_light * f * pbh_ann_rate(m_dm, sv, m_pbh) /
                m_pbh * rho_dm_avg_0 / kpc_to_cm**2 * 1e3 * integral /
                (4 * np.pi))

    return _phi_ex(e)


def phi_diff(e, m_dm, sv, m_pbh, f, fs=fs_0):
    """Total isotropic gamma-ray flux from DM around PBHs.

    Parameters
    ----------
    e : float
        Gamma-ray energy.
    m_dm : float
        DM mass.
    sv : float
        Self-annihilation cross section.
    m_pbh : float
        PBH mass.
    f : float
        Relative PBH abundance.
    fs : str
        DM annihilation final state.

    Returns
    -------
    float
        phi, (GeV cm^2 sr s)^{-1}.
    """
    return (phi_gal(e, m_dm, sv, m_pbh, f, fs) +
            phi_ex(e, m_dm, sv, m_pbh, f, fs))


@np.vectorize
def diffuse_limit(m_dm,
                  m_pbh,
                  n_pbh,
                  f,
                  merger_rate_prior="LF",
                  alpha=0.95):
    """Computes the diffuse constraint on <sigma v>.

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
    # Critical value for chi2
    chi2_crit = chi2.ppf(alpha, len(e_egb))

    # Compute chi2 for a reference cross section
    sv_ref = 3e-26
    chi2_ref = np.sum(
        (phi_ex(e_egb, m_dm, sv_ref, m_pbh, f) / err_high_egb)**2)

    # Since chi2 ~ <sigma v>**(2/3) the limit can be computed analytically
    return sv_ref * (chi2_crit / chi2_ref)**(3 / 2)

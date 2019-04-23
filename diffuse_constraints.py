import numpy as np
from scipy.integrate import trapz, quad
from scipy.optimize import root_scalar
from constants import kpc_to_cm, exp_tau
from constants import hubble, dnde_interps, speed_of_light, rho_dm_avg_0
from constants import e_egb, phi_egb, err_high_egb, r_s_mw, rho_einasto
from constants import d_earth, pbh_ann_rate, fs_0


"""
Functions for computing diffuse flux from PBHs and a simple version of the
resulting constraint on <sigma v>.
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
def sv_bound_diff(m_dm, m_pbh, f, n_sigma=3, source="ex", fs=fs_0):
    """Computes a quick-and-dirty bound on <sigma v> by requiring the diffuse
    flux from PBHs to exceed the measured value by no more than n_sigma
    standard deviations.

    Parameters
    ----------
    m_dm : float
        DM mass.
    m_pbh : float
        PBH mass.
    f : float
        Relative PBH abundance.
    n_sigma : float
        Significance required for PBH signal to be considered significant.
    source : str
        Which contributions of PBHs to the diffuse flux to consider: "ex" for
        extragalactic only, "gal" for galactic only or "gal+ex" for both.
    fs : str
        DM annihilation final state.

    Returns
    -------
    float
        Bound on <sigma v>.
    """
    def objective(log10_sv):
        sv = 10**log10_sv
        if source == "ex":
            phi_dm = phi_ex(e_egb, m_dm, sv, m_pbh, f, fs)
        elif source == "gal":
            phi_dm = phi_gal(e_egb, m_dm, sv, m_pbh, f, fs)
        elif source == "gal+ex":
            phi_dm = phi_diff(e_egb, m_dm, sv, m_pbh, f, fs)
        return np.max((phi_dm - phi_egb) / err_high_egb) - n_sigma

    sol = root_scalar(
        objective, bracket=[-50, -10], x0=-26, xtol=1e-200, rtol=1e-8)
    assert sol.converged
    return 10**sol.root

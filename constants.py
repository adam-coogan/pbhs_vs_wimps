import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import stats
from scipy.interpolate import interp1d, interp2d

"""
Constants and utility functions.
"""

colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

kpc_to_cm = 3.086e21  # 1 kpc in cm
cm_to_kpc = 1 / kpc_to_cm
GeV_to_m_sun = 1 / 1.11543e57
m_sun_to_GeV = 1 / GeV_to_m_sun
GeV_to_erg = 0.001602
erg_to_GeV = 1 / GeV_to_erg
yr_to_s = 365. * 24 * 60**2
s_to_yr = 1 / yr_to_s
L_sun_to_GeV_s = 2.402e36
GeV_s_to_L_sun = 1 / L_sun_to_GeV_s
L_sun_to_erg_s = L_sun_to_GeV_s * GeV_to_erg
erg_s_to_L_sun = 1 / L_sun_to_erg_s
speed_of_light = 299792.458  # km / s

# Best-fit Einasto parameters for Milky Way halo. From PPPC.
alpha_mw = 0.17
r_s_mw = 28.44  # kpc
rho_e_mw = 0.033 * GeV_to_m_sun / cm_to_kpc**3  # M_sun / kpc^3
Omega_cdm = 0.2589  # from Planck 2015
Omega_m_0 = 0.3
Omega_Lambda_0 = 0.7
rho_dm_avg_0 = 1.15e-6 * GeV_to_m_sun / cm_to_kpc**3  # avg cosmological DM density
z_eq = 3500.
z_final = 30.  # redshift at which to end PBH halo evolution
d_earth = 8.33  # kpc
age_of_universe = 13e9  # yr
fermi_psf = 0.15 * np.pi / 180.  # arxiv:0902.1089
fermi_psf_solid_angle = 2.*np.pi*(1. - np.cos(fermi_psf))
# Particle physics
alpha_em = 1/137.
m_e = 0.511e-3  # GeV
m_mu = 105.7e-3  # GeV
m_tau = 1.776  # GeV
# Default detectability cuts and number of point sources.
# From arxiv:1610.07587, page 12.
fs_0 = "b"
n_u_0 = 19
flux_type_0 = "dnde"
flux_thresh_0 = 7e-10  # cm^-2 s^-1
b_cut_0 = 20  # deg
h_hubble = 0.7  # H_0 = h 100 km/s / Mpc

def hubble(z):  # km/s / Mpc
    return (100 * h_hubble) * np.sqrt(Omega_m_0 * (1 + z)**3 + Omega_Lambda_0)

def load_exp_tau_interp():
    e_et, zp_et, exp_tau_tab = np.loadtxt("data/exptau.csv").T
    e_et = np.array(sorted(list(set(e_et))))
    zp_et = np.array(sorted(list(set(zp_et))))
    exp_tau_tab = exp_tau_tab.reshape([len(e_et), len(zp_et)])
    return interp2d(e_et, zp_et, exp_tau_tab.T)

def load_spec_interps():
    # Load data as a structured array
    data = np.genfromtxt("data/dN_dlog10x_gammas.dat", dtype=float, delimiter=" ", names=True)
    fss = data.dtype.names[2:]
    m_dm_tab = data["mDM"]
    m_dms = np.array(sorted(list(set(m_dm_tab))))
    log10x_tab = data["Log10x"]
    log10xs = np.array(sorted(list(set(log10x_tab))))
    dnde_interps = {}

    for fs in fss:
        dn_dlog10xs = data[fs].reshape([len(m_dms), len(log10xs)])
        interpolator = interp2d(log10xs, m_dms, dn_dlog10xs, fill_value=0.)
        dnde_interps[fs] = lambda e, m_dm: interpolator(np.log10(e / m_dm), m_dm).flatten() / (np.log(10) * e)

    return dnde_interps

def load_int_spec_interps(e_low=1, e_high=None):
    """Loads interpolators for integrated photon spectra for different final states,
    using data from PPPC4DMID.

    Parameters
    ----------
    e_low, e_high : float
        Energy range over which to compute integrated fluxes. If `e_high` is `None`,
        it will be set to the DM mass.

    Returns
    -------
    `dict`s of `interp1d`s for (int de dn/de) and (int de e dn/de) as a function of
    DM mass; a list of final states in the data file.
    """
    # Load data as a structured array
    data = np.genfromtxt("data/dN_dlog10x_gammas.dat", dtype=float, delimiter=' ', names=True)
    fss = data.dtype.names[2:]
    log10x_tab = data["Log10x"]
    m_dm_tab = data["mDM"]
    m_dms = np.array(sorted(list(set(m_dm_tab))))
    if e_high is None:
        e_high = np.max(m_dms)
    int_dnde_interps = {}
    int_e_dnde_interps = {}

    for fs in fss:
        dn_dlog10x_tab = data[fs]
        int_dndes = []
        int_e_dndes = []

        for m_dm in m_dms:
            # Select table entries for DM mass
            dm_idxs = np.where(m_dm_tab == m_dm)[0]
            es = m_dm * 10**log10x_tab[dm_idxs]
            dnde = dn_dlog10x_tab[dm_idxs] / (np.log(10) * es)
            # Select energies in [e_low, e_high]
            e_idxs = np.where((e_low <= es) & (es <= e_high))[0]
            es = es[e_idxs]
            dnde = dnde[e_idxs]
            # Compute integrals
            int_dndes.append(integrate.trapz(dnde, es))
            int_e_dndes.append(integrate.trapz(es*dnde, es))

        int_dndes = np.array(int_dndes)
        int_e_dndes = np.array(int_e_dndes)
        # Construct interpolators
        int_dnde_interps[fs] = interp1d(m_dms, int_dndes)
        int_e_dnde_interps[fs] = interp1d(m_dms, int_e_dndes)

    return int_dnde_interps, int_e_dnde_interps, fss

int_dnde_interps, int_e_dnde_interps, fss = load_int_spec_interps()

def to_galactic_coords(r, th, phi, deg=True):
    """Converts from spherical coordinates centered on the GC to galactic coordinates.

    Notes
    -----
    Assumes Earth is located at (d_earth, 0, 0).
    """
    x = r * np.sin(th) * np.cos(phi)
    y = r * np.sin(th) * np.sin(phi)
    z = r * np.cos(th)

    d = np.sqrt((x - d_earth)**2 + y**2 + z**2)
    b = np.pi/2 - np.arccos(z / d)
    l = np.arctan2(y, x - d_earth)

    if deg:
        return np.stack([d, 180/np.pi*b, 180/np.pi*l])
    else:
        return np.stack([d, b, l])

def rho_einasto(r, rho_e=rho_e_mw, r_s=r_s_mw, alpha=alpha_mw):
    """Einasto density profile, M_sun/kpc^3."""
    return rho_e * np.exp(-2/alpha * ((r / r_s)**alpha - 1))

# Total MW DM mass in M_sun. Cross-checked with PPPC.
m_mw_dm = integrate.quad(lambda r: 4*np.pi*r**2*rho_einasto(r, rho_e_mw, r_s_mw, alpha_mw), 0., np.inf, epsabs=0, epsrel=1e-4)[0]

def n_mw_pbhs(f, m_pbh):
    """Returns the number of PBHs in the Milky Way DM halo.
    """
    return f * m_mw_dm / m_pbh

def gamma_tr_sample(x_min, x_max, shape=1, scale=1):
    """Use inverse CDF sampling to sample from a truncated gamma distribution.
    """
    distro = stats.gamma(a=shape, loc=0, scale=scale)
    size = np.max([x_min.size, x_max.size])
    cdf_min = distro.cdf(x_min)
    cdf_max = distro.cdf(x_max)
    samples = distro.ppf(cdf_min + np.random.uniform(size=size) * (cdf_max - cdf_min))
    return samples

I100 = 1.48e-7 * 1e3 # GeV^-1 cm^-2 s^-1 sr^-1
gamma_fermi = 2.31
e_cut = 362.  # GeV

def phi_g_egb_fermi(e):
    """Fermi extragalactic gamma ray background flux.

    Returns
    -------
    float
        EGB in GeV^-1 cm^-2 s^-1 sr^-1.
    """
    return I100 * (e / 0.1)**(-gamma_fermi) * np.exp(-e / e_cut)

def mantissa_exp(x):
    exp = np.floor(np.log10(x))
    return x/10**exp, exp

def sci_fmt(val):
    m, e = mantissa_exp(val)
    if e == 0:
        return "{:g}".format(m)
    else:
        e_str = "{:g}".format(e)
        if m == 1:
            return r"10^{" + e_str + "}"
        else:
            m_str = "{:g}".format(m)
            return (r"{" + m_str + r"} \times 10^{" + e_str + "}")

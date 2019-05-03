import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import trapz, cumtrapz
from scipy.stats import binom

from constants import (fs_0, flux_type_0, flux_thresh_0, b_cut_0, n_mw_pbhs)
from pbhhalosim import PBHHaloSim

# Directory containing tables for p(f | m_pbh, n_pbh)
post_f_dir = "data/posteriors_f/"
# Directory for p_gamma tables
p_gamma_dir = "data/p_gammas/"
# Masses for LIGO O3
ligo_masses = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Range of fs for which p(f | m_pbh, n_pbh) was computed
log10_f_min, log10_f_max = -6, 0
f_min, f_max = 1e-6, 1


class Distribution_f:
    """
    Represents p(f|m_pbh, n_pbh).

    Examples
    --------
    This class is a wrapper around data tables. It is straightforward to
    initialize and evaluate:

    >>> p_f = Distribution_f(m_pbh=10, n_pbh=1, merger_rate_prior="LF")
    >>> p_f(1e-4)
    array(80.79149236)

    When attributes are reset, the relevant tables will be reloaded if
    possible:

    >>> p_f.m_pbh = 0.5
    >>> p_f(1e-4)
    array(4.69992612)
    >>> p_f.m_pbh = 0.01
    ValueError: p(f|n_pbh) has not been calculated for this PBH mass
    """

    def __init__(self, m_pbh, n_pbh, merger_rate_prior):
        """Initializer. In our paper we only consider one experiment for each
        PBH mass, so this uniquely determines which experiment to load the f
        distribution for.

        Parameters
        ----------
        m_pbh : float
            PBH mass.
        n_pbh : float
            Number of PBH detections
        merger_rate_prior : str
            Prior on R: either "LF" (log-flat, the conservative choice) or "J"
            (Jeffrey's).
        """
        self.m_pbh = m_pbh
        self.n_pbh = n_pbh
        self.merger_rate_prior = merger_rate_prior

    def _load_p_f_gw(self):
        """Loads p(f|m_pbh, n_pbh) into an interpolator.
        """
        if (self.m_pbh == 10):
            experiment = "ET"
        elif (self.m_pbh == 100):
            experiment = "SKA"
        elif self.m_pbh in ligo_masses:
            experiment = "O3"

        try:
            fs, p_fs = np.loadtxt(
                "{}Posterior_f_{}_Prior_{}_M={:.1f}_N={}.txt".format(
                    post_f_dir, experiment, self.merger_rate_prior, self.m_pbh,
                    self.n_pbh)).T
        except OSError:
            raise Exception("p(f|m_pbh, n_pbh) table not found. Make sure it "
                            "was computed for this value of n_pbh.")

        # Normalization constant
        norm = trapz(p_fs, fs)

        self._pdf_interp = interp1d(
            fs, p_fs / norm, bounds_error=False, fill_value=0.)
        cdf_vals = np.insert(cumtrapz(p_fs / norm, fs), 0, 0.)
        self._cdf_interp = interp1d(
            fs, cdf_vals, bounds_error=False, fill_value="extrapolate")
        # Inverse CDF
        self._ppf_interp = interp1d(
            cdf_vals, fs, bounds_error=False, fill_value="extrapolate")

    def _pdf(self, f):
        """Evaluates p(f|m_pbh, n_pbh).

        Parameters
        ----------
        f : float
            Relative PBH abundance.

        Returns
        -------
        float
            Posterior probability for f.
        """
        if self._pdf_interp is None:
            self._load_p_f_gw()
        return self._pdf_interp(f)

    def cdf(self, f):
        """Computes the CDF.

        Parameters
        ----------
        f : float
            Relative PBH abundance.

        Returns
        -------
        float
            CDF up to f.
        """
        if self._cdf_interp is None:
            self._load_p_f_gw()
        return self._cdf_interp(f)

    def ppf(self, f):
        """Computes the the percentile point function (ie, quantile function or
        inverse CDF).

        Parameters
        ----------
        f : float
            Relative PBH abundance.

        Returns
        -------
        float
            CDF up to f.
        """
        if self._ppf_interp is None:
            self._load_p_f_gw()
        return self._ppf_interp(f)

    @property
    def merger_rate_prior(self):
        """str : prior on merger rate."""
        return self._merger_rate_prior

    @merger_rate_prior.setter
    def merger_rate_prior(self, val):
        if val not in ["LF", "J"]:
            raise ValueError("Invalid merger rate prior")
        else:
            self._merger_rate_prior = val
            self._pdf_interp = None
            self._cdf_interp = None
            self._ppf_interp = None

    @property
    def m_pbh(self):
        """float : PBH mass."""
        return self._m_pbh

    @m_pbh.setter
    def m_pbh(self, val):
        if val not in ligo_masses + [10, 100]:
            raise ValueError("p(f|n_pbh) has not been calculated for this PBH "
                             "mass")
        else:
            self._m_pbh = val
            self._pdf_interp = None
            self._cdf_interp = None
            self._ppf_interp = None

    @property
    def n_pbh(self):
        """float : number of PBH detections."""
        return self._n_pbh

    @n_pbh.setter
    def n_pbh(self, val):
        self._n_pbh = val
        self._pdf_interp = None
        self._cdf_interp = None
        self._ppf_interp = None


class Distribution_N_gamma:
    """
    Represents p(n_gamma|m_pbh, f, m_dm, sv). Provides functions to save
    tables containing the distribution's parameters, which are computed using
    PBHHaloSim.

    Examples
    --------
    This class can compute the p_gamma table for a given PBH mass and grid of
    DM masses and cross sections. The example below uses the default analysis
    cuts and 1000 MC samples to compute p_gamma for each point in the grid.
    Since test is set to True, the results are written to a file in the
    `p_gammas/test/` directory.

    >>> m_dms = np.geomspace(1e1, 1e4, 3)
    >>> svs = np.geomspace(1e-44, 1e-23, 4)
    >>> dist_n_gamma = Distribution_N_gamma(m_pbh=0.5, n_samples=1000,
    ...                                     test=True)
    >>> dist_n_gamma.save_p_gamma_table(m_dms, svs)

    Existing p_gamma tables will be loaded when other functions requiring them
    are called. For example, we can get the probability that an individual PBH
    passes the detection cuts and error in the MC estimate:

    >>> sv, m_dm = 1e-35, 100.
    >>> dist_n_gamma = Distribution_N_gamma(m_pbh=0.5)
    >>> dist_n_gamma.p_gamma(sv, m_dm), dist_n_gamma.p_gamma_err(sv, m_dm)
    (array([5.44540144e-11]), array([6.62457273e-13]))
    """

    def __init__(self,
                 m_pbh,
                 fs=fs_0,
                 flux_type=flux_type_0,
                 b_cut=b_cut_0,
                 flux_thresh=flux_thresh_0,
                 n_samples=50000,
                 test=False):
        """Initializer.

        Parameters
        ----------
        m_pbh : float
            PBH mass.
        fs : str
            Annihilation final state.
        flux_type : str
            Use "dnde" to compute fluxes or "e dnde" to compute energy fluxes.
        b_cut : float
            Specifies the latitude (|b|) cut for calculating the number of
            detectable PBH halos.
        flux_thresh : float
            Flux detectability threshold in cm^-2 s^-1 (for flux_type "dnde")
            or erg cm^-2 s^-1 (for flux_type "e dnde").
        n_samples : int
            Number of iterations of the simulation to run.
        test : bool
            Setting this to True reads and writes all data tables to test/
            subdirectories. This is useful when worried about overwriting large
            tables that took a long time to compute.
        """
        self.m_pbh = m_pbh
        self.fs = fs
        self.flux_type = flux_type
        self.b_cut = b_cut
        self.flux_thresh = flux_thresh
        self.n_samples = n_samples
        self.test = test

    def _get_p_gamma_val(self, m_dm, sv):
        def helper(m_dm, sv):
            """Computes p_gamma, the parameter for the binomial distribution
            p(n_gamma|m_dm, sv, m_pbh, f), by running the PBH point source
            simulation.

            Returns
            -------
            MC estimate of p_gamma and its MC error.
            """
            sim = PBHHaloSim(
                m_pbh=self.m_pbh,
                f=1,  # doesn't affect p_gamma
                m_dm=m_dm,
                sv=sv,
                fs=self.fs,
                flux_type=self.flux_type,
                b_cut=self.b_cut,
                flux_thresh=self.flux_thresh,
                n_samples=self.n_samples)
            sim.run()
            return sim.pr_det, sim.pr_det_err

        return np.vectorize(helper)(m_dm, sv)

    def save_p_gamma_table(self, m_dms, svs):
        """Generates a table containing p_gamma.

        Parameters
        ----------
        m_dms : np.array
            Must contain more than one element.
        svs : np.array
            Must contain more than one element.
        """
        if m_dms.size <= 1:
            raise ValueError("m_dms must have more than one element")
        if svs.size <= 1:
            raise ValueError("svs must have more than one element")

        # Compute the table values
        sv_col = np.repeat(svs, m_dms.size)
        m_dm_col = np.tile(m_dms, svs.size)
        p_gammas, p_gamma_errs = self._get_p_gamma_val(m_dm_col, sv_col)

        # Save the data table
        fname = "{}{}p_gamma_M={:.1f}.csv".format(
            p_gamma_dir, "test/" if self.test else "", self.m_pbh)
        np.savetxt(fname,
                   np.stack([sv_col, m_dm_col, p_gammas, p_gamma_errs]).T,
                   header=("p_gamma for M_PBH = {:.1f} M_sun.\n"
                           "Columns are: <sigma v> (cm^3/s), m_DM (GeV), "
                           "p_gamma, MC error.").format(self.m_pbh))

    def _load_p_gamma_table(self):
        """Loads interpolators mapping (m_dm, sv) to p_gamma and its error.
        """
        try:
            fname = "{}{}p_gamma_M={:.1f}.csv".format(
                p_gamma_dir, "test/" if self.test else "", self.m_pbh)
            sv_col, m_dm_col, p_gamma_col, p_gamma_err_col = np.loadtxt(
                fname).T
        except OSError:
            raise Exception("p_gamma table not found. Construct it with "
                            "`save_p_gamma_table()`.")
        # Package tables into RegularGridInterpolators
        self.svs = np.unique(sv_col)
        self.m_dms = np.unique(m_dm_col)
        p_gammas = p_gamma_col.reshape([self.svs.size, self.m_dms.size])
        p_gamma_errs = p_gamma_err_col.reshape(
            [self.svs.size, self.m_dms.size])
        p_gamma_rg = RegularGridInterpolator((self.svs, self.m_dms), p_gammas)
        p_gamma_err_rg = RegularGridInterpolator((self.svs, self.m_dms),
                                                 p_gamma_errs)

        # Vectorize the RegularGridInterpolators
        @np.vectorize
        def p_gamma(sv, m_dm):
            return p_gamma_rg([sv, m_dm])

        @np.vectorize
        def p_gamma_err(sv, m_dm):
            return p_gamma_err_rg([sv, m_dm])

        self._p_gamma = p_gamma
        self._p_gamma_err = p_gamma_err

    def pmf(self, n_gamma, sv, f, m_dm):
        """Evaluates the PMF.

        Parameters
        ----------
        n_gamma : int
            Number of PBHs detected as point sources by Fermi.
        sv : float or np.array
            DM self-annihilation cross section.
        f : float
            Relative PBH abundance.
        m_dm : float or np.array
            DM mass.

        Returns
        -------
        np.array
            p(n_gamma|m_pbh, f, m_dm, sv)
        """
        if self._p_gamma is None:
            self._load_p_gamma_table()
        # Maximum value of n_gamma
        self.b = np.floor(n_mw_pbhs(f, self.m_pbh))
        return binom.pmf(n_gamma, n=self.b, p=self.p_gamma(sv, m_dm))

    def cdf(self, n_gamma, sv, f, m_dm):
        """Evaluates the CDF by summing over all values of n_gamma up to and
        including the specified value.

        Parameters
        ----------
        n_gamma : int
            Number of PBHs detected as point sources by Fermi.
        sv : float or np.array
            DM self-annihilation cross section.
        f : float
            Relative PBH abundance.
        m_dm : float or np.array
            DM mass.

        Returns
        -------
        np.array
            CDF up to n_gamma
        """
        if self._p_gamma is None:
            self._load_p_gamma_table()

        # Maximum value of n_gamma
        self.b = np.floor(n_mw_pbhs(f, self.m_pbh))
        n_gammas = np.arange(0, n_gamma + 1)
        return binom.pmf(
            n_gammas,
            n=int(np.floor(n_mw_pbhs(f, self.m_pbh))),
            p=self.p_gamma(sv, m_dm)).sum()

    def p_gamma(self, sv, m_dm):
        if self._p_gamma is None:
            self._load_p_gamma_table()
        return self._p_gamma(sv, m_dm)

    def p_gamma_err(self, sv, m_dm):
        if self._p_gamma_err is None:
            self._load_p_gamma_table()
        return self._p_gamma_err(sv, m_dm)

    @property
    def m_pbh(self):
        return self._m_pbh

    @m_pbh.setter
    def m_pbh(self, val):
        self._m_pbh = val
        # Dump the current tables
        self._p_gamma = None
        self._p_gamma_err = None

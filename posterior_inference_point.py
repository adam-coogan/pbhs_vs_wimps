import numpy as np
from constants import fs_0, flux_type_0, flux_thresh_0, b_cut_0, n_u_0
from constants import n_mw_pbhs
from scipy.integrate import quad, trapz
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import binom
from scipy import special
from pbhhalosim import PBHHaloSim
from posterior_inference_shared import f_min, f_max, Posterior

"""
Classes for performing posterior analysis with point source constraints.
"""

# Directory for p_gamma tables
p_gamma_dir = "data/p_gammas/"


class Distribution_N_gamma:
    """
    Represents p(n_gamma|m_pbh, f, m_dm, sv). Provides functions to save
    tables containing the distribution's parameters, which are computed using
    PBHHaloSim.
    """

    def __init__(self,
                 m_pbh,
                 fs=fs_0,
                 flux_type=flux_type_0,
                 b_cut=b_cut_0,
                 flux_thresh=flux_thresh_0,
                 n_samples=50000,
                 test=True):
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
            sv_col, m_dm_col, p_gamma_col, p_gamma_err_col = np.loadtxt(
                "{}p_gamma_M={:.1f}.csv".format(p_gamma_dir, self.m_pbh)).T
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

    def __call__(self, n_gamma, sv, f, m_dm):
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
        return binom.pmf(n_gamma, n=np.floor(n_mw_pbhs(f, self.m_pbh)),
                         p=self.p_gamma(sv, m_dm))

    # def cdf(self, n_gamma, sv, f, m_dm):
    #     if self._p_gamma is None:
    #         self._load_p_gamma_table()
    #     print(n_gamma, int(np.floor(n_mw_pbhs(f, self.m_pbh))),
    #           self.p_gamma(sv, m_dm))
    #     return binom.pmf(
    #         np.arange(0, n_gamma + 1),
    #         n=int(np.floor(n_mw_pbhs(f, self.m_pbh))),
    #         p=self.p_gamma(sv, m_dm)).sum()

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


class Distribution_U:
    """
    Represents the unassociated source count probability p(n_u|n_gamma).
    """

    def __init__(self, lambda_prior="LF"):
        """
        Parameters
        ----------
        prior : "LF", "U"
            Specifies the prior for lambda, the astrophysical fake rate.
            Defaults to the conservative choice, "LF". The optimistic case is
            "U", and "J" is in-between.
        """
        self.lambda_prior = lambda_prior

    def __call__(self, n_gamma, n_u):
        """Evaluates the PDF.

        Parameters
        ----------
        n_gamma : int
            Number of PBHs detectable as gamma-ray sources.
        n_u : int
            Number of observed unassociated point sources.

        Returns
        -------
        np.array
            p(n_u|n_gamma).
        """
        def helper(n_gamma, n_u):
            if n_u > n_gamma:
                if self.lambda_prior == "LF":
                    return 1. / (n_u - n_gamma)
                elif self.lambda_prior == "J":
                    return (special.gamma(0.5 + n_u - n_gamma) /
                            special.gamma(1. + n_u - n_gamma))
                elif self.lambda_prior == "U":
                    return 1.
            else:
                return 0

        return np.vectorize(helper)(n_gamma, n_u)

    @property
    def lambda_prior(self):
        return self._lambda_prior

    @lambda_prior.setter
    def lambda_prior(self, val):
        if val not in ["LF", "J", "U"]:
            raise ValueError("Invalid prior on lambda")
        else:
            self._lambda_prior = val


class PointSourcePosterior(Posterior):
    """
    Class for performing posterior analysis with point-source constraints. See
    the submission scripts and posterior_analysis_tutorial.ipynb for examples.
    """

    def __init__(self,
                 m_pbh,
                 n_pbh,
                 n_u=n_u_0,
                 merger_rate_prior="LF",
                 lambda_prior="LF",
                 sv_prior="U",
                 fs=fs_0,
                 test=True):
        """Initializer.

        Parameters
        ----------
        m_pbh : float
            PBH mass.
        n_pbh : float
            Number of PBH detections
        n_u : int
            Number of observed unassociated point sources.
        merger_rate_prior : str
            Prior on R: either "LF" (log-flat, the conservative choice) or "J"
            (Jeffrey's).
        sv_prior: str
            Determines which prior to use for <sigma v>: "U" (uniform, the
            conservative choice) or "LF" (log-flat).
        fs : str
            DM annihilation final state.
        test : bool
            Setting this to True reads and writes all data tables to test/
            subdirectories. This is useful when worried about overwriting large
            tables that took a long time to compute.
        """
        # Subclasses whose properties need to be synchronized with this
        # object's must be instantiated before calling the superclass
        # initializer
        self._p_n_gamma = Distribution_N_gamma(m_pbh, test=test)
        self._p_u = Distribution_U(lambda_prior)

        super().__init__(m_pbh, n_pbh, merger_rate_prior, sv_prior, fs, test)

        self.n_u = n_u
        self.lambda_prior = lambda_prior

    def save_p_gamma_table(self, m_dms, svs):
        """Saves the current p_gamma table. Wraps
        Distribution_N_gamma.save_p_gamma_table()."""
        self._p_n_gamma.save_p_gamma_table(m_dms, svs)

    def load_p_gamma_table(self):
        """Loads p_gamma table for the current parameter set. Wraps
        Distribution_N_gamma.load_p_gamma_table()."""
        self._p_n_gamma.load_p_gamma_table()

    def integrand(self, f, n_gamma, sv, m_dm):
        """Computes the value of the posterior integrand/summand.

        Parameters
        ----------
        f : np.array
            PBH relative abundance.
        n_gamma : np.array
            Number of PBHs passing Fermi point source cuts.
        sv : np.array
            DM self-annihilation cross sections.
        m_dm : np.array
            DM masses.

        Returns
        -------
        np.array
            Float/summand value.
        """
        def helper(f, n_gamma, sv, m_dm):
            return (self.p_sv(sv) * self.p_f(f) * self.p_u(n_gamma) *
                    self.p_n_gamma(n_gamma, sv, f, m_dm))

        return np.vectorize(helper)(f, n_gamma, sv, m_dm)

    def _get_trapz_f_samples(self, fs, integrand_vals, frac=0.1, n_low=50,
                             n_peak=225, n_high=125):
        """Resamples log-spaced points below, around and above the posterior
        integrand's peak.

        Parameters
        ----------
        fs : np.array
            Array of f values with shape [N].
        integrand_vals : np.array
            Array of integrand values with shape [N, M]. The rows correspond to
            the values in fs and columns to different values of n_gamma.
        n_low : int
            Number of points to sample below peak. If peak is flush against
            fs[0], then n_low more values will be sampled around the peak.
        n_peak : int
            Number of points to sample around peak.
        n_high : int
            Number of points to sample above peak. If peak is flush against
            fs[-1], then n_high more values will be sampled around the peak.

        Returns
        -------
        np.array
            Array of shape [n_low + n_peak + n_high, M] containing resampled f
            values.
        """
        # Define the integrand's peak
        integrand_max = integrand_vals.max(axis=0)
        min_sample_val = 0.1 * integrand_max
        d = (np.sign(min_sample_val - integrand_vals[:-1]) -
             np.sign(min_sample_val - integrand_vals[1:]))

        # Select the peak of the integrand
        f_low = fs[np.argmax(d > 0, axis=0)]
        idx_high = np.argmax(d < 0, axis=0)
        idx_high[idx_high == 0] = -1
        f_high = fs[idx_high]

        # Sample around peak
        f_samples = np.zeros(
            [n_low + n_peak + n_high, integrand_vals.shape[1]])
        for i in range(integrand_vals.shape[1]):
            n_peak_cur = n_peak
            f_samples_low = np.array([])
            f_samples_high = np.array([])

            if f_low[i] == fs[0]:  # peak is flush against fs[0]
                n_peak_cur += n_low
            else:
                f_samples_low = np.geomspace(fs[0], f_low[i], n_low)

            if f_high[i] == fs[-1]:  # peak is flush against fs[-1]
                n_peak_cur += n_high
            else:
                f_samples_high = np.geomspace(f_high[i], fs[-1], n_high)

            f_samples_peak = np.geomspace(f_low[i], f_high[i], n_peak_cur)
            f_samples[:, i] = np.concatenate(
                [f_samples_low, f_samples_peak, f_samples_high])

        return f_samples

    def _get_posterior_val(self, sv, m_dm):
        """Computes the posterior for <sigma v>. Supports broadcasting over sv
        and m_dm.

        Parameters
        ----------
        sv : np.array
            DM self-annihilation cross sections.
        m_dm : np.array
            DM masses.

        Returns
        -------
        np.array
            Posterior PDF at the specified point.
        """
        def helper(sv, m_dm):  # computes posterior at a single point
            # Compute integrand values over an initial f grid
            fs = np.geomspace(f_min, f_max, 100)
            n_gammas = np.arange(0, self.n_u + 1, 1)
            n_gamma_mg, f_mg = np.meshgrid(n_gammas, fs)
            integrand_vals = self.integrand(f_mg, n_gamma_mg, sv, m_dm)

            # Get grid of fs below, around and above the integrand's peak
            f_mg = self._get_trapz_f_samples(fs, integrand_vals)
            n_gamma_mg = n_gammas * np.ones_like(f_mg)
            integrand_vals = self.integrand(f_mg, n_gamma_mg, sv, m_dm)

            return trapz(integrand_vals, f_mg, axis=0).sum()

        return np.vectorize(helper)(sv, m_dm)

    def filename_suffix(self):
        """Add extra info to filename."""
        return "ps_{}_prior_lambda={}".format(super().filename_suffix(),
                                              self.lambda_prior)

    def p_n_gamma(self, n_gamma, sv, f, m_dm):
        """Computes p(n_gamma|m_pbh, f, m_dm, sv)."""
        return self._p_n_gamma(n_gamma, sv, f, m_dm)

    def p_gamma(self, sv, m_dm):
        """Returns the probability p_gamma that an individual PBH is
        detectable.
        """
        return self._p_n_gamma.p_gamma(sv, m_dm)

    def p_u(self, n_gamma):
        """Computes p(n_u|n_gamma)."""
        return self._p_u(n_gamma, self.n_u)

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, val):
        # Synchronizes final state
        self._p_n_gamma.fs = val
        self._fs = val

    @Posterior.m_pbh.setter
    def m_pbh(self, val):
        # Synchronize PBH mass
        Posterior.m_pbh.fset(self, val)
        self._p_n_gamma.m_pbh = val

    @property
    def lambda_prior(self):
        return self._lambda_prior

    @lambda_prior.setter
    def lambda_prior(self, val):
        # Synchronize astrophysical fake rate prior
        self._lambda_prior = val
        self._p_u.lambda_prior = val

from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

from constants import post_sv_ci, fs_0


"""
Shared infrastructure for point-source and diffuse posterior analyses.

Notes
------------
Each probability distribution is a callable (either a function or a
class).

Distributions that take a while to compute save their parameters to tables that
can easily be loaded. This includes p(f|m_pbh, n_pbh) and
p(n_gamma|m_dm, sv, m_pbh, f).

If changing a parameter requires time-consuming updates to the distribution's
parameters, it should be an argument to its initializer. If not, it should be
an argument to `__call__`.
"""

# Directory containing tables for p(f | m_pbh, n_pbh)
post_f_dir = "data/posteriors_f/"
# Output directory for <sigma v> posterior tables
post_sv_dir = "data/posteriors_sv/"
# Directory for p_gamma tables
p_gamma_dir = "data/p_gammas/"
# Directory for <sigma v> bounds
sv_bounds_dir = "data/bounds/"
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

        self.p_f = interp1d(fs, p_fs, bounds_error=False, fill_value=0.)

    def __call__(self, f):
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
        if self.p_f is None:
            self._load_p_f_gw()
        return self.p_f(f)

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
            self.p_f = None

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
            self.p_f = None

    @property
    def n_pbh(self):
        """float : number of PBH detections."""
        return self._n_pbh

    @n_pbh.setter
    def n_pbh(self, val):
        self._n_pbh = val
        self.p_f = None


class Prior_sv:  # __init__(sv_prior), __call__(sv)
    """Represents the prior p(sv)."""
    def __init__(self, sv_prior="U"):
        """
        Parameters
        ----------
        sv_prior : "LF", "U"
            Specifies which prior to use. Defaults to a uniform prior, the most
            conservative choice.
        """
        self.sv_prior = sv_prior

    def call(self, sv):
        """p(<sigma v>), the prior on <sigma v>.
        """
        if self.sv_prior == "LF":
            return 1 / sv
        elif self.sv_prior == "U":
            return 1
        # return np.vectorize(helper)(sv)

    @property
    def sv_prior(self):
        return self._sv_prior

    @sv_prior.setter
    def sv_prior(self, val):
        if val not in ["U", "LF"]:
            raise ValueError("Invalid prior on <sigma v>")
        else:
            self._sv_prior = val


class Posterior(ABC):
    """
    Base class for computing p(<sigma v> | ...), saving and loading data tables
    and computing upper bounds.
    """

    def __init__(self, m_pbh, n_pbh, merger_rate_prior="LF",
                 sv_prior="U", fs=fs_0, test=False):
        """Initializer.

        Parameters
        ----------
        m_pbh : float
            PBH mass.
        n_pbh : float
            Number of PBH detections
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
        # Must instantiate these before assigning other attributes. These are
        # wrapped with @property to keep the attributes m_pbh, n_pbh and
        # merger_rate_prior synchronized by hiding the underlying classes.
        self._p_f = Distribution_f(m_pbh, n_pbh, merger_rate_prior)
        self._p_sv = Prior_sv(sv_prior)

        self.m_pbh = m_pbh
        self.n_pbh = n_pbh
        self.fs = fs

        self.merger_rate_prior = merger_rate_prior
        self.sv_prior = sv_prior

        self.test = test

    @abstractmethod
    def _get_posterior_val(self, sv, m_dm):
        """Computes the posterior for <sigma v>. Must support broadcasting over
        sv and m_dm.
        """
        pass

    def save_posterior_table(self, svs, m_dms):
        """Generates a table containing the posterior for <sigma v>.

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

        # Compute unnormalized posterior
        sv_col = np.repeat(svs, m_dms.size)
        m_dm_col = np.tile(m_dms, svs.size)
        un_post_vals_col = self._get_posterior_val(sv_col, m_dm_col)
        np.savetxt(
            "{}{}posterior_sv_{}.csv".format(post_sv_dir,
                                             "test/" if self.test else "",
                                             self.filename_suffix()),
            np.stack([sv_col, m_dm_col, un_post_vals_col]).T,
            header=self.header())

        # Compute normalized posterior
        un_post_vals = un_post_vals_col.reshape([svs.size, m_dms.size])
        n_post_vals = un_post_vals_col.reshape([svs.size, m_dms.size]).copy()
        for i, (m_dm, un_post) in enumerate(zip(m_dms, un_post_vals.T)):
            # `quad` will not give any higher accuracy than `trapz`
            n_post_vals[:, i] = un_post / trapz(un_post, svs)

        np.savetxt(
            "{}{}normalized_posterior_sv_{}.csv".format(
                post_sv_dir, "test/" if self.test else "",
                self.filename_suffix()),
            np.stack([sv_col, m_dm_col, n_post_vals.flatten()]).T,
            header=self.header())

    def sv_bounds(self, alpha=0.95, save=True):
        """Computes and saves bounds on <sigma v>.

        Returns
        -------
        np.array
            Bounds on <sigma v> at each of the DM masses in the posterior
            tables for the given PBH mass and number. Saves these bounds to the
            data/bounds/ directory.
        """
        svs, m_dms, post_vals = self.load_posterior(normalized=True)
        sv_mg, m_dm_mg = np.meshgrid(svs, m_dms)
        sv_bounds = np.zeros_like(m_dms)

        # Compute bound for each DM mass
        for i, (m_dm, p_vals) in enumerate(zip(m_dms, post_vals.T)):
            sv_bounds[i] = post_sv_ci(svs, p_vals)

        if save:
            np.savetxt(
                "{}{}sv_bounds_{}.csv".format(sv_bounds_dir,
                                              "test/" if self.test else "",
                                              self.filename_suffix()),
                np.stack([m_dms, sv_bounds]).T,
                header=("{}% CI bounds on <sigma v>.\nColumns: m_DM (GeV), "
                        "<sigma v> (cm^3/s).").format(100 * alpha))

        return m_dms, sv_bounds

    def load_posterior(self, normalized=False):
        """Loads a table of posterior values for <sigma v>.

        Returns
        -------
        m_dms, svs, post_vals
            post_vals is defined so that:
                post_vals[i, j] = posterior(m_dms[i], svs[j]).
        """
        sv_col, m_dm_col, post_col = np.loadtxt(
            "{}{}{}posterior_sv_{}.csv".format(
                post_sv_dir, "test/" if self.test else "",
                "normalized_" if normalized else "", self.filename_suffix())).T
        svs = np.unique(sv_col)
        m_dms = np.unique(m_dm_col)
        post_vals = post_col.reshape([svs.size, m_dms.size])
        return svs, m_dms, post_vals

    def filename_suffix(self):
        return ("M={:.1f}_N={}_prior_rate={}_"
                "prior_sv={}").format(self.m_pbh, self.n_pbh,
                                      self.merger_rate_prior, self.sv_prior)

    def header(self):
        """Header for posterior table file"""
        return ("Posterior for <sigma v>.\n"
                "Columns: <sigma v> (cm^3/s), m_DM (GeV), posterior.")

    def p_f(self, f):
        """Computes p(f | m_pbh, n_pbh)"""
        return self._p_f(f)

    def p_sv(self, sv):
        """Computes prior p(sv)"""
        return self._p_sv.call(sv)

    # Properties must also set the corresponding properties of Posterior's
    # attributes to keep them synchronized!
    @property
    def m_pbh(self):
        return self._m_pbh

    @m_pbh.setter
    def m_pbh(self, val):
        # Synchronize attribute
        self._m_pbh = val
        self._p_f.m_pbh = val

    @property
    def n_pbh(self):
        return self._n_pbh

    @n_pbh.setter
    def n_pbh(self, val):
        # Synchronize attribute
        self._n_pbh = val
        self._p_f.n_pbh = val

    @property
    def merger_rate_prior(self):
        return self._merger_rate_prior

    @merger_rate_prior.setter
    def merger_rate_prior(self, val):
        # Synchronize attribute
        self._merger_rate_prior = val
        self._p_f.merger_rate_prior = val

    @property
    def sv_prior(self):
        return self._sv_prior

    @sv_prior.setter
    def sv_prior(self, val):
        # Synchronize attribute
        self._sv_prior = val
        self._p_sv.sv_prior = val

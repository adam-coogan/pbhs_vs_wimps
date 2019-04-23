import numpy as np
from scipy.integrate import quad, trapz
from constants import e_egb, err_high_egb, fs_0
from diffuse_constraints import phi_ex
from posterior_inference_shared import f_min, f_max, Posterior


p_gamma_dir = "data/p_gammas/"


"""
Classes for performing posterior analysis with diffuse constraints.
"""


class DiffusePosterior(Posterior):
    def __init__(self, m_pbh, n_pbh, merger_rate_prior="LF", sv_prior="U",
                 fs=fs_0, test=True):
        """
        Parameters
        ----------
        prior: "U", "LF"
            Determines which prior to use. Defaults to the conservative choice,
            "U".
        """
        # Subclasses whose properties need to be synchronized with this
        # object's must be instantiated before calling the superclass
        # initializer
        super().__init__(m_pbh, n_pbh, merger_rate_prior, sv_prior, fs, test)

    def integrand(self, f, sv, m_dm):
        """Computes the value of the posterior integrand/summand.

        Parameters
        ----------
        sv : float
        f : float
        """
        def helper(f, sv, m_dm):
            log_prob = np.log(self.p_sv(sv) * self.p_f(f))
            phi_dms = phi_ex(e_egb, m_dm, sv, self.m_pbh, f, self.fs)
            log_prob += -0.5 * np.sum((phi_dms / err_high_egb)**2 +
                                      np.log(2 * np.pi * err_high_egb**2))

            return np.exp(log_prob)

        return np.vectorize(helper)(f, sv, m_dm)

    def _get_f_quad_points(self, fs, integrand_vals, frac=0.1, n=10):
        """Samples log-space points around the peak of the posterior integrand.

        Returns
        -------
        Array of log-spaced values of f out to where the integrand decreases to
        10% of its maximum value, and the f at which it attains its maximum
        value.
        """
        integrand_max = integrand_vals.max(axis=0)
        # Choose some fraction of the max value to sample out to
        min_sample_val = frac * integrand_max
        d = (np.sign(min_sample_val - integrand_vals[:-1]) -
             np.sign(min_sample_val - integrand_vals[1:]))
        try:
            f_low = fs[max(0, np.where(d > 0)[0][0])]
        except IndexError:
            f_low = fs[0]
        try:
            f_high = fs[min(len(fs)-1, np.where(d < 0)[0][-1])]
        except IndexError:
            f_high = fs[-1]

        return np.append(np.geomspace(f_low, f_high, n-1),
                         fs[np.argmax(integrand_vals, axis=0)])

    def _get_trapz_f_samples(self, fs, integrand_vals, frac=1e-10, n_low=75,
                             n_high=350):
        """Resamples log-spaced points below and above the posterior
        integrand's peak.

        Parameters
        ----------
        fs : np.array with shape N
        integrand_vals : np.array with shape N x M
            Array of integrand values, with rows corresponding to the values in
            fs and columns to different values of n_gamma.
        n_low : int
            Number of points to sample below peak. If peak is flush against
            fs[0], then n_low more values will be sampled around the peak.
        n_high : int
            Number of points to sample above peak. If peak is flush against
            fs[-1], then n_high more values will be sampled around the peak.

        Returns
        -------
        np.array with shape n_low+n_peak+n_high x M
            Resampled f values
        """
        # Define the integrand's peak
        integrand_max = integrand_vals.max(axis=0)
        min_sample_val = frac * integrand_max
        d = (np.sign(min_sample_val - integrand_vals[:-1]) -
             np.sign(min_sample_val - integrand_vals[1:]))

        # Select above and below peak
        f_peak = fs[integrand_vals.argmax(axis=0)]
        f_low = fs[np.argmax(d > 0, axis=0)]
        idx_high = np.argmax(d < 0, axis=0)
        idx_high[idx_high == 0] = -1
        f_high = fs[idx_high]

        return np.concatenate([np.geomspace(f_low, f_peak, n_low),
                               np.geomspace(f_peak, f_high, n_high)])

    def _get_posterior_val(self, sv, m_dm):
        """Computes the posterior for <sigma v>. Supports broadcasting over sv
        and m_dm. See documentation for `posterior_integrand`.
        """
        def helper(sv, m_dm):
            # Compute integrand values over an initial f grid
            fs = np.geomspace(f_min, f_max, 20)
            sv_mg, f_mg = np.meshgrid(sv, fs)
            integrand_vals = self.integrand(f_mg, sv_mg, m_dm)
            # Resample fs
            f_mg = self._get_trapz_f_samples(
                fs, integrand_vals, n_low=50, n_high=150)
            sv_mg = sv * np.ones_like(f_mg)
            # Compute integral over new grid
            integrand_vals = self.integrand(f_mg, sv_mg, m_dm)
            integral_vals = trapz(integrand_vals, f_mg, axis=0)
            return integral_vals

        return np.vectorize(helper)(sv, m_dm)

    def filename_suffix(self):
        """Add extra info to filename"""
        return "diff_{}_prior".format(super().filename_suffix())

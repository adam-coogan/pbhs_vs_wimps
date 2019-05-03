import numbers
import numpy as np
from scipy.special import gammainc

from src.constants import gamma_tr_sample, GeV_to_erg, pbh_ann_rate
from src.constants import (r_s_mw, alpha_mw, n_mw_pbhs, d_earth,
                           to_galactic_coords)
from src.constants import int_dnde_interps, int_e_dnde_interps, kpc_to_cm
from src.constants import fs_0, flux_type_0, flux_thresh_0, b_cut_0

"""
Class for performing a Monte Carlo simulation of PBHs to assess their
detectability as point sources. Also includes a function to compute a simple
version of the resulting constraint on <sigma v>.
"""


class PBHHaloSim(object):
    """Performs Monte Carlo analysis of detectability of PBHs surrounded by DM
    halos as gamma ray point sources.

    Examples
    --------
    The simulation is used by initializing it with a set of parameters and
    running it:

    >>> np.random.seed(10)
    >>> sim = PBHHaloSim(m_pbh=0.5, f=1e-3, m_dm=100, sv=1e-32)
    >>> sim.run()

    After this, a large number of attributes are set to study. The most
    relevant for our work is the probability of a PBH being detectable:

    >>> sim.pr_det, sim.pr_det_err
    (1.3382265383158258e-08, 1.1556284331516314e-09)

    Some other attributes are described below.

    Attributes
    ----------
    positions_abs : np.array
        Sampled PBH coordinates (r, th, phi) in the galactic frame.
    positions : np.array
        Galactic coordinates (d, b, l) for sampled PBHs.
    n_halo_pbhs : float
        Number of PBHs in the Milky Way.
    ann_rates : np.array
        PBH annihilation rates.
    fluxes : np.array
        Fluxes of simulated PBHs. Note that the is not equal to the physical
        flux distribution for PBHs in the Milky Way if importance sampling is
        used for the angular or radial coordinates.
    pr_det : float
        Probability that an individual PBH is detectable.
    pr_det_err : float
        95% Monte Carlo error bars on this probability.
    n_det : float
        Expected number of detectable PBHs in the Milky Way.
    n_det_err : float
        95% Monte Carlo error bars on this quantity.
    """

    @classmethod
    def final_states(cl):
        """Returns list of valid DM annihilation final states."""
        return ["e", "c", "b", "t", "W", "Z", "g", "h"]

    def __init__(self,
                 m_pbh,
                 f,
                 m_dm=100,
                 sv=3e-26,
                 fs=fs_0,
                 flux_type=flux_type_0,
                 b_cut=b_cut_0,
                 flux_thresh=flux_thresh_0,
                 n_samples=1000):
        """Sets up the simulation.

        Parameters
        ----------
        m_pbh : float or float -> np.array
            PBH mass or function for sampling from the PBH mass distribution.
            Only the former (ie, a monochromatic PBH mass distribution) is
            currently supported.
        f : float
            Relative PBH abundance.
        m_dm : float
            Dark matter mass.
        sv : float
            Dark matter thermally-averaged self-annihilation cross section.
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
        """
        if fs not in PBHHaloSim.final_states():
            raise ValueError("Invalid final state")
        self.m_pbh = m_pbh
        self.f = f
        self.m_dm = m_dm
        self.sv = sv
        self.fs = fs
        self.flux_type = flux_type
        self.b_cut = b_cut
        self.flux_thresh = flux_thresh
        self.n_samples = n_samples
        # Internal parameters for sampling r
        self._x_shape = 3 / alpha_mw
        self._x_scale = alpha_mw * r_s_mw**alpha_mw / 2
        self._x_exp = 1 / alpha_mw
        self._r_exp = alpha_mw
        # Parameters for sampling PBH latitude and longitude
        self.slat_ms = np.ones(self.n_samples)  # max sin of absolute latitude
        self.phi_ms = np.pi * np.ones(self.n_samples)  # max absolute longitude
        # Factor for importance sampling
        self.pr_sampling_correction = np.ones(self.n_samples)

        # DM annihilation spectrum
        if self.flux_type == "e dnde":  # for energy flux
            self.flux_fact = GeV_to_erg * int_e_dnde_interps[fs](self.m_dm)
        elif self.flux_type == "dnde":  # for flux
            self.flux_fact = int_dnde_interps[fs](self.m_dm)
        else:
            raise ValueError("Flux type not supported")

    def _sample_pbh_masses(self):
        """Samples from the PBH mass function and computes the number of PBHs
        in the Milky Way halo.

        Notes
        -----
        Currently only supports monochromatic mass distributions. Should be
        easy to extend.
        """
        if isinstance(self.m_pbh, numbers.Number):
            self.m_pbhs = self.m_pbh * np.ones(self.n_samples)
            # Number of PBHs in the Milky Way halo
            self.n_halo_pbhs = n_mw_pbhs(self.f, np.mean(self.m_pbhs))
        else:
            raise ValueError("Extended PBH mass functions not supported")

    def _ann_rates(self):
        """Computes DM annihilation rate Gamma in halos around simulated PBHs
        for the sampled PBHs.
        """
        self.ann_rates = pbh_ann_rate(self.m_dm, self.sv, self.m_pbhs)

    def _max_detectable_distances(self):
        """Compute max distance from Earth at which PBHs with sampled masses
        could be detected in kpc.

        Notes
        -----
        Must be called after `_sample_pbh_masses()` and `_ann_rates()`.
        """
        fluxes_1kpc = self._flux_helper(self.ann_rates, self.flux_fact, 1.)
        self.d_ms = np.sqrt(fluxes_1kpc / self.flux_thresh)

    def _get_angular_dist_bounds(self, efficient_angular_sampling=True):
        """Determines bounds for sampling each PBH's angular coordinates.

        Notes
        -----
        This function first computes the maximum distance from Earth at which
        each PBH could be detected. It then finds the extent in each direction
        for the bounding (lat, lon) rectangle, and computes the corresponding
        correction to the detection probability.

        Must be called after `_sample_pbh_masses()` and `_ann_rates()`, and
        before `_sample_positions()`.
        """
        if efficient_angular_sampling:
            self._max_detectable_distances()
            # Bounds on sine of latitude from galactic plane
            self.slat_ms = np.ones(self.n_samples)
            self.slat_ms[self.d_ms < d_earth] = (
                self.d_ms[self.d_ms < d_earth] / d_earth)
            # Longitude bounds
            self.phi_ms = np.pi * np.ones(self.n_samples)
            self.phi_ms[self.d_ms < d_earth] = (np.arcsin(
                self.d_ms[self.d_ms < d_earth] / d_earth))
            # Importance sampling correction
            self.pr_sampling_correction *= (4*self.slat_ms * self.phi_ms /
                                            (4*np.pi))

    def _radial_samples(self, n):
        """Sample from the PBH radial distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        np.array
            n draws from the PBH radial distribution.
        """
        return np.random.gamma(shape=self._x_shape,
                               scale=self._x_scale,
                               size=n)**self._x_exp

    def _radial_dist_cdf(self, r):
        """Cumulative distribution function for the radial coordinate. Used to
        compute importance sampling correction when r is sampled from a
        restricted range.

        Parameters
        ----------
        r : float or np.array
            Value at which to compute CDF.

        Returns
        -------
        float or np.array
            CDF at r.
        """
        return gammainc(self._x_shape, r**self._r_exp / self._x_scale)

    def _radial_samples_tr(self, r_min, r_max):
        return gamma_tr_sample(r_min**self._r_exp, r_max**self._r_exp,
                               shape=self._x_shape,
                               scale=self._x_scale)**self._x_exp

    def _sample_positions(self, truncate_radial_samples=True):
        """Sample PBH positions and convert to galactic coordinates.

        Parameters
        ----------
        truncate_radial_samples : boolean
            If `true`, uses inverse CDF sampling to only sample values for the
            radial coordinate in regions where the PBHs can be detected. This
            significantly improves accuracy at the cost of speed.
        """
        if truncate_radial_samples:
            r_min = np.max([np.zeros(self.n_samples), d_earth - self.d_ms], 0)
            r_max = d_earth + self.d_ms
            # Inverse CDF method
            rs = self._radial_samples_tr(r_min, r_max)
            # Correct the probabilities based on probability of r being
            # in [r_min, r_max]
            self.pr_sampling_correction *= (self._radial_dist_cdf(r_max) -
                                            self._radial_dist_cdf(r_min))
        else:
            rs = self._radial_samples(self.n_samples)

        # Uniformly sample the radial coordinates
        ths = np.arccos(np.random.uniform(-self.slat_ms, self.slat_ms))
        phis = np.random.uniform(-self.phi_ms, self.phi_ms)
        self.positions_abs = np.stack([rs, ths, phis])

        self.positions = to_galactic_coords(*self.positions_abs)

    def _flux_helper(self, ann_rate, flux_fact, dist):
        """Helper function to compute the gamma-ray flux for a PBH. In addition
        to being used to compute the fluxes for the simulated PBHs, this is
        required for computing d_ms, the maximum distances at which the PBHs
        can be detected.

        To-do
        -----
        Make sure the units of this are the same as flux_thresh (cm^-2 s^-1)!

        Parameters
        ----------
        ann_rate : float
            DM annihilation rate around PBH, Hz.
        dist : float
            PBH distance from Earth, kpc.

        Returns
        -------
        float or np.array
            Gamma-ray flux in cm^-2 s^-1 if flux_type is "dnde" or energy flux
            in erg cm^-2 s^-1 if flux_type is "e dnde".
        """
        return ann_rate * flux_fact / (4*np.pi * (kpc_to_cm*dist)**2)

    def _fluxes(self):
        """Computes gamma-ray fluxes or energy fluxes for the simulated PBHs.
        """
        self.fluxes = self._flux_helper(self.ann_rates, self.flux_fact,
                                        self.positions[0])

    def _pr_det(self):
        """Determines probability of a PBH passing the flux and |b| cuts, as
        well as the associated Monte Carlo error.
        """
        # Indices of PBHs passing cuts
        self.passed_b_cut_idx = np.abs(self.positions[1]) > self.b_cut
        self.passed_flux_cut_idx = self.fluxes > self.flux_thresh

        # Values in this array are set to one for PBHs passing all cuts
        self.detectable = np.zeros(self.n_samples)
        self.detectable[self.passed_b_cut_idx & self.passed_flux_cut_idx] = 1.
        # Importance sampling correction
        self.detectable *= self.pr_sampling_correction

        # Probability for a PBH in the MW halo to pass all cuts
        self.pr_det = np.mean(self.detectable)
        # 95% confidence interval. This form holds thanks to the central limit
        # theorem.
        self.pr_det_err = (1.96 * np.std(self.detectable) /
                           np.sqrt(self.n_samples))

    def _n_det_expected(self):
        """Determines the expected number of PBHs (in the Milky Way halo, not
        the simulation) passing the flux and |b| cuts, as well as the
        associated Monte Carlo error.
        """
        self.n_det = self.n_halo_pbhs * self.pr_det
        self.n_det_err = self.n_halo_pbhs * self.pr_det_err

    def run(self,
            efficient_angular_sampling=True,
            truncate_radial_samples=True):
        """Runs the simulation. This is the main function the user should
        interact with.

        Notes
        -----
        If efficient_angular_sampling and truncate_radial_samples are False,
        _n_det_expected() can be reevaluated with different flux and |b| cuts.
        Otherwise, it cannot, since the sampled positions are dependent on
        flux_thresh.

        Parameters
        ----------
        efficient_angular_sampling : bool
            Indicates whether or not to restrict the range from which the PBH
            angular coordinates are sampled.
        truncate_radial_samples : bool
            Indicates whether or not to restrict the range from which the PBH
            radial coordinates are sampled.
        """
        # Functions must be called in this order
        self._sample_pbh_masses()
        self._ann_rates()
        self._get_angular_dist_bounds(efficient_angular_sampling)
        self._sample_positions(truncate_radial_samples)
        self._fluxes()
        self._pr_det()
        self._n_det_expected()

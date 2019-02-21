import numbers
import numpy as np
from scipy.special import gammainc

from constants import gamma_tr_sample
from constants import r_s_mw, alpha_mw, n_mw_pbhs, d_earth, to_galactic_coords
from constants import int_dnde_interps, int_e_dnde_interps
from constants import kpc_to_cm, yr_to_s, age_of_universe, h_hubble
from constants import fs_0, n_u_0, flux_type_0, flux_thresh_0, b_cut_0

class PBHHaloSim(object):
    """Class for performing Monte Carlo analysis of detectability of PBHs
    surrounded by DM halos as gamma ray point sources.
    """
    @classmethod
    def final_states(cl):
        """Returns list of valid DM annihilation final states."""
        return ["e", "c", "b", "t", "W", "Z", "g", "h"]

    def __init__(self, mass_dist, f_pbh, m_dm=100, sv=3e-26, fs=fs_0,
                 flux_type=flux_type_0, b_cut=b_cut_0,
                 flux_thresh=flux_thresh_0, n_samples=1000):
        """
        Parameters
        ----------
        mass_dist : float -> np.array or float
            Function to sample from PBH mass distribution. Pass a float to indicate
            a monochromatic mass distribution at the indicated value in M_sun.
        f_pbh : float
            Omega_PBH / Omega_CDM.
        m_dm : float
            Dark matter mass, GeV
        sv : float
            Dark matter thermally-averaged self-annihilation cross section, cm^3/s.
        fs : string
            Annihilation final state.
        flux_type : string
            Use "dnde" to compute fluxes or "e dnde" to compute energy fluxes.
        b_cut : float
            Specifies the latitude cut for calculating the number of detectable PBH
            halos.
        flux_thresh : float
            Flux detectability threshold in cm^-2 s^-1 (for `flux_type` "dnde") or
            erg cm^-2 s^-1 (for `flux_type` "e dnde"). Defaults to the conservative
            threshold for the energy flux above 1 GeV from arXiv:1601.06781.
        n_samples : int
            Number of iterations of the simulation to run.
        """
        if fs not in PBHHaloSim.final_states():
            raise ValueError("Invalid final state")
        self.mass_dist = mass_dist
        self.f_pbh = f_pbh
        self.m_dm = m_dm
        self.sv = sv
        self.fs = fs
        self.flux_type = flux_type
        self.b_cut = b_cut
        self.flux_thresh = flux_thresh
        self.n_samples = n_samples
        # Internal parameters
        self._x_shape = 3 / alpha_mw
        self._x_scale = alpha_mw * r_s_mw**alpha_mw / 2
        self._x_exp = 1 / alpha_mw
        self._r_exp = alpha_mw
        self.slat_ms = np.ones(self.n_samples)  # max sin of absolute latitude
        self.phi_ms = np.pi * np.ones(self.n_samples)  # max absolute longitude
        self.pr_sampling_correction = np.ones(self.n_samples)
        # DM annihilation spectrum
        if self.flux_type == "e dnde":  # for energy flux
            self.flux_fact = GeV_to_erg * int_e_dnde_interps[fs](self.m_dm)
        elif self.flux_type == "dnde":  # for flux
            self.flux_fact = int_dnde_interps[fs](self.m_dm)
        else:
            raise ValueError("Flux type not supported")

    def _sample_pbh_masses(self):
        """Samples from the PBH mass function.

        Notes
        -----
        Currently only supports monochromatic mass distributions. Should be
        easy to extend.
        """
        if isinstance(self.mass_dist, numbers.Number):
            self.m_pbhs = self.mass_dist * np.ones(self.n_samples)
            # Number of PBHs in the Milky Way halo
            self.n_halo_pbhs = n_mw_pbhs(self.f_pbh, np.mean(self.m_pbhs))
        else:
            raise ValueError("Extended PBH mass functions not supported")

    def _ann_rates(self):
        """Computes DM annihilation rate Gamma_ann in halos around simulated
        PBHs.

        Notes
        -----
        From Byrnes2019.

        Returns
        -------
        UCMH annihilation rate in 1/s.
        """
        rho_max = self.m_dm / (self.sv * age_of_universe * yr_to_s)
        r_cut = h_hubble * 1.3e-7 * (100./self.m_dm * self.sv/3e-26)**(4/9) * (self.m_pbhs/1.)**(1/3)
        # Added missing factor of 1/2
        self.ann_rates = (4*np.pi*self.sv*rho_max**2 * (kpc_to_cm*r_cut)**3 / (2*self.m_dm**2))
        return self.ann_rates

    def _get_angular_dist_bounds(self, efficient_angular_sampling=True):
        """Determines bounds for sampling each PBH's angular coordinates.

        Notes
        -----
        This function first computes the maximum distance from Earth at which each
        PBH could be detected. It then finds the extent in each direction for the bounding
        (lat, lon) rectangle, and computes the corresponding correction to the detection
        probability.
        Must be called after `_sample_pbh_masses()` and `_ann_rates()` and
        before `_sample_positions()`!
        """
        if efficient_angular_sampling:
            self._max_detectable_distances()
            # Bounds on sine of latitude from galactic plane
            self.slat_ms = np.ones(self.n_samples)
            self.slat_ms[self.d_ms < d_earth] = self.d_ms[self.d_ms < d_earth] / d_earth
            # Longitude bounds
            self.phi_ms = np.pi * np.ones(self.n_samples)
            self.phi_ms[self.d_ms < d_earth] = (np.arcsin(self.d_ms[self.d_ms < d_earth] / d_earth))
            # Determine correction for probabilities
            self.pr_sampling_correction *= 4*self.slat_ms * self.phi_ms / (4*np.pi)

    def _max_detectable_distances(self):
        """Compute max distance from Earth at which PBHs with sampled masses
        could be detected, kpc."""
        fluxes_1kpc = self._flux_helper(self.ann_rates, self.flux_fact, 1.)
        self.d_ms = np.sqrt(fluxes_1kpc / self.flux_thresh)
        return self.d_ms

    def _radial_samples(self, n):
        """Sample from the PBH radial distribution.

        Notes
        -----
        Makes use of the factor that the PDF for x = r^alpha is Gamma(shape, scale).
        """
        return np.random.gamma(shape=self._x_shape,
                               scale=self._x_scale,
                               size=n)**self._x_exp

    def _radial_dist_cdf(self, r):
        """Cumulative distribution function for the radial coordinate.
        """
        return gammainc(self._x_shape, r**self._r_exp / self._x_scale)

    def _radial_samples_tr(self, r_min, r_max):
        return gamma_tr_sample(r_min**self._r_exp, r_max**self._r_exp,
                               shape=self._x_shape,
                               scale=self._x_scale)**self._x_exp

    def _sample_positions(self, truncate_radial_samples=True):
        """Samples the PBHs positions in galactic coordinates.

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

        ths = np.arccos(np.random.uniform(-self.slat_ms, self.slat_ms))
        phis = np.random.uniform(-self.phi_ms, self.phi_ms)
        self.positions_abs = np.stack([rs, ths, phis])

        self.positions = to_galactic_coords(*self.positions_abs)
        return self.positions

    def _flux_helper(self, ann_rate, flux_fact, dist):
        """
        Parameters
        ----------
        ann_rate : Hz
        dist : kpc
        """
        return ann_rate * flux_fact / (4*np.pi * (kpc_to_cm*dist)**2)

    def _fluxes(self):
        """Computes gamma-ray flux for simulated PBHs.

        Returns
        -------
        Energy fluxes if erg / cm^2 / s if flux_type is "e dnde". Fluxes in
        1 / cm^2 / s if flux_type is "e dnde".
        """
        self.fluxes = self._flux_helper(self.ann_rates, self.flux_fact, self.positions[0])
        return self.fluxes

    def _pr_det(self):
        """Determines probability of a PBH passing the flux and |b| cuts.
        """
        # Indices of PBHs passing cuts
        self.passed_b_cut_idx = np.abs(self.positions[1]) > self.b_cut
        self.passed_flux_cut_idx = self.fluxes > self.flux_thresh

        # Values in this array are set to one for PBHs passing all cuts
        self.detectable = np.zeros(self.n_samples)
        self.detectable[self.passed_b_cut_idx & self.passed_flux_cut_idx] = 1.
        # Correction for sampling over particular solid angles and radii
        self.detectable *= self.pr_sampling_correction

        # Probability for a PBH in the MW halo to pass all cuts
        self.pr_det = np.mean(self.detectable)
        # 95% confidence interval
        self.pr_det_err = 1.96 * np.std(self.detectable) / np.sqrt(self.n_samples)

        return self.pr_det, self.pr_det_err

    def _n_det_expected(self):
        """Determines the number of PBHs passing the flux and |b| cuts.
        """
        self.n_det = self.n_halo_pbhs * self.pr_det
        self.n_det_err = self.n_halo_pbhs * self.pr_det_err

        return self.n_det, self.n_det_err

    def run(self, efficient_angular_sampling=True, truncate_radial_samples=True):
        """Runs the simulation.

        Notes
        -----
        If `efficient_angular_sampling` and `truncate_radial_samples` are
        `False`, `_n_det_expected()` can be reevaluated with
        different flux and |b| cuts. Otherwise, it cannot, since the sampled
        positions are dependent on `flux_thresh`.
        """
        # Functions must be called in this order
        self._sample_pbh_masses()
        self._ann_rates()
        self._get_angular_dist_bounds(efficient_angular_sampling)
        self._sample_positions(truncate_radial_samples)
        self._fluxes()
        self._pr_det()
        self._n_det_expected()

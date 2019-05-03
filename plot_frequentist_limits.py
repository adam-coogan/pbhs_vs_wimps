from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np

from src.constants import colors
from src.diffuse_analysis import diffuse_limit
from src.distributions import Distribution_f
from src.point_source_analysis import point_source_limit

"""
Generates DM self-annihilation cross section limit plots using point-source and
diffuse gamma-ray observations.

To regenerate figure 2, run:

>>> python plot_frequentist_limits.py -plot_limits

To regenerate figure 3, run:

>>> python plot_frequentist_limits.py -plot_ps_diff
"""

# Directory containing GAMBIT contours
gambit_dir = "data/gambit/"
# Models for which to plot the envelope. The SingletDM contour will be plotted
# separately.
gambit_models = ["CMSSM", "MSSM7", "NUHM1", "NUHM2"]


def plot_gambit_contour_envelope(ax, color=None, padding=10, level=6):
    """Plots envelope of 95% CL profile likelihood contours computed by the
    GAMBIT collaboration for several new physics scenarios. The light grey
    contour indicates the singlet DM model.

    Parameters
    ----------
    ax : axes
        Axes on which to plot.
    color : str or None
        Contour color. If None, defaults to black.
    padding : int
        Amount to remove from beginning of GAMBIT contour files. If not set to
        ~10, the plot will contain vertical and horizontal lines around the
        edges.
    level : int
        Level of the contour. Leave this at 6 for 95% CL contours.
        TODO: Christoph said level should be 3, not 6...
    """
    # Grid size
    n_gambit_grid_rows, n_gambit_grid_cols = 101, 203
    # Construct parameter meshgrid
    m_dm_g = np.logspace(1, 4, n_gambit_grid_cols)[padding:]
    sv_g = np.logspace(-45, -23, n_gambit_grid_rows)[padding:]
    m_dm_g_mg, sv_g_mg = np.meshgrid(m_dm_g, sv_g)

    # Add the singlet contour separately
    singlet_contour = np.load(
        "{}contours_SingletDM.npy".format(gambit_dir)).T[padding:, padding:]
    ax.contourf(
        m_dm_g_mg,
        sv_g_mg,
        singlet_contour,
        levels=[0, level],
        alpha=0.2,
        colors=[color])

    # Construct envelope of other contours
    envelope = np.inf * np.ones(
        [n_gambit_grid_rows - padding, n_gambit_grid_cols - padding])
    for model in gambit_models:
        gambit_contour = np.load("{}contours_{}.npy".format(
            gambit_dir, model)).T[padding:, padding:]
        envelope = np.min([envelope, gambit_contour], axis=0)

    ax.contourf(
        m_dm_g_mg,
        sv_g_mg,
        envelope,
        levels=[0, level],
        alpha=0.35,
        colors=[color])


def plot_ps_diff(m_pbhs, n_pbhs, m_dms, fs, rate_priors,
                 fname="frequentist_sv_limits_ps_diff_comparison.pdf"):
    fig, axes = plt.subplots(
        n_pbhs.shape[0],
        n_pbhs.shape[1],
        sharex=True,
        sharey=True,
        figsize=(6, 7))

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            m_pbh = m_pbhs[i, j]
            n_pbh = n_pbhs[i, j]

            # Cross section limits
            # Different priors for merger rates
            for rate_prior, ls in zip(rate_priors, ['-', '-.']):
                # PBH abundance
                f = fs[(m_pbh, n_pbh, rate_prior)]

                # Diffuse limit
                lab = "Diffuse" if rate_prior == "J" else ""
                sv_diff_limits = diffuse_limit(m_dms, m_pbh, n_pbh, f,
                                               rate_prior)
                ax.loglog(m_dms, sv_diff_limits, label=lab, linestyle=ls,
                          color=colors[2])

                # Point source limit
                lab = "Point source" if rate_prior == "J" else ""
                sv_ps_limits = point_source_limit(m_dms, m_pbh, n_pbh, f,
                                                  rate_prior)
                ax.loglog(m_dms, sv_ps_limits, label=lab, linestyle=ls,
                          color=colors[3])

            # BSM physics contours
            plot_gambit_contour_envelope(ax, 'k')
            # Thermal relic cross section
            ax.text(1.12e1, 7e-26, "Thermal relic", alpha=0.6, fontsize=6)
            ax.axhline(
                3e-26, linestyle='--', color='k', alpha=0.6, linewidth=0.75)
            # Unitarity bound
            ax.loglog(m_dms, (m_dms / 1e3)**6 * 7e-41, '--k', alpha=0.6,
                      linewidth=0.75)
            ax.text(6e3, 1.7e-35, "Unitarity", fontsize=6, rotation=34,
                    alpha=0.6, horizontalalignment="center",
                    verticalalignment="center")

            # Formatting
            ax.set_xlim(m_dms[[0, -1]])
            ax.set_ylim(1e-44, 1e-23)
            ax.set_title(r"$(M_{\mathrm{PBH}}, N_{\mathrm{PBH}}) = $" +
                         "({:g} $M_\odot$, {:d})".format(m_pbh, n_pbh),
                         fontsize=9.0)
            if i == axes.shape[0] - 1:
                ax.set_xlabel(r"$m_{\chi}$ (GeV)", fontsize=12)
            if j == 0:
                ax.set_ylabel(
                    r"$f_{\chi}{}^4 (\sigma v)_0$ (cm$^3$/s)", fontsize=12)
            if i == 0 and j == 0:
                ax.plot([1e-50, 1e-50], [1e-50, 1e-50], color='k',
                        linestyle='-', label='Jeffreys Rate Prior')
                ax.plot([1e-50, 1e-50], [1e-50, 1e-50], color='k',
                        linestyle='-.', label='Log-flat Rate Prior')
                ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig("figures/{}".format(fname), bbox_inches='tight',
                pad_inches=0.05)


def plot_limits(m_pbhs, n_pbhs, m_dms, fs, rate_prior,
                fname="frequentist_sv_limits.pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    pbh_colors = {0.5: colors[0], 10: colors[1], 100: colors[2]}

    for i in range(m_pbhs.shape[0]):
        for j, ls in zip(range(n_pbhs.shape[1]), ["-", "--"]):
            m_pbh = m_pbhs[i, j]
            n_pbh = n_pbhs[i, j]
            # PBH abundance
            f = fs[(m_pbh, n_pbh, rate_prior)]
            # Diffuse limit
            sv_diff_limits = diffuse_limit(m_dms, m_pbh, n_pbh, f, prior)
            # Point source limit
            sv_ps_limits = point_source_limit(m_dms, m_pbh, n_pbh, f, prior)
            # Plot strongest gamma-ray constraint
            sv_bound = np.min([sv_diff_limits, sv_ps_limits], axis=0)
            ax.loglog(m_dms, sv_bound, color=pbh_colors[m_pbh], linestyle=ls,
                      linewidth=1)

    # BSM physics contours
    plot_gambit_contour_envelope(ax, 'k')
    # Thermal relic cross section
    ax.text(1.12e1, 7e-26, "Thermal relic", alpha=0.6, fontsize=6)
    ax.axhline(
        3e-26, linestyle='--', color='k', alpha=0.6, linewidth=0.75)
    # Unitarity bound
    ax.loglog(m_dms, (m_dms / 1e3)**6 * 7e-41, '--k', alpha=0.6,
              linewidth=0.75)
    ax.text(1e3, 2.2e-40, "Unitarity", fontsize=6, rotation=39,
            alpha=0.6, horizontalalignment="center",
            verticalalignment="center")

    # Formatting
    ax.set_xlim(m_dms[[0, -1]])
    ax.set_ylim(1e-44, 1e-23)
    ax.set_xlabel(r"$m_{\chi}$ (GeV)", fontsize=10.0)
    ax.set_ylabel(r"$f_{\chi}{}^4 (\sigma v)_0$ (cm$^3$/s)", fontsize=10.0)
    # N_min labels
    ax.text(1.2e1, 1e-31, r"$N_{\mathrm{ET}} = 1$", fontsize=7,
            color=pbh_colors[10])
    ax.text(1.2e1, 8e-34, r"$N_{\mathrm{SKA}} = 10$", fontsize=7,
            color=pbh_colors[100])
    ax.text(1.5e3, 4e-36, r"$N_{\mathrm{O3}} = 1$", fontsize=7,
            color=pbh_colors[0.5])
    # N_max labels
    ax.text(1.2e1, 7e-36, r"$N_{\mathrm{SKA}} = 80$", fontsize=7,
            color=pbh_colors[100])
    ax.text(1.2e1, 3e-39, r"$N_{\mathrm{ET}} = 24000$", fontsize=7,
            color=pbh_colors[10])
    ax.text(1.3e3, 2e-43, r"$N_{\mathrm{O3}} = 80$", fontsize=7,
            color=pbh_colors[0.5])

    fig.tight_layout()
    fig.savefig("figures/{}".format(fname), bbox_inches='tight',
                pad_inches=0.05)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-n_m_dms", type=int, default=30, help="number of DM masses")
    parser.add_argument(
        "-plot_limits",
        action='store_true',
        help="remake constraint plot (fig 2)")
    parser.add_argument(
        "-plot_ps_diff",
        action='store_true',
        help="remake point-source vs diffuse comparison plot (fig 3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parameter grids
    # n_pbhs = np.array([[80, 24000, 80]]).T
    # m_pbhs = np.array([[0.5, 10, 100]]).T
    n_pbhs = np.array([[1, 1, 10], [80, 24000, 80]]).T
    m_pbhs = np.array([[0.5, 10, 100], [0.5, 10, 100]]).T
    m_dms = np.logspace(1, 4, args.n_m_dms)
    rate_priors = ["J", "LF"]
    # Use 5th percentile for point-estimate for f
    f_percentile = 0.05
    # Compute f for each detection scenario and rate prior
    fs = {}
    for i in range(n_pbhs.shape[0]):
        for j in range(n_pbhs.shape[1]):
            m_pbh = m_pbhs[i, j]
            n_pbh = n_pbhs[i, j]

            for prior in rate_priors:
                fs[(m_pbh, n_pbh, prior)] = Distribution_f(
                    m_pbh, n_pbh, prior).ppf(f_percentile)

    if args.plot_ps_diff:
        print("Plotting diffuse and point-source limits for different rate "
              "priors")
        plot_ps_diff(m_pbhs, n_pbhs, m_dms, fs, rate_priors)

    # Constraint plot
    if args.plot_limits:
        print("Plotting annihilation cross section limits")
        plot_limits(m_pbhs, n_pbhs, m_dms, fs, rate_prior="LF")

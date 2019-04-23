from argparse import ArgumentParser
import numpy as np
from posterior_inference_point import PointSourcePosterior
from scipy.integrate import cumtrapz
from constants import n_u_0
import matplotlib.pylab as plt


"""
Script for running the Bayesian analysis. Creates tables in data/p_gammas/ and
data/posteriors_sv/.

Notes
-----
* The required directory structure is the following:
    - data/p_gammas/: for storing p_gamma tables computed by the point source
      MC.
    - data/posteriors_f/: posteriors p(f|m_pbh, n_pbh) for GW experiments and
      SKA.
    - data/posteriors_sv/: for storing <sigma v> posterior tables.
* Other required files:
    - constants.py: constants and utility functions.
    - pbhhalosim.py: class for point source MC.
    - posterior_inference_point.py: class for posterior inference for <sigma v>
      (using Point sources).

Command line arguments
----------------------
-m_pbh : float
-n_pbh : int
-o, --overwrite_p_gamma : bool
    If True, the p_gamma table will be recomputed, overwriting the current file
    if it exists.
-n_samples : int
-log10_m_dm_min : float
-log10_m_dm_max : float
-n_m_dm : int
-log10_sv_min : float
-log10_sv_max : float
-n_sv : int
-v, --verbose : bool
"""


def parse_args():
    parser = ArgumentParser()
    # Required
    parser.add_argument("-m_pbh", type=float, required=True, help="PBH mass, M_sun")
    parser.add_argument("-n_pbh", type=int, required=True, help="number of detected PBHs")
    # Optional
    parser.add_argument("-n_u", "--n_unassociated", type=int, default=n_u_0, help="number of unassociated point sources")
    # parser.add_argument("-o", "--overwrite_p_gamma", type=bool, default=True, help="if True, overwrites existing p_gamma tables")
    parser.add_argument("-n_samples", default=100000, type=int, help="number of MC samples to use for p_gamma calculation")
    parser.add_argument("-m_dm_min", default=1e1, type=float, help="minimum DM mass in GeV")
    parser.add_argument("-m_dm_max", default=1e4, type=float, help="maximum DM mass in GeV")
    parser.add_argument("-n_m_dm", default=200, type=int, help="number of DM masses")
    parser.add_argument("-sv_min", default=1e-45, type=float, help="minimum <sigma v> in cm^3/s")
    parser.add_argument("-sv_max", default=1e-25, type=float, help="maximum <sigma v> in cm^3/s")
    parser.add_argument("-n_sv", default=200, type=int, help="number of <sigma v> values")
    parser.add_argument("-v", "--verbose", default=True, type=bool, help="if True, prints progress messages")
    parser.add_argument("-p_R", "--merger_rate_prior", default="LF", type=str, help="prior for PBH merger rate: 'LF' or 'J'")
    parser.add_argument("-p_lambda", "--lambda_prior", default="LF", type=str, help="prior for lambda: 'LF' or 'U'")
    parser.add_argument("-p_sv", "--sv_prior", default="U", type=str, help="prior for <sigma v>: 'U' or 'LF'")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Generate outfile names for plots...
    filestr = ("_ps_M={:.1f}_N={:d}_prior_rate={}_prior_sv={}_"
               "prior_lambda={}.pdf").format(args.m_pbh, args.n_pbh,
                                             args.merger_rate_prior,
                                             args.lambda_prior, args.sv_prior)

    # Setup
    # The priors are set here, and default to the conservative choices
    post_ps = PointSourcePosterior(
        m_pbh=args.m_pbh,
        n_pbh=args.n_pbh,
        merger_rate_prior=args.merger_rate_prior)

    # Make the p_gamma table
    if args.verbose:
        print("Making p_gamma table")
    m_dms = np.geomspace(args.m_dm_min, args.m_dm_max, args.n_m_dm)
    svs = np.geomspace(args.sv_min, args.sv_max, args.n_sv)

    print(m_dms)
    print(svs)
    post_ps.save_p_gamma_table(m_dms, svs)

    # Compute posterior for <sigma v>
    if args.verbose:
        print("Computing posterior for <sigma v>")
    post_ps.save_posterior_table(svs, m_dms, "trapz")

    if args.verbose:
        print("Generating some plots")

    # Load results
    svs, m_dms, un_post_vals = post_ps.load_posterior(normalized=False)
    _, _, post_vals = post_ps.load_posterior(normalized=True)

    # Plot posterior(<sigma v>) and its CDF to make sure the <sigma v>
    # grid isn't clipping anything
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i, m_dm in enumerate(m_dms):
        ax.plot(svs, post_vals[:, i],
                label=r"$m_{\mathrm{DM}} = %.1f$ GeV" % m_dm)
    ax.set_ylabel("PDF")

    ax = axes[1]
    for i, m_dm in enumerate(m_dms):
        ax.plot(svs[1:], cumtrapz(post_vals[:, i], svs),
                label=r"$m_{\mathrm{DM}} = %.1f$ GeV" % m_dm)
    ax.set_ylabel("CDF")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_xlabel(r"$\langle \sigma v \rangle$ (cm$^3$/s)")
        ax.legend()

    plt.savefig("figures/sv_bounds/monitoring" + filestr, bbox_inches='tight')

    # Limit plot
    plt.figure()

    # Load the unnormalized posterior table
    svs, m_dms, un_post_vals = post_ps.load_posterior(normalized=False)

    # Compute 95% upper limit on <sigma v>
    m_dms, sv_bounds = post_ps.sv_bounds(save=False)
    plt.plot(m_dms, sv_bounds, 'r')

    # Plot unnormalized posterior values
    m_dm_mg, sv_mg = np.meshgrid(m_dms, svs)
    post_pcmesh = plt.pcolormesh(m_dm_mg, sv_mg, un_post_vals,
                                 edgecolor="face")
    plt.colorbar(post_pcmesh)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(m_dms[[0, -1]])
    plt.ylim(svs[[0, -1]])
    plt.savefig("figures/sv_bounds/post_sv" + filestr, bbox_inches='tight')

    if args.verbose:
        print("Done computing")

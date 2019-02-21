from argparse import ArgumentParser
import numpy as np
from posterior_inference import save_p_gamma_table, load_p_gamma
from posterior_inference import load_p_f_gw, save_posterior_table


"""
Script for running the Bayesian analysis. Creates tables in data/p_gammas/ and
data/posteriors_sv/.

Notes
-----
* The required directory structure is the following:
    - data/p_gammas/: for storing p_gamma tables computed by the point source MC.
    - data/posteriors_f/: posteriors p(f|m_pbh, n_pbh) for GW experiments and SKA.
    - data/posteriors_sv/: for storing <sigma v> posterior tables.
* Cuts on |b| and Phi and the value of N_U are the defaults from constants.py.
* The default grid over <sigma v> works for m_pbh = 0.2, ..., 10 with n_det =
1, 10 and the default grid of DM masses. It has *not* been tested with n_det =
100, or the SKA posteriors for f that Daniele is going to provide.
* Other required files:
    - constants.py: constants and utility functions.
    - pbhhalosim.py: class for point source MC.
    - posterior_inference.py: functions for building the posterior for <sigma v>.

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
    parser.add_argument("-m_pbh", type=float, required=True, help="PBH mass, M_sun")
    parser.add_argument("-n_pbh", type=int, required=True, help="Number of detected PBHs")
    parser.add_argument("-o", "--overwrite_p_gamma", type=bool, default=True, help="If True, overwrites existing p_gamma tables")
    parser.add_argument("-n_samples", default=100000, type=int, help="Number of MC samples to use for p_gamma calculation")
    parser.add_argument("-log10_m_dm_min", default=1, type=float, help="log10 of minimum DM mass in GeV")
    parser.add_argument("-log10_m_dm_max", default=4, type=float, help="log10 of maximum DM mass in GeV")
    parser.add_argument("-n_m_dm", default=200, type=int, help="Number of DM masses")
    parser.add_argument("-log10_sv_min", default=-45, type=float, help="log10 of minimum <sigma v> in cm^3/s")
    parser.add_argument("-log10_sv_max", default=-25, type=float, help="log10 of maximum <sigma v> in cm^3/s")
    parser.add_argument("-n_sv", default=200, type=int, help="Number of <sigma v> values")
    parser.add_argument("-v", "--verbose", default=True, type=bool, help="If True, prints progress messages")
    return parser.parse_args()


def save_and_load_p_gammas(args):
    """Creates and loads the p_gamma table.

    Returns
    -------
    m_dms, svs, p_gamma
    """
    if args.verbose:
        print("Making new p_gamma table")
    m_dms = np.logspace(args.log10_m_dm_min, args.log10_m_dm_max, args.n_m_dm)
    svs = np.logspace(args.log10_sv_min, args.log10_sv_max, args.n_sv)
    save_p_gamma_table(args.m_pbh, m_dms, svs, args.n_samples)

    if args.verbose:
        print("Loading new p_gamma table")
    return load_p_gamma(args.m_pbh)


if __name__ == '__main__':
    args = parse_args()

    # Make/load p_gamma
    if not args.overwrite_p_gamma:
        try:
            m_dms, svs, p_gamma, p_gamma_err = load_p_gamma(args.m_pbh)
            if args.verbose:
                print("Using existing p_gamma table")
        except:
            m_dms, svs, p_gamma, p_gamma_err = save_and_load_p_gammas(args)
    else:
        m_dms, svs, p_gamma, p_gamma_err = save_and_load_p_gammas(args)

    # Loads p(f|N_PBH) for GW experiments, computed by Bradley
    if args.verbose:
        print("Loading p(f|n_pbh) for GW experiments")
    p_f = load_p_f_gw(args.m_pbh, args.n_pbh)

    # Compute posterior for <sigma v>
    if args.verbose:
        print("Computing posterior for <sigma v>")
    save_posterior_table(svs, args.n_pbh, p_f, p_gamma, args.m_pbh, m_dms)
    if args.verbose:
        print("Done computing")

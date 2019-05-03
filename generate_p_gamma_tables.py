from argparse import ArgumentParser
import numpy as np

from distributions import Distribution_N_gamma


"""
Creates table containing p_gamma values.

Notes
-----
* The required directory structure is the following:
    - data/p_gammas/: for storing p_gamma tables computed by the point source
      MC.

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
    parser.add_argument("-m_pbh", type=float, required=True,
                        help="PBH mass, M_sun")
    # Optional
    parser.add_argument(
        "-n_samples", default=100000, type=int,
        help="number of MC samples to use for p_gamma calculation")

    parser.add_argument("-m_dm_min", default=1e1, type=float,
                        help="minimum DM mass in GeV")
    parser.add_argument("-m_dm_max", default=1e4, type=float,
                        help="maximum DM mass in GeV")
    parser.add_argument("-n_m_dm", default=200, type=int,
                        help="number of DM masses")

    parser.add_argument("-sv_min", default=1e-44, type=float,
                        help="minimum <sigma v> in cm^3/s")
    parser.add_argument("-sv_max", default=1e-23, type=float,
                        help="maximum <sigma v> in cm^3/s")
    parser.add_argument("-n_sv", default=200, type=int,
                        help="number of <sigma v> values")

    parser.add_argument(
        "-test", action='store_true',
        help="save results to p_gammas/test/ and use fixed random seed")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Making p_gamma table")
    if args.test:
        print("Running in test mode")
        np.random.seed(7)

    m_dms = np.geomspace(args.m_dm_min, args.m_dm_max, args.n_m_dm)
    svs = np.geomspace(args.sv_min, args.sv_max, args.n_sv)

    dist_n_gamma = Distribution_N_gamma(args.m_pbh,
                                        n_samples=args.n_samples,
                                        test=args.test)
    dist_n_gamma.save_p_gamma_table(m_dms, svs)

    print("Done!")

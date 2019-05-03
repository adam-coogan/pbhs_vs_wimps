from argparse import ArgumentParser
import numpy as np

from src.distributions import Distribution_N_gamma


"""
Creates a table containing p_gamma values.

To regenerate the tables used for the calculations in the paper, run:

>>> python generate_p_gamma_tables.py

To see the command line arguments, run:

>>> python generate_p_gamma_tables.py -h

This script requires the directory `data/p_gammas/` to exist, since this is
where the tables are saved. If running with the `-test` flag, the directory
`data/p_gammas/test/` must exist.
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

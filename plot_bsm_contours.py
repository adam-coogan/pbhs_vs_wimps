import numpy as np
from matplotlib import pyplot as plt

"""
Generates a plot with the BSM contours from figures 2 and 3, and shaded regions
within these for which at least one model has f_chi > 10%. Run with:

>>> python plot_bsm_contours.py
"""

if __name__ == "__main__":
    # Since the data files contain -ln(L), this gives 95% CL contours
    level = 3
    # Shade regions with f > 10%
    f_level = 0.1

    # Formatting
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e1, 1e4)
    plt.ylim(1e-43, 1e-23)
    plt.xlabel(r"$m_{\chi}$ (GeV)", fontsize=12)
    plt.ylabel(r"$f_{\chi}{}^4 (\sigma v)_0$ (cm$^3$/s)", fontsize=12)

    neg_lnL_singlet = np.load("data/gambit/contours_SingletDM.npy").T

    # Unpack parameter grid
    svs = 10**neg_lnL_singlet[1:, 0]
    m_dms = 10**neg_lnL_singlet[0, 1:]
    m_dm_mg, sv_mg = np.meshgrid(m_dms, svs)

    # Plot singlet 95% CL contour
    plt.contour(m_dm_mg, sv_mg, neg_lnL_singlet[1:, 1:], levels=[3],
                colors=["b"])

    # Plot f > f_level for singlet model
    fs_singlet = np.load("data/gambit/contours_f_SingletDM.npy").T
    assert np.all(svs == 10**fs_singlet[1:, 0])
    assert np.all(m_dms == 10**fs_singlet[0, 1:])
    plt.contourf(m_dm_mg, sv_mg,
                 fs_singlet[1:, 1:] * (neg_lnL_singlet[1:, 1:] < 3),
                 levels=[f_level, np.inf], colors=["k"], alpha=0.3)

    # Plot 95% CL envelope for other BSM models
    min_neg_lnL = np.inf * np.ones(neg_lnL_singlet[1:, 1:].shape)
    for model in ["CMSSM", "MSSM7", "NUHM1", "NUHM2"]:
        neg_lnL = np.load("data/gambit/contours_{}.npy".format(model)).T

        assert np.all(svs == 10**neg_lnL[1:, 0])
        assert np.all(m_dms == 10**neg_lnL[0, 1:])

        min_neg_lnL = np.min([min_neg_lnL, neg_lnL[1:, 1:]], axis=0)

    plt.contour(m_dm_mg, sv_mg, min_neg_lnL, levels=[3], colors=["r"])

    # Maximum f over all but the singlet model
    f_max = np.zeros(neg_lnL[1:, 1:].shape)
    for model in ["CMSSM", "MSSM7", "NUHM1", "NUHM2"]:
        fs = np.load("data/gambit/contours_f_{}.npy".format(model)).T
        assert np.all(svs == 10**fs[1:, 0])
        assert np.all(m_dms == 10**fs[0, 1:])
        f_max = np.max([f_max, fs[1:, 1:]], axis=0)

    # Plot f > f_level for other BSM models
    plt.contourf(m_dm_mg, sv_mg, f_max * (min_neg_lnL < 3),
                 levels=[f_level, np.inf], colors=["g"], alpha=0.3)
    # Manually construct legend
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ["b", "k", "r", "g"]]
    legend_labels = ["Singlet 95% CL",
                     r"Singlet, $f_\chi > " + "{}".format(f_level) + r"$",
                     "BSM 95% CL",
                     r"$BSM, f_\chi > " + "{}".format(f_level) + r"$"]
    plt.legend(proxies, legend_labels, loc="lower right")

    plt.show()

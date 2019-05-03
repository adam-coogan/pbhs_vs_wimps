# Constraining weakly interacting massive particles with primordial black holes

## Reproducing results

The figures can be remade with the following commands:
* Figure 1:
    `python plot_pbh_fraction.py`
* Figure 2:
    `python plot_frequentist_limits.py -plot_limits`
* Figure 3:
    `python plot_frequentist_limits.py -plot_ps_diff`

Computing the point-source limits requires making tables containing `p_gamma`, the probability that an individual PBH is detectable by ermi-LAT. The tables used for the analysis in the paper are contained in the directory `data/p_gammas/`. These may take a few hours to recompute. They can be regenerated with
    `python generate_p_gamma_tables.py -test`
With the `-test` flag, the `p_gamma` tables will be written to `data/p_gammas/test/` instead of overwriting the precomputed tables. The script can also be used to generate `p_gamma` tables over different `(m_dm, sv)` grids. 

## `bayesian` branch
The Bayesian branch contains code for performing a Bayesian analysis of the cross section limits.
* The notebook `plot_bayesian_limits.ipynb` generates Bayesian versions of figures 2 and 3 assuming a uniform prior on the cross section.
* `submit_posterior_calcs_diff.py` generates tables of the posterior over a grid of `(m_dm, sv)` points for the diffuse analysis. The posterior is very simple: we marginalize over `f`, assume the background model for the EGB perfectly matches observations, and simply penalize the extragalactic flux from PBHs for exceeding zero using a Gaussian likelihood for each bin.
* The analysis point-source analysis assumes that astrophysical sources can appear in the unassociated point source catalogue with an unknown rate, which we marginalize over assuming a Jeffreys prior. The script `submit_posterior_calcs_ps.py` generates tables of `p_gamma` and the posterior over an `(m_dm, sv)` grid.

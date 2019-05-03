# Code for "Primordial black holes as silver bullets for new physics at the weak scale"

## Reproducing plots

Short description of which scripts to run to regenerate each plot.

## `bayesian` branch
The Bayesian branch contains code for performing a Bayesian analysis of the cross section limits.
* The notebook `plot_bayesian_limits.ipynb` generates Bayesian versions of figures 2 and 3 assuming a uniform prior on the cross section.
* `submit_posterior_calcs_diff.py` generates tables of the posterior over a grid of `(m_dm, sv)` points for the diffuse analysis. The posterior is very simple: we marginalize over `f`, assume the background model for the EGB perfectly matches observations, and simply penalize the extragalactic flux from PBHs for exceeding zero using a Gaussian likelihood for each bin.
* The analysis point-source analysis assumes that astrophysical sources can appear in the unassociated point source catalogue with an unknown rate, which we marginalize over assuming a Jeffreys prior. The script `submit_posterior_calcs_ps.py` generates tables of `p_gamma` and the posterior over an `(m_dm, sv)` grid.

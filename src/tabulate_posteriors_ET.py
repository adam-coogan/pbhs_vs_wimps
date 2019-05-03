""" TabuulatePosteriors_ET.py

Calculate and tabulate posteriors on f_PBH
given N_obs detections with Einstein telescope.

Outputs a .txt to the 'results' folder.

"""

import numpy as np
import matplotlib.pylab as plt

from scipy.special import gamma, loggamma
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import poisson
from scipy.misc import comb

from tqdm import tqdm
import Cosmo
import MergerRate as MR

#---------------------------

import argparse

#Parse the arguments!   
parser = argparse.ArgumentParser(description='...')
#parser.add_argument('-M_PBH','--M_PBH', help='PBH mass in solar masses', type=float, default=10.0)
parser.add_argument('-N_obs','--N_obs', help='Number of observed mergers', type=int, default=1)
parser.add_argument('-prior','--prior', help="Merger rate prior, either 'J' for Jeffrey's prior or 'LF' for log-flat prior", type=str, default="J")

args = parser.parse_args()
#M_PBH = args.M_PBH
N_obs = args.N_obs
prior_type = args.prior

#We fix M_PBH = 10.0 solar mass in this case
M_PBH = 10.0

if (prior_type not in ["J", "LF"]):
    raise ValueError('Prior must be one of : "J", "LF"')

#Prior scales as R^-alpha
if (prior_type == "J"):
    alpha_prior = 0.5
elif (prior_type == "LF"):
    alpha_prior = 1.0

print("Calculating ET posteriors for M_PBH = " + str(M_PBH) + " M_sun; N = " + str(N_obs) + " observed events...")
print("   Prior: ", prior_type)

#--------------------------

z_ET, frac_ET = np.loadtxt("../data/ET_redshift.txt", unpack=True)

# Selection function f(z) for ET
# Depends on luminosity distance d_L
# and matches reported f(z) at z > 40
def selection_ET(z):
    z_hor = z_ET[-1]
    DL_hor = Cosmo.calcdL(z_hor)
    DL = Cosmo.calcdL(z)
    return np.clip((1/DL - 1/DL_hor)*4.5e4, 0, 1)

def calcdVTdz(z, T=0.67):
    dVdz = Cosmo.dVdz(z)*1e-9 #Mpc^3 -> Gpc^3
    return (T*dVdz/(1+z))*selection_ET(z)
    
#print("Fraction detectable at z = 40:", selection_ET(40.0))
    
#--------------------

z_start = 40.0
z_end = 100.0


# Calculate number of detected events
# integrating over redshift 
def CalcLambda(f, M_PBH):
    z_list = np.logspace(np.log10(z_start), np.log10(z_end), 20)

    integ = np.array([calcdVTdz(z, T = 0.67)*MR.MergerRate(z, f, M_PBH, withHalo=True) for z in z_list])
    L1 = np.trapz(integ, z_list)
    return L1


#Adam and I verified this - we spent roughly 1 hour worrying about whether this was
#Bayes enough. It was sufficiently Bayes. Do not worry in future. 
#        ||
#        ||
#        ||
#        VV
#P(R_eff)
def Posterior(R_eff, N, VT=1.0):
    lam = VT*R_eff

    prior = lam**-alpha_prior
    
    p_poiss = poisson.pmf(N, lam)
    lgamma = loggamma(N + 1 - alpha_prior)
    A = (N-alpha_prior)*np.log(lam) - lam - lgamma

    return VT*np.exp(A)


def calcCredible(R, P):
    cum_dist =  cumtrapz(P, R, initial=0) 
    inv_cum = interp1d(cum_dist, R)
    min90 = inv_cum(0.05)
    med = inv_cum(0.5)
    max90 = inv_cum(0.95)
    upper90 = inv_cum(0.90)
    return min90, med, max90, upper90

def round_to(x, nearest=10):
    return int(round(x / nearest) * nearest)
    
def P_sensitivity(VT, VT_0):
    sig = 0.30
    lVT = np.log(VT)
    lVT_0 = np.log(VT_0)
    P = (1/VT)*(2*np.pi*sig**2)**-0.5*np.exp(-(lVT-lVT_0)**2/(2*sig**2))
    return np.nan_to_num(P)

def P_marginal(R_eff, N):

    integ = lambda x: Posterior(R_eff, N, x)*P_sensitivity(x, VT_0=1.0)    
    return quad(integ, 0, 5.0, points = np.logspace(-5,1.1,20))[0]

def get_f_interval(N):

    R_list = np.logspace(-1, 8, 1000)
    P_marg_list = np.array([P_marginal(R, N) for R in R_list])
    
    minR, medR, maxR, upper90 = calcCredible(R_list, P_marg_list)
    
    return f_of_R(minR), f_of_R(medR), f_of_R(maxR), f_of_R(upper90)

#-------------------------------
N_f = 200
f_list = np.logspace(-6, 0, N_f)

def get_posterior(N, M_PBH):
    
    R_list = np.zeros(N_f)
    for i in tqdm(range(N_f)):
        R_list[i] = CalcLambda(f_list[i], M_PBH)
    P_marg_list = np.array([P_marginal(R, N) for R in R_list])

    
    R_of_f = UnivariateSpline(f_list, R_list, k=4, s=0)
    dR_df = R_of_f.derivative()
    P_of_f = P_marg_list*dR_df(f_list)
    
    return np.nan_to_num(P_of_f)

    
P = get_posterior(N_obs, M_PBH)
    
print("Posterior is normalised to:", np.trapz(P, f_list))

f_lower  = calcCredible(f_list, P)[0]
print("95% Upper limit on f:", f_lower)

mstr = str(round(M_PBH, 1))
Nstr = str(N_obs)

htxt = "Posterior distribution for f, given N = " + Nstr + " merger events in ET. PBH Mass: " + mstr + " M_sun"
htxt += "Prior: " + prior_type
htxt += "\nColumns: f, P(f|N)"

np.savetxt("../data/posteriors_f/Posterior_f_ET_Prior_" + prior_type + "_M=" + mstr+"_N=" + Nstr + ".txt", list(zip(f_list, P)), header=htxt)
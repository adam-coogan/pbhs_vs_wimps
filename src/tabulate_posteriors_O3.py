""" TabuulatePosteriors_O3.py

Calculate and tabulate posteriors on f_PBH
given N_obs detections with LIGO/Virgo O3.

Outputs a .txt to the 'results' folder.

"""


import numpy as np
import matplotlib.pylab as plt

from scipy.special import gamma, loggamma
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d, UnivariateSpline

from tqdm import tqdm
import Cosmo
import MergerRate as MR

from scipy.stats import poisson
#---------------------------

import argparse

#Parse the arguments!   
parser = argparse.ArgumentParser(description='Calculate and tabulate posteriors on f_PBH given N_obs detections with LIGO O3. Outputs a .txt file to the ../data/posteriors_f folder.')
parser.add_argument('-M_PBH','--M_PBH', help='PBH mass in solar masses (0.2 -> 1.0)', type=float, default=0.5)
parser.add_argument('-N_obs','--N_obs', help='Number of observed mergers', type=int, default=1)
parser.add_argument('-prior','--prior', help="Merger rate prior, either 'J' for Jeffrey's prior or 'LF' for log-flat prior", type=str, default="J")


args = parser.parse_args()
M_PBH = args.M_PBH
N_obs = args.N_obs
prior_type = args.prior

if ((M_PBH < 0.2) or (M_PBH > 1.0)):
    raise ValueError('M_PBH must be between 0.2 and 1.0 (Solar Masses)')

if (prior_type not in ["J", "LF"]):
    raise ValueError('Prior must be one of : "J", "LF"')

#Prior scales as R^-alpha
if (prior_type == "J"):
    alpha_prior = 0.5
elif (prior_type == "LF"):
    alpha_prior = 1.0

print("   Calculating LIGO posteriors for M_PBH = " + str(M_PBH) + " M_sun; N = " + str(N_obs) + " observed events...")
print("   Prior: ", prior_type)

#--------------------------

#LIGO O3 sensitive time-volume
VT_O3 = 1.8e-4 #Gpc^3 yr

#Load in horizon distances and increase by 50% for O3
m_list, horizon_list = np.loadtxt("../data/SubSolar_Horizon.txt", unpack=True)
horizon_interp = interp1d(m_list, 1.5*horizon_list, bounds_error=True)

#Calculate maximum redshift of observations
z_list = np.linspace(0, 1, 20)
dL_list = np.asarray([Cosmo.calcdL(z) for z in z_list])
z_of_dL = interp1d(dL_list, z_list)

#Horizon redshift
z_max = z_of_dL(horizon_interp(M_PBH))
#--------------------
    


#P(R_eff)
def Posterior(R_eff, N, VT=VT_O3):
    lam = VT*R_eff
    #print(lam)
    prior = lam**-alpha_prior
    
    p_poiss = poisson.pmf(N, lam)
    lgamma = loggamma(N + 1 - alpha_prior)
    A = (N-alpha_prior)*np.log(lam) - lam - lgamma

    return VT*np.exp(A)
    

def calcCredible(x, y):
    cum_dist =  cumtrapz(y, x, initial=0) 
    inv_cum = interp1d(cum_dist, x)
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
    return (1/VT)*(2*np.pi*sig**2)**-0.5*np.exp(-(lVT-lVT_0)**2/(2*sig**2))

def P_marginal(R_eff, N):
    integ = lambda x: Posterior(R_eff, N, x)*P_sensitivity(x, VT_0=VT_O3)
    return quad(integ, 0, 5.0, points = np.logspace(-3,1.1,10))[0]

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
        R_list[i] = MR.AverageMergerRate( f_list[i], M_PBH, z1 = 0, z2 = z_max, withHalo=True)
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

htxt = "Posterior distribution for f, given N = " + Nstr + " merger events at LIGO O3. PBH Mass: " + mstr + " M_sun. "
htxt += "Prior: " + prior_type
htxt += "\nColumns: f, P(f|N)"

np.savetxt("../data/posteriors_f/Posterior_f_O3_Prior_" + prior_type + "_M=" + mstr+"_N=" + Nstr + ".txt", list(zip(f_list, P)), header=htxt)

""" Cosmo.py

Functions/constants for calculating cosmological quantities
such as the Hubble rate and various distances.

"""

import numpy as np

from scipy.integrate import cumtrapz, quad, dblquad
from scipy.interpolate import interp1d

#--- Some constants ------------
#-------------------------------

G_N = 4.302e-3 #(pc/solar mass) (km/s)^2
G_N_Mpc = 1e-6*4.302e-3 #(Mpc/solar mass) (km/s)^2

h = 0.678
Omega_DM = 0.1186/(h**2)
H0 = 100.0*h #(km/s) Mpc^-1
H0_peryr = 67.8*(3.24e-20)*(60*60*24*365)
ageUniverse = 13.799e9 #y
Omega_L = 0.692
Omega_m = 0.308
Omega_r = 9.3e-5

D_H = (3000/h) #Mpc

c = 3e5 #km/s


#-------------------------------

#--- Useful cosmological functions ---
#-------------------------------------


rho_critical = 3.0*H0**2/(8.0*np.pi*G_N_Mpc) #Solar masses per Mpc^3

def Hubble(z):
    return H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def Hubble2(z):
    return H0*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def HubbleLaw(age):
    return H0_peryr*age

def rho_z(z):
    return 3.0*Hubble2(z)**2/(8*np.pi*G_N)

def t_univ(z):
    integ = lambda x: 1.0/((1+x)*Hubble(x))
    return quad(integ, z, np.inf)[0]

def Omega_PBH(f):  
    return f*Omega_DM

#Luminosity distance (Mpc)
def calcdL(z):
    c = 3.06594845e-7
    return c*(1+z)*quad(lambda x: Hubble(x)**-1, 0, z)[0]
    
#https://arxiv.org/pdf/astro-ph/9905116.pdf
#Comoving distance (in a flat universe, in Mpc)
def D_C(z):
    return c*quad(lambda z: 1/Hubble2(z), 0, z)[0]
    
#Comoving volume out to redshift z (in Mpc^3)
def V_C(z):
    return (4*np.pi/3)*(D_C(z))**3
    
def dVdz(z):
    return 4*np.pi*D_C(z)**2*(c/Hubble2(z))
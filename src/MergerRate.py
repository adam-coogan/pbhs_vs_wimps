"""MergerRate.py

Code for calculating the PBH merger rate as a function
of PBH mass and fraction (along with Remapping.py).
Note: some functions are replicated between the two files.

Adapted from https://github.com/bradkav/BlackHolesDarkDress. 

See https://arxiv.org/abs/1805.09034 for more details.

Throughout withHalo=True (withHalo=False) performs the calculation
with (without) including Dark Matter halos formed around the PBHs. 

"""

import numpy as np

from scipy.integrate import cumtrapz, quad, dblquad
from scipy.interpolate import interp1d, UnivariateSpline

import Cosmo

import Remapping as Remap

from matplotlib import pyplot as plt

#--- Some constants ------------
#-------------------------------

G_N = 4.302e-3 #(pc/solar mass) (km/s)^2
G_N_Mpc = 1e-6*4.302e-3 #(Mpc/solar mass) (km/s)^2


z_eq = 3375.0
rho_eq = 1512.0 #Solar masses per pc^3
sigma_eq = 0.005 #Variance of DM density perturbations at equality
lambda_max = 3.0 #Maximum value of lambda = 3.0*z_dec/z_eq (i.e. binaries decouple all the way up to z_dec = z_eq)

alpha = 0.1


#------------------------


#Mean interPBH separation
def xbar(f, M_PBH):
    return (3.0*M_PBH/(4*np.pi*rho_eq*(0.85*f)))**(1.0/3.0)
    
    
#Semi-major axis as a function of decoupling redshift
def semimajoraxis(z_pair, f, M_PBH):
    Mtot = M_PBH
    X = 3.0*z_eq*0.85*f/z_pair
    return alpha*xbar(f, M_PBH)*(f*0.85)**(1.0/3.0)*((X/(0.85*f))**(4.0/3.0))
    

def bigX(x, f, M_PBH):
    return (x/(xbar(f,M_PBH)))**3.0
    
#Calculate x (comoving PBH separation) as a function of a
def x_of_a(a, f, M_PBH, withHalo = False):
    
    xb = xbar(f, M_PBH)
    
    if (not withHalo):        
        return ((a * (0.85*f) * xb**3)/alpha)**(1.0/4.0)
    
    elif (withHalo):                                                              
        xb_rescaled = xb * ((M_PBH + Remap.M_halo(z_decoupling(a, f, M_PBH), M_PBH))/M_PBH )**(1./3.)            
        return ((a * (0.85*f) * xb_rescaled**3)/alpha)**(1.0/4.0)
    
#Calculate a (semi-major axis) in terms of x
def a_of_x(x, f, M_PBH):
    
    xb = xbar(f, M_PBH)
    return (alpha/(0.85*f))*x**4/xb**3    
    
#Maximum semi-major axis
def a_max(f, M_PBH, withHalo = False):
    Mtot = 1.0*M_PBH
    if (withHalo):
        Mtot += M_halo(z_eq, M_PBH)
    return alpha*xbar(f, Mtot)*(f*0.85)**(1.0/3.0)*((lambda_max)**(4.0/3.0))
    
def z_decoupling(a, f, mass):
    return (1. + z_eq)/(1./3 * bigX(x_of_a(a, f, mass), f, mass)/(0.85*f)) - 1.
    

def n_PBH(f, M_PBH): 
    return (1e3)**3*Cosmo.rho_critical*Cosmo.Omega_PBH(f)/M_PBH #PBH per Gpc^3
    

    
#------ Probability distributions ----
#-------------------------------------

def j_X(x, f, M_PBH):
    return bigX(x, f, M_PBH)*0.5*(1+sigma_eq**2/(0.85*f)**2)**0.5

def P_j(j, x, f, M_PBH):
    y = j/j_X(x, f, M_PBH)
    return (y**2/(1+y**2)**(3.0/2.0))/j

def P_a_j(a, j, f, M_PBH):
    xval = x_of_a(a, f, M_PBH)
    X = bigX(xval, f, M_PBH)
    xb = xbar(f, M_PBH)
    measure = (3.0/4.0)*(a**-0.25)*(0.85*f/(alpha*xb))**0.75
    return P_j(j, xval, f, M_PBH)*np.exp(-X)*measure
    
def P_a_j_withHalo(a, j, f, M_PBH):
    
    xval = x_of_a(a, f, M_PBH, withHalo = True)
    X = bigX(xval, f, M_PBH)
    xb = xbar(f, M_PBH)
    
    measure = (3.0/4.0)*(a**-0.25)*(0.85*f/(alpha*xb))**0.75    
    measure *= ((M_PBH + Remap.M_halo(z_decoupling(a, f, M_PBH), M_PBH))/M_PBH )**(3./4.)
    
    return P_j(j, xval, f, M_PBH)*np.exp(-X)*measure
    
def P_la_lj(la,lj, f, M_PBH):
    j = 10.**lj
    a = 10.**la
    return P_a_j(a, j, f, M_PBH)*a*j*(np.log(10)**2) #/Norm1
    
def P_la_lj_withHalo(la,lj, f, M_PBH):
    j = 10.**lj
    a = 10.**la
    return P_a_j_withHalo(a, j, f, M_PBH)*a*j*(np.log(10)**2) #/Norm1
    
def t_coal(a, e, M_PBH):
    Q = (3.0/170.0)*(G_N*M_PBH)**(-3) # s^6 pc^-3 km^-6
    tc = Q*a**4*(1-e**2)**(7.0/2.0) #s^6 pc km^-6
    tc *= 3.086e+13 #s^6 km^-5
    tc *= (3e5)**5 #s
    return tc/(60*60*24*365) #in years

def j_coal(a, t, M_PBH):
    Q = (3.0/170.0)*(G_N*M_PBH)**-3 # s^6 pc^-3 km^-6
    tc = t*(60*60*24*365)
    tc /= (3e5)**5
    tc /= 3.086e+13
    return (tc/(Q*a**4))**(1.0/7.0)
    
def P_binary(f, M_PBH):
    amin = 0
    amax = a_max(f, M_PBH)

    P1 = lambda y,x,f,M_PBH: P_a_j(x, y, f, M_PBH)
    Norm1 =  dblquad(P1, amin, amax, lambda x: 0,  lambda x: 1, args=(f, M_PBH), epsrel=1e-20)[0]
    return Norm1
    

#---- Merger Time Distribution ---
#---------------------------------    

def P_t_integ(a, t, f, M_PBH, withHalo=False):
        
    c = 3.e5 #km/s
    Q = (c**6)*(3.0/170.0)*(G_N*M_PBH)**-3 # pc^-3
    t_pc = t*(60*60*24*365)*c*3.24078e-14 #Time in years -> Time in parsec    
    ecc_sq = 1-(t_pc*1.0/(Q*a**4))**(2.0/7.0)
    if (ecc_sq < 0):
        return 0
    ecc = np.sqrt(ecc_sq)
    j_ecc = np.sqrt(1. - ecc**2.)
    
    P1 = 1.
    if (withHalo == False):
        P1 = P_a_j(a, j_ecc, f, M_PBH)
    else:
        P1 = P_a_j_withHalo(a, j_ecc, f, M_PBH)
    
    djdt = j_ecc/(7*t)
    return P1*djdt

#Time in years
def P_of_t_analytical(t, f, M_PBH, withHalo=False): 
        
    amin = 1e-6
    amax = a_max(f, M_PBH)

        
    avals = np.logspace(np.log10(amin), np.log10(amax), 200) #pc
    test = np.asarray([P_t_integ(a, t, f, M_PBH, withHalo) for a in avals])
    
    integr = np.trapz(test, avals)


    return integr
    
    
def P_of_t_withHalo(t, f, M_PBH):
    
    
    do_plots = False
    #do_plots = True
    
    N_a = 500
    
    amin = 1e-6
    amax = Remap.semimajoraxis_full(z_eq, f, M_PBH)
    
    a_i = np.logspace(np.log10(amin), np.log10(amax), N_a)
    a_f = np.vectorize(Remap.calc_af)(a_i, M_PBH)
    
    
    ind_cut = np.argmax(a_f)
    a_cut = a_i[ind_cut]
    
    a_f_interp = UnivariateSpline(a_i, a_f, k=3, s=0)
    daf_dai = a_f_interp.derivative(n=1)

    Jac = np.sqrt(a_f/a_i)*(1/daf_dai(a_i) - 0.0*a_i/a_f)
        
    P = a_f*0.0
    
    c = 3.e5 #km/s
    Q = (c**6)*(3.0/170.0)*(G_N*M_PBH)**-3 # pc^-3
    t_pc = t*(60*60*24*365)*c*3.24078e-14 #Time in years -> Time in parsec    
    ecc_sq = 1-(t_pc*1.0/(Q*a_f**4))**(2.0/7.0)
    
    for i in range(N_a):
        if (ecc_sq[i] < 0):
            P[i] = 0
        else:
            ecc = np.sqrt(ecc_sq[i])
            j_ecc = np.sqrt(1. - ecc**2.)
            j_i = np.sqrt(a_f[i]/a_i[i])*j_ecc
    
            P[i] = P_a_j_withHalo(a_i[i], j_i, f, M_PBH)
    
            djdt = j_ecc/(7*t)
            P[i] *= djdt
    
    
    if (do_plots):
        plt.figure()
    
        plt.loglog(a_i[:ind_cut],a_f[:ind_cut])
        plt.loglog(a_i[ind_cut:],a_f[ind_cut:])
    
        plt.xlabel(r"Initial semi-major axis, $a_i\,\,[\mathrm{pc}]$")
        plt.ylabel(r"Final semi-major axis, $a_f \,\,[\mathrm{pc}]$")

        plt.loglog([1e-4, 4e-1], [1e-4, 4e-1], linestyle='--', color='k')

        plt.xlim(1e-6, 4e-1)
        plt.ylim(1e-6, 4e-1)
        
        plt.gca().set_aspect('equal')
        ax2 = plt.gca().twinx()
        ax2.loglog(a_i, P)
        
        plt.figure()
        plt.loglog(a_i,  np.abs(np.sqrt(a_f/a_i)*(1/daf_dai(a_i))))
        plt.show()
    
    P *= Jac
    
    if (do_plots):
        plt.figure()
        plt.plot(a_f[:ind_cut],P[:ind_cut])
        plt.plot(a_f[ind_cut:][::-1],-P[ind_cut:][::-1])
        plt.show()

    return np.trapz(P[:ind_cut], a_f[:ind_cut]) - np.trapz(P[ind_cut:][::-1],a_f[ind_cut:][::-1])


#---- Merger Rates ---
#---------------------------------   

def MergerRate_test(z, f, M_PBH):
    return 0.5*n_PBH(f, M_PBH)*P_of_t_analytical(Cosmo.t_univ(z), f, M_PBH, withHalo=True)
    

def MergerRate(z, f, M_PBH, withHalo=False):    
    P_of_t_fun = P_of_t_analytical
    if (withHalo):
        P_of_t_fun = P_of_t_withHalo

    return 0.5*n_PBH(f, M_PBH)*P_of_t_fun(Cosmo.t_univ(z), f, M_PBH)
    
    
def AverageMergerRate( f, M_PBH, z1, z2, withHalo=False):
    z_list = np.linspace(z1, z2, 20)
    Merge_list = np.array([MergerRate(z, f, M_PBH, withHalo) for z in z_list])
    return np.trapz(Merge_list, z_list)/(z2-z1)
    
    



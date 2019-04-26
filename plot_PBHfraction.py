import numpy as np
import matplotlib.pylab as plt

from scipy.special import gamma
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import interp1d, UnivariateSpline


def calcCredible(x, y):
    cum_dist =  cumtrapz(y, x, initial=0) 
    inv_cum = interp1d(cum_dist/cum_dist[-1], x)
    min90 = inv_cum(0.05)
    med = inv_cum(0.5)
    max90 = inv_cum(0.95)
    upper90 = inv_cum(0.90)
    return min90, med, max90, upper90

def round_to(x, nearest=10):
    return int(round(x / nearest) * nearest)


def getPolygon(X, Y1, Y2, YMAX):
    if (Y1[-1] < YMAX):
        Y1 = np.append(Y1, YMAX*1.001)
        Y2 = np.append(Y2, Y2[-1]*1.001)
        X = np.append(X, X[-1]*1.001)
    if (Y2[-1] < YMAX):
        Y1 = np.append(Y1, Y1[-1]*1.001)
        Y2 = np.append(Y2, YMAX*1.001)
        X = np.append(X, X[-1]*1.001)
        
    interp_inv_A = interp1d(np.log10(Y1), np.log10(X))
    interp_inv_B = interp1d(np.log10(Y2), np.log10(X))
    X_A = 10**interp_inv_A(np.log10(YMAX))
    X_B = 10**interp_inv_B(np.log10(YMAX))
    X_A_LIST = np.append(X[X < X_A], X_A)
    X_B_LIST = np.append(X[X < X_B], X_B)
    Y_A_LIST = np.append(Y1[X < X_A], YMAX)
    Y_B_LIST = np.append(Y2[X < X_B], YMAX)
    return np.append(X_A_LIST, X_B_LIST[::-1]), np.append(Y_A_LIST, Y_B_LIST[::-1])
    
def getCutOff(X, Y, YMAX):
    interp_inv = interp1d(np.log10(Y), np.log10(X))
    X_new = 10**interp_inv(np.log10(YMAX))
    X_out = X[Y < YMAX]
    X_out = np.append(X_out, X_new)
    Y_out = Y[Y < YMAX]
    Y_out = np.append(Y_out, YMAX)
    return X_out, Y_out
    
def get_f_intervals(f_list, P_list):
    N = len(P_list)
    f_med = np.zeros(N)
    f_min = np.zeros(N)
    f_max = np.zeros(N)
    
    for i in range(N):
        f_min[i], f_med[i], f_max[i], junk = calcCredible(f_list[i], P_list[i])

    return f_min, f_med, f_max

mstrings = {
    "O3": "0.5",
    "ET": "10.0",
    "SKA": "100.0"
}

Nlists = {
    "O3": np.array([1, 2, 3, 4, 5, 10, 30, 81]),
    "ET": np.array([1, 2, 3, 4, 5, 10, 1000, 10000, 20000, 24000]),
    "SKA": np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
}

lims = {
    "O3": 0.1128, #Figure 3 of https://arxiv.org/pdf/1808.04771.pdf
    "ET": 4.13e-03, #Figure 11 of https://arxiv.org/pdf/1805.09034.pdf
    "SKA": 3.2e-4 #2 sigma radio from https://arxiv.org/pdf/1812.07967.pdf
}

def addToPlot(exp, prior, color, cut=True):
    Nlist = Nlists[exp]
    Mstr = mstrings[exp]
    lim = lims[exp]
    
    f = [np.loadtxt("data/posteriors_f/Posterior_f_" + exp + "_Prior_" + prior + "_M=" + Mstr + "_N=" + str(N) + ".txt",unpack=True)[0] for N in Nlist]
    P = [np.loadtxt("data/posteriors_f/Posterior_f_" + exp + "_Prior_" + prior + "_M=" + Mstr + "_N=" + str(N) + ".txt",unpack=True)[1] for N in Nlist]
        
    if (exp == "SKA"):
        zo = 1.8
    elif (exp == "O3"):
        zo = 1.8
    elif (exp == "ET"):
        zo = 1.5
    
    f_min, f_med, f_max = get_f_intervals(f, P)

    if (cut):
        X, Y = getPolygon(Nlist, f_min, f_max, lim)
        plt.fill(X, Y, color=color, alpha = 0.5, lw=0.0, zorder=zo)
        N_new, f_new = getCutOff(Nlist, f_med, lim)
        
    else:
        plt.fill_between(Nlist, f_min, f_max, color=color, alpha = 0.5, lw=0.0, zorder=zo)
        N_new = 1.0*Nlist
        f_new = 1.0*f_med
        
    plt.loglog(N_new, f_new, '-', color=color, zorder=(zo+0.05))
    return N_new[-1]
    

plt.figure(figsize=(3.5,3))

ax = plt.gca()



N1 = addToPlot("O3", "J", color='C0')
plt.plot([N1*0.4, N1*2.1],[1.02*lims["O3"],1.02*lims["O3"]],linestyle='-', color='dimgray', lw=0.8, zorder=2.0)
plt.fill_between([N1*0.4, N1*2.1], y1 = 1.02*lims["O3"], y2 = 1.5*lims["O3"], facecolor='lightgrey', edgecolor='None', hatch='//////', lw=0.8, zorder=2.0)

N1 = addToPlot("ET", "J", color='C1')
plt.plot([N1*0.45, N1*2.1],[1.02*lims["ET"],1.02*lims["ET"]],linestyle='-', color='dimgray', lw=0.8, zorder=2.0)
plt.fill_between([N1*0.45, N1*2.1], y1 = 1.02*lims["ET"], y2 = 1.5*lims["ET"], facecolor='lightgrey', edgecolor='None', hatch='//////', lw=0.8, zorder=2.0)

N1 = addToPlot("SKA", "J", color='C2', cut=True)
plt.plot([N1*0.65, N1*1.4],[1.02*lims["SKA"],1.02*lims["SKA"]],linestyle='-', color='dimgray', lw=0.8, zorder=2.0)
plt.fill_between([N1*0.65, N1*1.4], y1 = 1.02*lims["SKA"], y2 = 1.5*lims["SKA"], facecolor='lightgrey', edgecolor='None', hatch='//////', lw=0.8, zorder=2.0)


plt.ylim(1e-6, 1)
plt.xlim(1, 5e4)

plt.xlabel("Number of observed PBH candidates, $N_\mathrm{PBH}$")
plt.ylabel(r"PBH fraction $f_\mathrm{PBH}$")


x0 = 0.97
y0 = 0.03
dy = 0.065

plt.text(x0, y0 + 2*dy, r"LIGO O3, $M_\mathrm{PBH} = 0.5 \,M_\odot$", 
                        fontsize=9.0, ha='right', va='bottom', color='C0',transform=ax.transAxes)
plt.text(x0, y0 + dy, r"ET $(z \geq 40)$, $M_\mathrm{PBH} = 10 \,M_\odot$", 
                        fontsize=9.0, ha='right', va='bottom', color='C1',transform=ax.transAxes)
plt.text(x0, y0, r"SKA, $M_\mathrm{PBH} = 100 \,M_\odot$", 
                        fontsize=9.0, ha='right', va='bottom', color='C2',transform=ax.transAxes)

plt.savefig("figures/Constraints_fPBH.pdf", bbox_inches='tight',pad_inches = 0.05)

plt.show()
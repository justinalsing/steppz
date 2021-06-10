import numpy as np
from scipy.integrate import cumtrapz

# implement the Simha+2014 SFH model
def sfh(tuniv, tau, alpha, beta, Z, nsteps=200):
    
    # time-grid
    t = np.concatenate([tuniv * (1. - np.logspace(-3, -1e-10, nsteps))[::-1], np.array([tuniv])])

    # SFH
    logsfr = -np.log((t/tau)**alpha[i] + (tau/t)**beta)
    logsfr = logsfr - max(logsfr)
    sfr = np.exp(logsfr)
    sfr = sfr/np.trapz(x=t, y=sfr)

    # compute the metallicity history based on the integral of the SFH
    zt = np.concatenate([np.array([0.]), cumtrapz(x=t, y=sfr)])
    
    # re-normalize the metalicity history such that the mean metalicity over the last 10% of star formation = target Z
    zt = zt * Z / zt[-1]
        
    return t, sfr*1e-9, zt

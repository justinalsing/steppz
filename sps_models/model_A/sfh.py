import numpy as np
from scipy.special import gammainc
from scipy.integrate import cumtrapz

# implement the Simha+2014 SFH model
def sfh(tuniv, sf_start, tau, sf_trunc, sf_slope, Z, nsteps=200):
    
    # age
    tage = tuniv - sf_start
    
    # time from beginning of star formation until truncation
    sf_trunc_ = sf_trunc - sf_start
    
    # sfr at the truncation time
    sfr_trunc = sf_trunc_ * np.exp(- sf_trunc_ / tau) # star formation rate at the truncation time

    # end of star formation
    sf_end = tuniv * (sf_slope >= 0) + min(sf_start + (sf_trunc_ - sfr_trunc/sf_slope), tuniv) * int(sf_slope < 0)
    sf_end_ = sf_end - sf_start # time to end of star formation from the beginning of star formation
    
    # set up the time-grid based on the start and end times
    t = np.concatenate([np.array([0.]), np.flip((sf_end + sf_start) - np.logspace(np.log10(sf_start), np.log10(sf_end), nsteps)), np.array([tuniv + 1e-10])])
    
    # normalization (such that integrated SFH = 1.)
    norm = (tau**2 * gammainc(2, sf_trunc_/tau)) + sfr_trunc*(sf_end_ - sf_trunc_) + 0.5*sf_slope*(sf_end_ - sf_trunc_)**2
        
    # star formation rate
    sfr = np.clip(((t-sf_start) * np.exp(-(t-sf_start)/tau) * (t <= sf_trunc)  + (sfr_trunc + sf_slope*(t - sf_trunc))*(t > sf_trunc) ) / norm, 0, np.inf)

    # compute the metallicity history based on the integral of the SFH
    zt = np.concatenate([np.array([0.]), cumtrapz(x=t, y=sfr)])
    
    # re-normalize the metalicity history such that the mean metalicity over the last 10% of star formation = target Z
    zt = zt * Z / zt[-1]
        
    return t, sfr*1e-9, zt

# compute sSFRs from thetas
def compute_sSFR(theta, tuniv):
    
    nsamples = theta.shape[0]
    
    # parameters
    f_sf_start = theta[:,0] 
    lntau = theta[:,1]
    f_sf_trunc = theta[:,2]
    sf_slope_phi = theta[:,3]
    
    # re-parameterize to Simha+ parameters
    sf_start = tuniv*f_sf_start
    sf_trunc = sf_start + f_sf_trunc * (tuniv - sf_start)
    tau = np.exp(lntau)
    sf_slope = np.tan(sf_slope_phi)

    # time from beginning of star formation until truncation
    sf_trunc_ = sf_trunc - sf_start

    # sfr at the truncation time
    sfr_trunc = sf_trunc_ * np.exp(- sf_trunc_ / tau) # star formation rate at the truncation time

    # end of star formation
    sf_end = tuniv*(sf_slope >= 0).astype(np.int64) + np.minimum(sf_start + (sf_trunc_ - sfr_trunc/sf_slope), tuniv*np.ones(nsamples)) * (sf_slope < 0).astype(np.int64)
    sf_end_ = sf_end - sf_start # time to end of star formation from the beginning of star formation

    # normalization (such that integrated SFH = 1.)
    norm = (tau**2 * gammainc(2, sf_trunc_/tau)) + sfr_trunc*(sf_end_ - sf_trunc_) + 0.5*sf_slope*(sf_end_ - sf_trunc_)**2

    # star formation rate
    sSFR = np.clip((sfr_trunc + sf_slope*(tuniv - sf_trunc)) / norm, 0, np.inf)
    
    return sSFR*1e-9
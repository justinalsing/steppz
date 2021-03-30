import fsps
from astropy.cosmology import Planck15
import numpy as np
from sedpy.observate import load_filters, getSED
import sys
from sfh import *

# constants
lsun = 3.846e33  # erg/s
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc*10)**2) # L_sun/AA to erg/s/cm^2/AA at 10pc

# pull in command line args
parameters_file = sys.argv[1]
spectra_file = sys.argv[2]
photometry_file = sys.argv[3]
index = sys.argv[4]

# load the parameters
theta = np.load(parameters_file + 'parameters' + str(index) + '.npy')

# load the filters
kids_filters = ['omegacam_' + n for n in ['u','g','r','i']]
viking_filters = ['VISTA_' + n for n in ['Z','Y','J','H','Ks']]
filters = load_filters(kids_filters + viking_filters)

# set up the SPS model
model = fsps.StellarPopulation(zcontinuous=1, compute_vega_mags=False)

# set up parameters
model.params['sfh'] = 3 # tabular sfh
model.params['imf_type'] = 1 # Chabrier IMF
model.params['dust_type'] = 4 # Calzetti with power law modification
model.params['pmetals'] = -99

# turn off neublar emission
model.params['add_neb_emission'] = False
model.params['add_neb_continuum'] = False
model.params['nebemlineinspec'] = False

# turn off dust emission (only needed for IR)
model.params['add_dust_emission'] = False

# initialize the model by making a single call
z, log10Z, dust2, dust_index, f_sf_start, lntau, f_sf_trunc, phi = list(theta[0,:])
tuniv = Planck15.age(z).value
sf_start = tuniv * f_sf_start
sf_trunc = tuniv * f_sf_trunc + sf_start
sf_slope = np.tan(phi)
tage = tuniv - sf_start
Z = (10**log10Z)*0.0142
tau = np.exp(lntau)
t, sfr, zh = sfh(tuniv, sf_start, tau, sf_trunc, sf_slope, Z, nsteps=300)

# set parameters
model.params['zred'] = z
model.params['dust2'] = dust2
model.params['dust_index'] = dust_index
model.params['logzsol'] = log10Z
model.set_tabular_sfh(t, sfr)

# set the conditional parameters
model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
model.params['gas_logz'] = log10Z # set to the final metalicity
model.params['dust1'] = dust2

# compute rest-frame spectrum
wave, spec = model.get_spectrum(tage = tuniv)

# save the wave vector to file
np.save(spectra_file + 'wave.npy', wave)

# holder for photometry
photometry = np.zeros((theta.shape[0], len(filters)))

# load in the spectra without nebular emission (but with evolving metallicity history from "generate_spectra.py")
spectra = np.load(spectra_file + 'spectra' + str(index) + '.npy')

# now loop over the training parameters
for i in range(theta.shape[0]):

    # set parameters
    z, log10Z, dust2, dust_index, f_sf_start, lntau, f_sf_trunc, phi = list(theta[i,:])
    tuniv = Planck15.age(z).value
    sf_start = tuniv * f_sf_start
    tage = tuniv - sf_start
    sf_trunc = tage * f_sf_trunc + sf_start
    sf_slope= np.tan(phi)
    tau = np.exp(lntau)
    Z = (10**log10Z)*0.0142
    t, sfr, zh = sfh(tuniv, sf_start, tau, sf_trunc, sf_slope, Z, nsteps=300)

    # with nebular emission...
    model.params['add_neb_emission'] = True
    model.params['add_neb_continuum'] = True
    model.params['nebemlineinspec'] = True

    # set parameters
    model.params['zred'] = z
    model.params['dust2'] = dust2
    model.params['dust_index'] = dust_index
    model.params['logzsol'] = log10Z
    model.set_tabular_sfh(t, sfr)

    # set the conditional parameters
    model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
    model.params['gas_logz'] = log10Z # set to the final metalicity
    model.params['dust1'] = dust2

    # compute rest-frame spectrum
    wave0, spec0 = model.get_spectrum(tage = tuniv)

    # redshifted spectrum (and wavelength grid)
    wave, spec = wave0*(1. + z), spec0*(1. + z)

    # without nebular emission...
    model.params['add_neb_emission'] = False
    model.params['add_neb_continuum'] = False
    model.params['nebemlineinspec'] = False

    # set parameters
    model.params['zred'] = z
    model.params['dust2'] = dust2
    model.params['dust_index'] = dust_index
    model.params['logzsol'] = log10Z
    model.set_tabular_sfh(t, sfr)

    # set the conditional parameters
    model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
    model.params['gas_logz'] = log10Z # set to the final metalicity
    model.params['dust1'] = dust2

    # compute rest-frame spectrum
    wave0, spec0_ = model.get_spectrum(tage = tuniv)

    # add the nebular emission to the previous spectrum
    spectra[i,:] += (spec0 - spec0_)

    # redshift the spectrum and compute the photometry
    wave, spec = wave0*(1. + z), spectra[i,:]*(1.+z)

    # absolute magnitudes
    M = getSED(wave, lightspeed/wave**2 * spec * to_cgs_at_10pc, filters)

    # chalk em up
    photometry[i,:] = M

# save to disc...
np.save(spectra_file + 'spectra' + str(index) + '.npy', spectra)
np.save(photometry_file + 'photometry' + str(index) + '.npy', photometry)

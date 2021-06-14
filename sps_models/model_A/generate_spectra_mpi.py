import os
os.environ["SPS_HOME"] = '/cfs/home/alju5794/software/python-fsps/src/fsps/libfsps'
import fsps
from astropy.cosmology import Planck15
import numpy as np
from sedpy.observate import load_filters, getSED
import sys
from sfh import *
import time

# mpi set-up
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# pull in arguments from command line
root_directory = sys.argv[1]

# load the parameters for the default parameters file
theta = np.load(root_directory + 'parameters/parameters0' + '.npy')

# how many sets to run
n_sets = 64
n_samples = theta.shape[0]

# MPI set-up
sets_per_rank = n_sets // size # how many sets per node?
sets = range(sets_per_rank * rank, sets_per_rank * rank + sets_per_rank) # which sets to run on this node?

# constants
lsun = 3.846e33  # erg/s
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc*10)**2) # L_sun/AA to erg/s/cm^2/AA at 10pc

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

# velocity smoothing
model.params['smooth_velocity'] = True
model.params['sigma_smooth'] = 150
model.params['min_wave_smooth'] = 9e2
model.params['max_wave_smooth'] = 1e5

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
model.set_tabular_sfh(t, sfr, Z=zh)

# set the conditional parameters
model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
model.params['gas_logz'] = log10Z # set to the final metalicity
model.params['dust1'] = dust2

# compute rest-frame spectrum
wave, spec = model.get_spectrum(tage = tuniv)

# save the wave vector to file
np.save(root_directory + 'spectra/wave.npy', wave)

# holders for model spectra
spectra = np.zeros((n_samples, len(wave)))

# loop over sets
start = time.time()
for k in sets:

    # load in the parameters
    theta = np.load(root_directory + 'parameters/parameters' + str(k) + '.npy')

    # holders for model spectra
    spectra = np.zeros((n_samples, len(wave)))

    # now loop over the training parameters
    for i in range(n_samples):

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

        # set parameters
        model.params['zred'] = z
        model.params['dust2'] = dust2
        model.params['dust_index'] = dust_index
        model.set_tabular_sfh(t, sfr, Z=zh)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
        model.params['gas_logz'] = log10Z # set to the final metalicity
        model.params['dust1'] = dust2

        # compute rest-frame spectrum
        wave0, spec0 = model.get_spectrum(tage = tuniv)

        # chalk em up
        spectra[i,:] = spec0

    # save to disc
    np.save(root_directory + 'spectra/spectra' + str(k) + '.npy', spectra)

total_time = time.time() - start
np.savetxt(root_directory + 'timing_benchmark.txt', np.array([total_time]))

import os
os.environ["SPS_HOME"] = '/cfs/home/side0330/fsps'

import fsps
from astropy.cosmology import Planck15
import numpy as np
import pandas as pd
from sedpy.observate import load_filters, getSED
import sys
from sfh_templates import flexible_sfh

# mpi set-up
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# pull in arguments from command line
root_directory = sys.argv[1]

theta_init = np.load(root_directory + 'parameters/parameters0' + '.npy')
n_time_bins = 7
parameter_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_bins)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
"""Populate a pandas dataframe with the parameters array & parameter names"""
parameter_df_init = pd.DataFrame(data=theta_init, columns=parameter_names)
parameter_df_init['ID'] = parameter_df_init.index + 1
parameter_df_init['Z'] = (10**parameter_df_init['gaslog10Z'])*0.0142

# constants
lsun = 3.846e33  # erg/s
pc = 3.085677581467192e18  # in cm
lightspeed = 2.998e18  # AA/s
to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc*10)**2) # L_sun/AA to erg/s/cm^2/AA at 10pc

# how many sets to run
n_sets = 64
n_samples = parameter_df_init.shape[0]

# MPI set-up
sets_per_rank = n_sets // size # how many sets per node?
sets = range(sets_per_rank * rank, sets_per_rank * rank + sets_per_rank) # which sets to run on this node?

# load the filters
kids_filters = ['omegacam_' + n for n in ['u','g','r','i']]
viking_filters = ['VISTA_' + n for n in ['Z','Y','J','H','Ks']]
cosmos15_filters = ['ip_cosmos', 'v_cosmos', 'uvista_y_cosmos', 'r_cosmos', 'hsc_y',
                    'zpp', 'b_cosmos', 'uvista_h_cosmos', 'wircam_H', 'ia484_cosmos',
                    'ia527_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia738_cosmos',
                    'ia767_cosmos', 'ia427_cosmos', 'ia464_cosmos', 'ia505_cosmos',
                    'ia574_cosmos', 'ia709_cosmos', 'ia827_cosmos', 'uvista_j_cosmos',
                    'uvista_ks_cosmos', 'wircam_Ks', 'NB711.SuprimeCam',
                    'NB816.SuprimeCam']
filters = load_filters(kids_filters + viking_filters + cosmos15_filters)

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
param_slice_init = parameter_df_init.iloc[0]
tuniv = Planck15.age(param_slice_init['z'].item()).value

t, sfr, zh = flexible_sfh(param_slice_init)

# set parameters
model.params['zred'] = param_slice_init['z'].item()
model.params['dust2'] = param_slice_init['dust2'].item()
model.params['dust1'] = param_slice_init['dust2'].item() * param_slice_init['dust1_fraction'].item()
model.params['dust_index'] = param_slice_init['dust_index'].item()
model.set_tabular_sfh(t, sfr)

# set the conditional parameters
model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
model.params['gas_logz'] = (10 ** param_slice_init['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

# compute rest-frame spectrum
wave, spec = model.get_spectrum(tage=tuniv)

# save the wave vector to file
spec_output_dir = root_directory + 'spectra'
if not os.path.exists(spec_output_dir):
    os.makedirs(spec_output_dir)

np.save(root_directory + 'spectra/wave.npy', wave)

for iter_ in range(n_sets):

    # load in the parameters and stellar masses
    theta = np.load(root_directory + 'parameters/parameters' + str(iter_) + '.npy')
    stellar_masses = np.load(root_directory + 'spectra/stellar_masses' + str(iter_) + '.npy')

    parameter_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_bins)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
    """Populate a pandas dataframe with the parameters array & parameter names"""
    parameter_df = pd.DataFrame(data=theta, columns=parameter_names)
    parameter_df['ID'] = parameter_df.index + 1
    parameter_df['Z'] = (10 ** parameter_df['gaslog10Z']) * 0.0142

    # holder for photometry
    photometry = np.zeros((n_samples, len(filters)))

    # load in the spectra without nebular emission (but with evolving metallicity history from "generate_spectra.py")
    spectra = np.load(root_directory + 'spectra/spectra' + str(iter_) + '.npy')

    for param_ind_ in range(n_samples):

        param_slice = parameter_df.iloc[param_ind_]

        tuniv = Planck15.age(param_slice['z'].item()).value

        # get the time grid, sfr, and
        t, sfr, zh = flexible_sfh(param_slice)

        # with nebular emission...
        model.params['add_neb_emission'] = True
        model.params['add_neb_continuum'] = True
        model.params['nebemlineinspec'] = True

        # set parameters
        model.params['zred'] = param_slice_init['z'].item()
        model.params['dust2'] = param_slice_init['dust2'].item()
        model.params['dust1'] = param_slice_init['dust2'].item() * param_slice_init['dust1_fraction'].item()
        model.params['dust_index'] = param_slice_init['dust_index'].item()
        model.set_tabular_sfh(t, sfr)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
        model.params['gas_logz'] = (10 ** param_slice_init['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

        # compute rest-frame spectrum
        wave0, spec0 = model.get_spectrum(tage=tuniv)

        # redshifted spectrum (and wavelength grid)
        wave, spec = wave0*(1. + param_slice_init['z'].item()), spec0*(1. + param_slice_init['z'].item())

        # without nebular emission...
        model.params['add_neb_emission'] = False
        model.params['add_neb_continuum'] = False
        model.params['nebemlineinspec'] = False

        # set parameters
        model.params['zred'] = param_slice_init['z'].item()
        model.params['dust2'] = param_slice_init['dust2'].item()
        model.params['dust1'] = param_slice_init['dust2'].item() * param_slice_init['dust1_fraction'].item()
        model.params['dust_index'] = param_slice_init['dust_index'].item()
        model.set_tabular_sfh(t, sfr)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
        model.params['gas_logz'] = (10 ** param_slice_init['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

        # compute rest-frame spectrum
        wave0, spec0_ = model.get_spectrum(tage=tuniv)

        # add the nebular emission to the previous spectrum
        spectra[param_ind_, :] += (spec0 - spec0_)

        # redshift the spectrum and compute the photometry
        wave_, spec_ = wave0*(1. + param_slice_init['z'].item()), spectra[param_ind_, :]*(1.+param_slice_init['z'].item())

        # absolute magnitudes
        M = getSED(wave_, lightspeed/wave_**2 * spec_ * to_cgs_at_10pc, filters)

        # chalk em up
        photometry[param_ind_, :] = M + 2.5*np.log10(stellar_masses[param_ind_]) # stellar mass correction

    phot_output_dir = root_directory + 'photometry'
    if not os.path.exists(phot_output_dir):
        os.makedirs(phot_output_dir)

    # save to disc...
    np.save(root_directory + 'spectra/spectra' + str(iter_) + '.npy', spectra)
    np.save(root_directory + 'photometry/KV_photometry' + str(iter_) + '.npy', photometry[:, 0:9])
    np.save(root_directory + 'photometry/COSMOS15_photometry' + str(iter_) + '.npy', photometry[:, 9:])


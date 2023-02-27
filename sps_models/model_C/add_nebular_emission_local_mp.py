import os
# os.environ["SPS_HOME"] = '/cfs/home/side0330/fsps'
os.environ["SPS_HOME"] = '/home/sinandeger/fsps/'
import fsps
from astropy.cosmology import Planck15
import numpy as np
import pandas as pd
from sedpy.observate import load_filters, getSED
import sys
from tqdm import tqdm
from sfh_templates import flexible_sfh
import threading
import multiprocessing as mp
import re
from tqdm.contrib.concurrent import process_map  # or thread_map

# load the parameters for the default parameters file
# parameters_dir = '/cfs/home/side0330/models/steppz/continuity_prior/parameters'
cwd_ = os.getcwd()
parameters_path = cwd_+'/parameters/'


def generate_fsps_add_neb_spectra(param_file):

    """Specifications related to the fsps model"""
    # constants
    lsun = 3.846e33  # erg/s
    pc = 3.085677581467192e18  # in cm
    lightspeed = 2.998e18  # AA/s
    to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc * 10) ** 2)  # L_sun/AA to erg/s/cm^2/AA at 10pc

    # load the filters
    kids_filters = ['omegacam_' + n for n in ['u', 'g', 'r', 'i']]
    viking_filters = ['VISTA_' + n for n in ['Z', 'Y', 'J', 'H', 'Ks']]
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
    model.params['sfh'] = 3  # tabular sfh
    model.params['imf_type'] = 1  # Chabrier IMF
    model.params['dust_type'] = 4  # Calzetti with power law modification
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
    
    # build the parameter dataframe
    theta_ = np.load(param_file)
    n_time_ = 7
    param_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
    """Populate a pandas dataframe with the parameters array & parameter names"""
    param_df = pd.DataFrame(data=theta_, columns=param_names)
    param_df['ID'] = param_df.index + 1
    param_df['Z'] = (10 ** param_df['gaslog10Z']) * 0.0142
    
    spec_output_dir = cwd_ + '/spectra'
    param_id = str(re.findall(r'\d+', param_file)[0])
    
    stellar_masses = np.load(spec_output_dir + '/stellar_masses' + param_id + '.npy')

    # holder for photometry
    photometry = np.zeros((param_df.shape[0], len(filters)))

    # load in the spectra without nebular emission (but with evolving metallicity history from "generate_spectra.py")
    spectra = np.load(spec_output_dir + '/spectra' + param_id + '.npy')
    
    for param_ind_ in range(param_df.shape[0]):

        param_slice = param_df.iloc[param_ind_]

        tuniv = Planck15.age(param_slice['z'].item()).value
        
        # get the time grid, sfr, and
        t, sfr, zh = flexible_sfh(param_slice)

        # with nebular emission...
        model.params['add_neb_emission'] = True
        model.params['add_neb_continuum'] = True
        model.params['nebemlineinspec'] = True

        # set parameters
        model.params['zred'] = param_slice['z'].item()
        model.params['dust2'] = param_slice['dust2'].item()
        model.params['dust1'] = param_slice['dust2'].item() * param_slice['dust1_fraction'].item()
        model.params['dust_index'] = param_slice['dust_index'].item()
        model.set_tabular_sfh(t, sfr)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
        model.params['gas_logz'] = (10 ** param_slice['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

        # compute rest-frame spectrum
        wave0, spec0 = model.get_spectrum(tage=tuniv)

        # redshifted spectrum (and wavelength grid)
        wave, spec = wave0*(1. + param_slice['z'].item()), spec0*(1. + param_slice['z'].item())

        # without nebular emission...
        model.params['add_neb_emission'] = False
        model.params['add_neb_continuum'] = False
        model.params['nebemlineinspec'] = False

        # set parameters
        model.params['zred'] = param_slice['z'].item()
        model.params['dust2'] = param_slice['dust2'].item()
        model.params['dust1'] = param_slice['dust2'].item() * param_slice['dust1_fraction'].item()
        model.params['dust_index'] = param_slice['dust_index'].item()
        model.set_tabular_sfh(t, sfr)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
        model.params['gas_logz'] = (10 ** param_slice['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

        # compute rest-frame spectrum
        wave0, spec0_ = model.get_spectrum(tage=tuniv)

        # add the nebular emission to the previous spectrum
        spectra[param_ind_, :] += (spec0 - spec0_)

        # redshift the spectrum and compute the photometry
        wave_, spec_ = wave0*(1. + param_slice['z'].item()), spectra[param_ind_, :]*(1.+param_slice['z'].item())

        # absolute magnitudes
        M = getSED(wave_, lightspeed/wave_**2 * spec_ * to_cgs_at_10pc, filters)

        # chalk em up
        photometry[param_ind_, :] = M + 2.5*np.log10(stellar_masses[param_ind_]) # stellar mass correction

    phot_output_dir = cwd_ + '/photometry'

    if not os.path.exists(phot_output_dir):
        os.makedirs(phot_output_dir)

    # save to disc...
    np.save(phot_output_dir + '/KV_photometry' + param_id + '.npy', photometry[:, 0:9])
    np.save(phot_output_dir + '/COSMOS15_photometry' + param_id + '.npy', photometry[:, 9:])
    np.save(spec_output_dir + '/spectra' + param_id + '.npy', spectra)

parameter_files = []
for f_ in os.listdir(parameters_path):
    parameter_files.append(os.path.abspath(os.path.join(parameters_path, f_)))

# completed_spec_list_init = os.listdir(cwd_ + '/spectra/')
# completed_spec_list = [entry_ for entry_ in completed_spec_list_init if 'spectra' in entry_]
# inc_list = []
# for sp_ in completed_spec_list:
#     inc_list.append(str(re.findall(r'\d+', sp_)[0]))
#
# print([pf_ for pf_ in parameter_files])
# parameter_files_inc = [inc_ for inc_ in parameter_files if  str(re.findall(r'\d+', inc_)[0]) not in inc_list]
# print(parameter_files_inc)

# initialize cpu pool
pool = mp.Pool(processes=4)
# run
if __name__ == '__main__':

    # pool.map(generate_fsps_add_neb_spectra, [pf_ for pf_ in parameter_files])
    list(tqdm(pool.imap(generate_fsps_add_neb_spectra, [pf_ for pf_ in parameter_files]), total=len(parameter_files)))
# if __name__ == '__main__':
# process_map(generate_fsps_add_neb_spectra, [pf_ for pf_ in parameter_files])
# close the pool
pool.close()
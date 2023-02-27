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
# # mpi set-up
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# pull in arguments from command line
# root_directory = sys.argv[1]

# load the parameters for the default parameters file
# parameters_dir = '/cfs/home/side0330/models/steppz/continuity_prior/parameters'
cwd_ = os.getcwd()
parameters_path = cwd_+'/parameters/'
# # theta = np.load(parameters_path + '/parameters0' + '.npy')
# # n_time_bins = 7
# # parameter_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_bins)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
# # """Populate a pandas dataframe with the parameters array & parameter names"""
# # parameter_df = pd.DataFrame(data=theta, columns=parameter_names)
# # parameter_df['ID'] = parameter_df.index + 1
# # parameter_df['Z'] = (10**parameter_df['gaslog10Z'])*0.0142
# #
# # how many sets to run
# n_sets = 1
# n_samples = theta.shape[0]
#
# # # MPI set-up
# # sets_per_rank = n_sets // size # how many sets per node?
# # sets = range(sets_per_rank * rank, sets_per_rank * rank + sets_per_rank) # which sets to run on this node?
#
# # constants
# lsun = 3.846e33  # erg/s
# pc = 3.085677581467192e18  # in cm
# lightspeed = 2.998e18  # AA/s
# to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc*10)**2) # L_sun/AA to erg/s/cm^2/AA at 10pc
#
# # load the filters
# kids_filters = ['omegacam_' + n for n in ['u','g','r','i']]
# viking_filters = ['VISTA_' + n for n in ['Z','Y','J','H','Ks']]
# cosmos15_filters = ['ip_cosmos', 'v_cosmos', 'uvista_y_cosmos', 'r_cosmos', 'hsc_y',
#                     'zpp', 'b_cosmos', 'uvista_h_cosmos', 'wircam_H', 'ia484_cosmos',
#                     'ia527_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia738_cosmos',
#                     'ia767_cosmos', 'ia427_cosmos', 'ia464_cosmos', 'ia505_cosmos',
#                     'ia574_cosmos', 'ia709_cosmos', 'ia827_cosmos', 'uvista_j_cosmos',
#                     'uvista_ks_cosmos', 'wircam_Ks', 'NB711.SuprimeCam',
#                     'NB816.SuprimeCam']
# filters = load_filters(kids_filters + viking_filters + cosmos15_filters)
#
# # set up the SPS model
# model = fsps.StellarPopulation(zcontinuous=3, compute_vega_mags=False)
#
# # set up parameters
# model.params['sfh'] = 3 # tabular sfh
# model.params['imf_type'] = 1 # Chabrier IMF
# model.params['dust_type'] = 4 # Calzetti with power law modification
# model.params['pmetals'] = -99
#
# # turn off neublar emission
# model.params['add_neb_emission'] = False
# model.params['add_neb_continuum'] = False
# model.params['nebemlineinspec'] = False
#
# # turn off dust emission (only needed for IR)
# model.params['add_dust_emission'] = False
#
# # velocity smoothing
# model.params['smooth_velocity'] = True
# model.params['sigma_smooth'] = 150
# model.params['min_wave_smooth'] = 9e2
# model.params['max_wave_smooth'] = 1e5
#
# # initialize the model by making a single call
# param_slice_init = parameter_df.iloc[0]
# tuniv = Planck15.age(param_slice_init['z'].item()).value
#
# t, sfr, zh = flexible_sfh(param_slice_init)
#
# # set parameters
# model.params['zred'] = param_slice_init['z'].item()
# model.params['dust2'] = param_slice_init['dust2'].item()
# model.params['dust1'] = param_slice_init['dust2'].item() * param_slice_init['dust1_fraction'].item()
# model.params['dust_index'] = param_slice_init['dust_index'].item()
# model.set_tabular_sfh(t, sfr, Z=zh)
#
# # set the conditional parameters
# model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
# model.params['gas_logz'] = (10 ** param_slice_init['gaslog10Z'].item()) * 0.0142  # set to the final metalicity
#
# # compute rest-frame spectrum
# wave, spec = model.get_spectrum(tage=tuniv)
#
# # save the wave vector to file
# spec_output_dir = cwd_ + '/spectra'
# if not os.path.exists(spec_output_dir):
#     os.makedirs(spec_output_dir)
#
# np.save(spec_output_dir + '/wave.npy', wave)

# for iter_ in range(n_sets):
#
#     # load in the parameters
#     theta = np.load(parameters_path + '/parameters' + str(iter_) + '.npy')
#
#     # holders for model spectra
#     spectra = np.zeros((n_samples, len(wave)))
#     stellar_masses = np.zeros(n_samples)
#
#     for param_ind_ in tqdm(range(n_samples)):
#
#         param_slice = parameter_df.iloc[param_ind_]
#
#         tuniv = Planck15.age(param_slice['z'].item()).value
#         # t_temp = np.concatenate([tuniv * (1. - np.logspace(-3, -1e-10, 200))[::-1], np.array([tuniv])])
#         # print(tuniv)
#         # print(t_temp)
#         # print(t_temp.shape)
#
#         # get the time grid, sfr, and
#         t, sfr, zh = flexible_sfh(param_slice)
#
#         # set parameters
#         model.params['zred'] = param_slice['z'].item()
#         model.params['dust2'] = param_slice['dust2'].item()
#         model.params['dust1'] = param_slice['dust2'].item()*param_slice['dust1_fraction'].item()
#         model.params['dust_index'] = param_slice['dust_index'].item()
#         # print(t)
#         # print(t.shape)
#         # # print(sfr)
#         # print(sfr.shape)
#         # print(np.where(t[1:] > t[:-1]))
#         model.set_tabular_sfh(t, sfr, Z=zh)
#
#         # set the conditional parameters
#         model.params['gas_logu'] = np.clip(np.log10(sfr[-1]*0.82 + 1e-30)*0.3125 + 0.9982, -4.0, -1.0) # Kaasinin+18
#         model.params['gas_logz'] = (10**param_slice['gaslog10Z'].item())*0.0142 # set to the final metalicity
#
#         # compute rest-frame spectrum
#         wave_iter, spec_iter = model.get_spectrum(tage=tuniv)
#
#         # chalk em up
#         spectra[param_ind_, :] = spec_iter
#         stellar_masses[param_ind_] = model.stellar_mass
#
#         # print(str(param_ind_))
#         # np.save(spec_output_dir + '/wave'+str(param_ind_)+'.npy', wave)
#         # np.save(spec_output_dir + '/spec'+str(param_ind_)+'.npy', spec)
#
#     # save to disc
#     np.save(spec_output_dir + '/spectra' + str(iter_) + '.npy', spectra)
#     np.save(spec_output_dir + '/stellar_masses' + str(iter_) + '.npy', stellar_masses)

#
# def generate_fsps_no_neb_spectra(param_files):
#
#     """Specifications related to the fsps model"""
#     # constants
#     lsun = 3.846e33  # erg/s
#     pc = 3.085677581467192e18  # in cm
#     lightspeed = 2.998e18  # AA/s
#     to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc * 10) ** 2)  # L_sun/AA to erg/s/cm^2/AA at 10pc
#
#     # set up the SPS model
#     model = fsps.StellarPopulation(zcontinuous=3, compute_vega_mags=False)
#
#     # set up parameters
#     model.params['sfh'] = 3  # tabular sfh
#     model.params['imf_type'] = 1  # Chabrier IMF
#     model.params['dust_type'] = 4  # Calzetti with power law modification
#     model.params['pmetals'] = -99
#
#     # turn off neublar emission
#     model.params['add_neb_emission'] = False
#     model.params['add_neb_continuum'] = False
#     model.params['nebemlineinspec'] = False
#
#     # turn off dust emission (only needed for IR)
#     model.params['add_dust_emission'] = False
#
#     # velocity smoothing
#     model.params['smooth_velocity'] = True
#     model.params['sigma_smooth'] = 150
#     model.params['min_wave_smooth'] = 9e2
#     model.params['max_wave_smooth'] = 1e5
#
#     for pf_ in param_files:
#         # build the parameter dataframe
#         theta_ = np.load(pf_)
#         n_time_ = 7
#         param_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
#         """Populate a pandas dataframe with the parameters array & parameter names"""
#         param_df = pd.DataFrame(data=theta_, columns=param_names)
#         param_df['ID'] = param_df.index + 1
#         param_df['Z'] = (10 ** param_df['gaslog10Z']) * 0.0142
#
#         # holders for model spectra
#         wl_ = np.load('/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz_sfh_testing/training_data/spectra/wave.npy')
#         spectra = np.zeros((param_df.shape[0], wl_.shape[0]))
#         stellar_masses = np.zeros(param_df.shape[0])
#
#         for param_ind_ in tqdm(range(param_df.shape[0])):
#
#             param_slice = param_df.iloc[param_ind_]
#
#             tuniv = Planck15.age(param_slice['z'].item()).value
#
#             # get the time grid, sfr, and
#             t, sfr, zh = flexible_sfh(param_slice)
#
#             # set parameters
#             model.params['zred'] = param_slice['z'].item()
#             model.params['dust2'] = param_slice['dust2'].item()
#             model.params['dust1'] = param_slice['dust2'].item() * param_slice['dust1_fraction'].item()
#             model.params['dust_index'] = param_slice['dust_index'].item()
#             model.set_tabular_sfh(t, sfr, Z=zh)
#
#             # set the conditional parameters
#             model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
#             model.params['gas_logz'] = (10 ** param_slice['gaslog10Z'].item()) * 0.0142  # set to the final metalicity
#
#             # compute rest-frame spectrum
#             wave_iter, spec_iter = model.get_spectrum(tage=tuniv)
#
#             # chalk em up
#             spectra[param_ind_, :] = spec_iter
#             stellar_masses[param_ind_] = model.stellar_mass
#
#         spec_output_dir = cwd_ + '/spectra'
#         if not os.path.exists(spec_output_dir):
#             os.makedirs(spec_output_dir)
#         # save to disc
#         np.save(spec_output_dir + '/spectra' + str(re.findall(r'\d+', pf_)[0]) + '.npy', spectra)
#         np.save(spec_output_dir + '/stellar_masses' + str(re.findall(r'\d+', pf_)[0]) + '.npy', stellar_masses)


def generate_fsps_no_neb_spectra(param_file):
    print(param_file)
    """Specifications related to the fsps model"""
    # constants
    lsun = 3.846e33  # erg/s
    pc = 3.085677581467192e18  # in cm
    lightspeed = 2.998e18  # AA/s
    to_cgs_at_10pc = lsun / (4.0 * np.pi * (pc * 10) ** 2)  # L_sun/AA to erg/s/cm^2/AA at 10pc

    # load the filters
    kids_filters = ['omegacam_' + n for n in ['u', 'g', 'r', 'i']]
    viking_filters = ['VISTA_' + n for n in ['Z', 'Y', 'J', 'H', 'Ks']]
    # cosmos15_filters = ['ip_cosmos', 'v_cosmos', 'uvista_y_cosmos', 'r_cosmos', 'hsc_y',
    #                     'zpp', 'b_cosmos', 'uvista_h_cosmos', 'wircam_H', 'ia484_cosmos',
    #                     'ia527_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia738_cosmos',
    #                     'ia767_cosmos', 'ia427_cosmos', 'ia464_cosmos', 'ia505_cosmos',
    #                     'ia574_cosmos', 'ia709_cosmos', 'ia827_cosmos', 'uvista_j_cosmos',
    #                     'uvista_ks_cosmos', 'wircam_Ks', 'NB711.SuprimeCam',
    #                     'NB816.SuprimeCam']
    cosmos20_filters = [
        'galex_NUV', 'u_megaprime_sagem',
        'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y',
        'uvista_y_cosmos', 'uvista_j_cosmos', 'uvista_h_cosmos', 'uvista_ks_cosmos',
        'ia427_cosmos', 'ia464_cosmos', 'ia484_cosmos', 'ia505_cosmos', 'ia527_cosmos',
        'ia574_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia709_cosmos', 'ia738_cosmos',
        'ia767_cosmos', 'ia827_cosmos',
        'NB711.SuprimeCam', 'NB816.SuprimeCam',
        'b_cosmos', 'v_cosmos', 'r_cosmos', 'ip_cosmos', 'zpp',
        'irac1_cosmos', 'irac2_cosmos'
    ]
    filters = load_filters(kids_filters + viking_filters + cosmos20_filters)

    # set up the SPS model
    model = fsps.StellarPopulation(zcontinuous=3, compute_vega_mags=False)

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

    # for pf_ in param_files:
    # build the parameter dataframe
    theta_ = np.load(param_file)
    n_time_ = 7
    param_names = ['N', 'gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_)] + ['dust2', 'dust_index', 'dust1_fraction', 'z']
    """Populate a pandas dataframe with the parameters array & parameter names"""
    param_df = pd.DataFrame(data=theta_, columns=param_names)
    param_df['ID'] = param_df.index + 1
    param_df['Z'] = (10 ** param_df['gaslog10Z']) * 0.0142

    # holders for model spectra
    wl_ = np.load('/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz_sfh_testing/training_data/spectra/wave.npy')
    spectra = np.zeros((param_df.shape[0], wl_.shape[0]))
    stellar_masses = np.zeros(param_df.shape[0])

    # holder for photometry
    photometry = np.zeros((param_df.shape[0], len(filters)))

    for param_ind_ in range(param_df.shape[0]):

        param_slice = param_df.iloc[param_ind_]

        tuniv = Planck15.age(param_slice['z'].item()).value

        # get the time grid, sfr, and
        t, sfr, zh = flexible_sfh(param_slice)

        # set parameters
        model.params['zred'] = param_slice['z'].item()
        model.params['dust2'] = param_slice['dust2'].item()
        model.params['dust1'] = param_slice['dust2'].item() * param_slice['dust1_fraction'].item()
        model.params['dust_index'] = param_slice['dust_index'].item()
        model.set_tabular_sfh(t, sfr, Z=zh)

        # set the conditional parameters
        model.params['gas_logu'] = np.clip(np.log10(sfr[-1] * 0.82 + 1e-30) * 0.3125 + 0.9982, -4.0, -1.0)  # Kaasinin+18
        model.params['gas_logz'] = (10 ** param_slice['gaslog10Z'].item()) * 0.0142  # set to the final metalicity

        # compute rest-frame spectrum
        wave_iter, spec_iter = model.get_spectrum(tage=tuniv)

        # chalk em up
        spectra[param_ind_, :] = spec_iter
        stellar_masses[param_ind_] = model.stellar_mass

        # redshift the spectrum and compute the photometry
        wave_, spec_ = wave_iter*(1. + param_slice['z'].item()), spectra[param_ind_, :]*(1.+param_slice['z'].item())

        # absolute magnitudes
        M = getSED(wave_, lightspeed/wave_**2 * spec_ * to_cgs_at_10pc, filters)

        # chalk em up
        photometry[param_ind_, :] = M + 2.5*np.log10(stellar_masses[param_ind_]) # stellar mass correction

    spec_output_dir = cwd_ + '/spectra'
    if not os.path.exists(spec_output_dir):
        os.makedirs(spec_output_dir)

    phot_output_dir = cwd_ + '/photometry'
    if not os.path.exists(phot_output_dir):
        os.makedirs(phot_output_dir)

    # save to disc
    # np.save(spec_output_dir + '/spectra' + str(re.findall(r'\d+', param_file)[0]) + '.npy', spectra)
    # np.save(spec_output_dir + '/stellar_masses' + str(re.findall(r'\d+', param_file)[0]) + '.npy', stellar_masses)
    np.save(phot_output_dir + '/KV_photometry' + str(re.findall(r'\d+', param_file)[0]) + '_no_emission.npy', photometry[:, 0:9])
    np.save(phot_output_dir + '/COSMOS20_photometry' + str(re.findall(r'\d+', param_file)[0]) + '_no_emission.npy', photometry[:, 9:])

    # alternative to re.findall(r'\d+', param_file)[0] -- > split at '.', split at '/'


parameter_files = []
for f_ in os.listdir(parameters_path):
    parameter_files.append(os.path.abspath(os.path.join(parameters_path, f_)))

# completed_spec_list_init = os.listdir(cwd_ + '/spectra/')
# completed_spec_list = [entry_ for entry_ in completed_spec_list_init if 'spectra' in entry_]
# inc_list = []
# for sp_ in completed_spec_list:
#     inc_list.append(str(re.findall(r'\d+', sp_)[0]))
# print(completed_spec_list)
# print(inc_list)

completed_phot_list_init = os.listdir(cwd_ + '/photometry/')
completed_phot_list = [entry_ for entry_ in completed_phot_list_init if 'no_emission' in entry_]
inc_list = []
for sp_ in completed_phot_list:
    inc_list.append(str(re.findall(r'\d+', sp_)[0]))
print(completed_phot_list)
print(np.unique(inc_list))

inc_list_ = [x for x in inc_list if x != '20']  # delete

# generate_fsps_no_neb_spectra(parameter_files)

# fsps_spec_thread = threading.Thread(target=generate_fsps_no_neb_spectra, args=(parameter_files,))
# fsps_spec_thread.start()
# fsps_spec_thread.join()

print([pf_ for pf_ in parameter_files])
parameter_files_inc = [inc_ for inc_ in parameter_files if str(re.findall(r'\d+', inc_)[0]) not in np.unique(inc_list_)]

print(parameter_files_inc)

# # initialize cpu pool
# pool = mp.Pool(processes=6)
# # run
# pool.imap(generate_fsps_no_neb_spectra, [pf_ for pf_ in parameter_files])
# # tqdm(pool.imap(generate_fsps_no_neb_spectra, [pf_ for pf_ in parameter_files_inc]), total=len(parameter_files_inc))
# # close the pool
# pool.close()

pool = mp.Pool(processes=6)
# run
if __name__ == '__main__':

    # pool.map(generate_fsps_add_neb_spectra, [pf_ for pf_ in parameter_files])
    list(tqdm(pool.imap(generate_fsps_no_neb_spectra, [pf_ for pf_ in parameter_files_inc]), total=len(parameter_files)))
# if __name__ == '__main__':
# process_map(generate_fsps_add_neb_spectra, [pf_ for pf_ in parameter_files])
# close the pool
pool.close()
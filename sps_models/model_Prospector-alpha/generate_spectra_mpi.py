import fsps
from astropy.cosmology import WMAP9
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from prospect.models import priors
import tqdm
import pickle
from sedpy.observate import load_filters

# mpi set-up
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# import the prospector model parameters file
import prospector_alpha_params as pfile

def prior():
    
    mm = pfile.MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=12.5)
    massmet = mm.sample()
    log10M = massmet[0]
    log10Z = massmet[1]
    
    nbins_sfh=7-1, 
    sigma=0.3, 
    df=2,
    logsfr = priors.StudentT(mean=np.zeros(nbins_sfh),
                             scale=np.ones(nbins_sfh)*sigma,
                             df=np.ones(nbins_sfh)*df)
    logsfr_ratios = logsfr.sample()
    
    #dust2 = pfile.model_params[2]['prior'].sample()
    dust2 = np.random.uniform(0, 2, size=1)**2
    
    dust_index = pfile.model_params[3]['prior'].sample()
    dust1_fraction = pfile.model_params[4]['prior'].sample()
    fagn = pfile.model_params[5]['prior'].sample()
    agn_tau = pfile.model_params[6]['prior'].sample()
    gas_logz = pfile.model_params[7]['prior'].sample()
    z = np.random.uniform(0., 2.5, size=1)

    return np.concatenate([log10M, log10Z, logsfr_ratios, dust2, dust_index, dust1_fraction, fagn, agn_tau, gas_logz, z])

def generate_magnitudes(theta, sps, obs):
    
    pfile.run_params['zred'] = theta[-1]
    mod = pfile.load_model(**pfile.run_params)
    mod.params['zred'] = theta[-1]
    
    # Generate spectrum
    spec, ma, sm = mod.mean_model(theta[0:-1], sps=sps, obs=obs)
    
    return ma

# pull in arguments from command line
root_directory = sys.argv[1]

# how many sets to run
n_sets = 64
n_samples = 100000

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

# loop over sets
for k in sets:

    training_theta = []
    training_mags = []
    training_absmags = []

    # now loop over the training parameters
    for i in range(n_samples):

        # sample prior
        theta = prior()
        
        # compute magnitudes
        mags = np.log10(generate_magnitudes(theta, sps, obs))/(-0.4)
        
        # adjust to unit mass absolute magnitudes
        absmags = mags - WMAP9.distmod(theta[-1]).value + 2.5*theta[0]
        
        training_theta.append(theta)
        training_mags.append(mags)
        training_absmags.append(absmags)

    # cast to np arrays
    training_theta = np.array(training_theta)
    training_mags = np.array(training_mags)
    training_absmags = np.array(training_absmags)

    # save to disc
    np.save(root_directory + 'parameters/parameters' + str(k) + '.npy', spectra)
    np.save(root_directory + 'spectra/KV_photometry' + str(k) + '.npy', training_absmags[:,0:9])
    np.save(root_directory + 'spectra/COSMOS15_photometry' + str(k) + '.npy', training_absmags[:,9:])


import sys
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
import photoz_hmII_params as pfile

# load models
obs, model, sps, noise = pfile.build_all()
spectrum, maggies, _ = model.mean_model(model.theta, sps=sps, obs=obs)

# prior
def prior():
    
    z = np.random.uniform(1e-5, 2.5)
    M = 10**np.random.uniform(6, 13)
    lnZ = np.random.uniform(-1.98, 0.19)
    dust2 = np.random.uniform(0, 2)**2
    tau = np.exp(np.random.uniform(np.log(0.316), np.log(100)))
    tmax = np.exp(np.random.uniform(np.log(7.3e-5), np.log(1.0)))
    dust_index = np.random.uniform(-1, 0.4)

    return np.array([z, M, lnZ, dust2, tau, tmax, dust_index])

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
obs['filters'] = filters

# loop over sets
for k in sets:

    training_theta = []
    training_mags = []
    training_absmags = []

    # now loop over the training parameters
    for i in range(n_samples):

        # sample prior
        theta = prior()
        z, M, log10Z, dust2, tau, tmax, dust_index = np.split(theta, len(theta))

        # compute magnitudes
        spec, maggies, mfrac = model.mean_model(theta, sps=sps, obs=obs)           
        mags = np.log10(maggies)/(-0.4)

        # adjust to unit mass absolute magnitudes
        absmags = mags - WMAP9.distmod(z).value + 2.5*np.log10(M*mfrac)
        
        training_theta.append(np.array([np.log10(M), log10Z, dust2, tau, tmax, dust_index, z]))
        training_mags.append(mags)
        training_absmags.append(absmags)

    # cast to np arrays
    training_theta = np.array(training_theta)
    training_mags = np.array(training_mags)
    training_absmags = np.array(training_absmags)

    # save to disc
    np.save(root_directory + 'parameters/parameters' + str(k) + '.npy', training_theta)
    np.save(root_directory + 'photometry/KV_photometry' + str(k) + '.npy', training_absmags[:,0:9])
    np.save(root_directory + 'photometry/COSMOS15_photometry' + str(k) + '.npy', training_absmags[:,9:])


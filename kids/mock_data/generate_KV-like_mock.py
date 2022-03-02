
import numpy as np
import tensorflow as tf
import pickle
from scipy import stats

import sys
sys.path.append('/cfs/home/alju5794/steppz/code')
from utils import *
from priors import *
from affine import *
from ndes import *

from speculator import *

# import the relevant models for the prior
log10sSFR_emulator = RegressionNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/sps_models/model_B/DPL_log10sSFR_emulator.pkl')
baseline_SFR_prior_log_prob = RegressionNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/sps_models/model_B/DPL_baseline_SFR_prior_logprob.pkl')

# redshift prior
def redshift_prior(z):
    
    return redshift_volume_prior(z) + tfd.Uniform(low=0, high=1.5).log_prob(z)

# set up the prior class
Prior = ModelABBaselinePrior(baselineSFRprior=baseline_SFR_prior_log_prob, 
                             log10sSFRemulator=log10sSFR_emulator, 
                             log10sSFRprior=log10sSFRpriorMizuki, 
                             log10sSFRuniformlimits=tfd.Uniform(low=-14, high=-7), 
                             redshift_prior=redshift_prior)

# initial walkers
n_walkers = 1000
chain = Prior.bijector.inverse(tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/sps_models/model_B/training_data_prior/parameters/parameters0.npy'), dtype=tf.float32)).numpy()
chain = chain[chain[:,-1] < 1.5,:]
current_state = [tf.convert_to_tensor(chain[0:n_walkers,:], dtype=tf.float32), tf.convert_to_tensor(chain[n_walkers:2*n_walkers,:], dtype=tf.float32)]

n_layers = 4
n_hidden = 128
filternames = ['omegacam_u', 'omegacam_g', 'omegacam_r', 'omegacam_i', 'VISTA_Z', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks']
root_dir = '/cfs/home/alju5794/steppz/sps_models/model_B/trained_models/'
filenames = ['model_{}x{}'.format(n_layers, n_hidden) + filtername for filtername in filternames]
model = PhotulatorModelStack(root_dir=root_dir, filenames=filenames)

# parameter transforms
n_sps_parameters = Prior.n_sps_parameters
transforms = [tfb.Identity() for _ in range(n_sps_parameters-1)]
transforms[1] = tfb.Invert(tfb.Square()) # dust2 -> sqrt(dust2)
transform = tfb.Blockwise(transforms)

# noise for bootstrapping over
sigma_bootstrap = np.load('/cfs/home/alju5794/steppz/kids/mock_data/sigma_bootstrap.npy')

SNR_limit = np.array([1., 1., 3., 1., 0., 0., 0., 0., 0.])
mag_limit = None

model_error = np.array([0.03, 0.03, 0.03, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
zp_error = np.array([0.05, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03])
frac_error = model_error + zp_error

# import selection function model
SelectionFunction1SDSS = BinomialNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/kids/mock_data/1SDSS.pkl')

# simulate mock data
n_selected = 0
count = 0
target_n_selected = int(sys.argv[2])
n_steps = 25

while (n_selected < target_n_selected) == True :
    
    # MCMC sample
    chain = affine_sample(Prior.log_prob, n_steps, current_state, progressbar=False)
    current_state = [chain[-1,0:n_walkers,:], chain[-1,n_walkers:,:]]
    parameters = Prior.bijector(chain).numpy().reshape((n_steps*2*n_walkers, 9))
    
    # emulate fluxes
    model_fluxes = model.fluxes(transform(parameters[:,1:]), parameters[:,0]).numpy() # nano maggies
    
    # add bootstrapped noise
    noise_sigmas = sigma_bootstrap[np.random.randint(0, len(sigma_bootstrap), parameters.shape[0]), :]
    flux_sigmas = np.sqrt(noise_sigmas**2 + (frac_error * model_fluxes)**2)
    mock_fluxes = model_fluxes + stats.norm.rvs(loc=0, scale=flux_sigmas)
    mock_mags = -2.5 * np.log10(mock_fluxes)
    cut = np.all(~np.isinf(mock_mags), axis=-1) * np.all(~np.isnan(mock_mags), axis=-1)
    mock_fluxes = mock_fluxes[cut,:]
    mock_mags = mock_mags[cut,:]
    noise_sigmas = noise_sigmas[cut,:]
    parameters = parameters[cut,:]

    # photometric selection
    photometric_cuts = np.ones(noise_sigmas.shape[0])
    if mag_limit is not None:
        photometric_cuts *= np.all(mock_mags + 9*2.5 < mag_limit, axis=-1)
    if SNR_limit is not None:
        SNR = mock_fluxes / noise_sigmas
        photometric_cuts *= np.all(SNR > SNR_limit, axis=-1)
    Sp = photometric_cuts
    
    # apply the photometric selection
    mock_mags = mock_mags[Sp == 1.,:] 
    noise_sigmas = noise_sigmas[Sp == 1.,:] 
    parameters = parameters[Sp == 1.,:]
    
    # apply spectroscopic selection
    pSDSS = tf.squeeze(SelectionFunction1SDSS(tf.concat([tf.convert_to_tensor(mock_mags, dtype=tf.float32), tf.convert_to_tensor(-2.5*np.log10(noise_sigmas), dtype=tf.float32)], axis=-1)), -1).numpy()
    Ss = np.random.binomial(1, pSDSS)
    
    # append to accumulated data vectors
    if count == 0:
        photometry = mock_mags
        errors = noise_sigmas
        selection = Ss
        thetas = parameters
    else:
        photometry = np.concatenate([photometry, mock_mags])
        errors = np.concatenate([errors, noise_sigmas])
        selection = np.concatenate([selection, Ss])
        thetas = np.concatenate([thetas, parameters])
        
    n_selected += sum(Ss)
    count += 1


f = open('/cfs/home/alju5794/steppz/kids/mock_data/KV-like_mock{}.pkl'.format(sys.argv[1]), 'wb')
pickle.dump([photometry, errors, selection, thetas], f)
f.close()
import pickle


import numpy as np
from speculator import Photulator, PhotulatorModelStack
import time

import sys
sys.path.append('/cfs/home/alju5794/steppz/code')
from utils import *
from priors import *
from likelihoods import *
from affine import *
from ndes import *

# number of bands
n_bands = 9

# assumed fractional model error per band
model_error = tf.constant([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# assumed ZP (fractional) error per band
zp_error = tf.constant([0.05, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# import data
fluxes, flux_sigmas, zspec, specsource, zb, zprior_sig = pickle.load(open('/cfs/home/alju5794/steppz/kids/data/KV450_cut_all.pkl', 'rb'))

# convert to tensors
flux_variances = tf.constant(np.atleast_2d(flux_sigmas**2).astype(np.float32), dtype=tf.float32)
fluxes = tf.constant(np.atleast_2d(fluxes).astype(np.float32), dtype=tf.float32)

# n_sigma_flux_cuts
n_sigma_flux_cuts = tf.constant([1., 1., 3., 1., 0., 0., 0., 0., 0.], dtype=tf.float32)

n_layers = 4
n_hidden = 128
filternames = ['omegacam_u', 'omegacam_g', 'omegacam_r', 'omegacam_i', 'VISTA_Z', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks']
root_dir = '/cfs/home/alju5794/steppz/sps_models/model_B/trained_models/'
filenames = ['model_{}x{}'.format(n_layers, n_hidden) + filtername for filtername in filternames]
emulator = PhotulatorModelStack(root_dir=root_dir, filenames=filenames)

# prior limits and associated hard priors
sfh_prior = AutoregressiveNeuralSplineFlow(restore=True, restore_filename='/cfs/home/alju5794/steppz/sps_models/model_B/NSF_DPL_SFH.pkl')
sps_prior = ModelABBaselinePrior(SFHPrior=sfh_prior)
n_sps_parameters = sps_prior.n_sps_parameters

# bijector from constrained parameter (physical) to unconstrained space for sampling. Note: no bijector for the normalization parameter N
# forward pass takes you from unconstrained space to physical parameter space
bijector = sps_prior.bijector

# declare any transforms that occur on the physical SPS parameters before passing to Speculator models
transforms = [tfb.Identity() for _ in range(n_sps_parameters-1)]
transforms[1] = tfb.Invert(tfb.Square()) # dust2 -> sqrt(dust2)
transform = tfb.Blockwise(transforms)

# input shape of latent parameters should be (n_walkers, n_galaxies, n_sps_parameters), output shape should be (n_walkers, n_galaxies)
@tf.function
def log_latentparameter_conditional(latentparameters, hyperparameters, fluxes, flux_variances, n_sigma_flux_cuts):
    
    # split the hyper parameters
    zero_points, additive_fractional_errors = tf.split(hyperparameters, (n_bands, n_bands), axis=-1)
    zero_points = tf.expand_dims(zero_points, axis=0)
    additive_fractional_errors = tf.expand_dims(additive_fractional_errors, axis=0)

    # convert latent parameters into physical parameters using bijector
    theta = sps_prior.bijector(latentparameters)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    
    # model and predicted fluxes (multiplied by zero points)
    model_fluxes = emulator.fluxes(transform(theta[...,1:]), N)
    predicted_fluxes = tf.multiply(model_fluxes, zero_points)
    
    # flux variances (quoted variances, scaled by zero points, plus additional fractional error terms)
    predicted_flux_variances = tf.add(flux_variances, tf.square(tf.multiply(additive_fractional_errors, predicted_fluxes)))

    # log likelihood
    log_likelihood_ = log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances, n_sigma_flux_cuts)
    
    # log-prior
    log_prior_ = sps_prior.log_prob(latentparameters)
    
    return log_likelihood_ + log_prior_

# input shape of hyperparameters should be (n_walkers, n_hyperparameters), output shape should be (n_walkers)
@tf.function
def log_hyperparameter_conditional(hyperparameters, model_fluxes, fluxes, flux_variances, n_sigma_flux_cuts):
    
    # split the hyper parameters
    zero_points, additive_fractional_errors = tf.split(hyperparameters, (n_bands, n_bands), axis=-1)
    zero_points = tf.expand_dims(zero_points, axis=1)
    additive_fractional_errors = tf.expand_dims(additive_fractional_errors, axis=1)
    
    # compute predicted fluxes from model fluxes and hyper parameters (ie., scale by zero-points)
    predicted_fluxes = tf.multiply(zero_points, model_fluxes)
    
    # flux variances (quoted variances, scaled by zero points, plus additional fractional error term)
    predicted_flux_variances = tf.add(flux_variances, tf.square(tf.multiply(additive_fractional_errors, predicted_fluxes)))

    # log likelihoods
    log_likelihood_ = log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances, n_sigma_flux_cuts)
    
    # log prior
    log_prior_ = hyperparameter_log_prior(hyperparameters)

    return tf.reduce_sum(log_likelihood_, axis=-1) + log_prior

# initial walker states
n_walkers = 300

# initialize latent
latent_current_state = [tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi.npy')[0:n_walkers,:,:].astype(np.float32), dtype=tf.float32), tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi.npy')[n_walkers:2*n_walkers,:,:].astype(np.float32), dtype=tf.float32)]

# initialize hyper-parameters
hyper_parameters_ = tf.concat([tf.ones(9, dtype=tf.float32), model_error + zp_error], axis=-1) + tf.random.normal([])
hyper_current_state = [hyper_parameters_ + tf.random.normal([n_walkers, hyper_parameters_.shape[0]], 0, 1e-3), hyper_parameters_ + tf.random.normal([n_walkers, hyper_parameters_.shape[0]], 0, 1e-3)]

# set up batching of latent parameters
n_latent = fluxes.shape[0] # number of galaxies
latent_batch_size = 10000
n_latent_batches = n_latent // latent_batch_size + int( (n_latent % latent_batch_size) > 0)
batch_indices = [np.arange(latent_batch_size*i, min(latent_batch_size*(i+1), n_latent)) for i in range(n_latent_batches)]

# how many MCMC steps?
n_steps = 600
n_sub_steps = 5

# loop over batches
for step in range(n_steps):

    # sample latent parameters, conditioned on hyper-parameters (do it in batches and concatenate them together)
    latent_samples_ = tf.concat([affine_sample_batch(log_latentparameter_conditional, 
                                                     n_sub_steps, 
                                                     [tf.gather(latent_current_state[0], batch_indices[_], axis=1), tf.gather(latent_current_state[1], batch_indices[_], axis=1)], 
                                                     args=[hyper_parameters_, tf.gather(fluxes, batch_indices[_], axis=0), tf.gather(flux_variances, batch_indices[_], axis=0), n_sigma_flux_cuts]) for _ in range(n_latent_batches)], axis=2)
    latent_current_state = tf.split(latent_samples_[-1,...], (n_latent_walkers, n_latent_walkers), axis=0) # set current walkers state
    latent_parameters_ = latent_current_state[np.random.randint(0, 2)][np.random.randint(0, n_latent_walkers),...] # latent-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # compute model fluxes for latent parameters that we'll now condition on (which remain fixed during the hyper-parameter sampling step)
    theta = sps_prior.bijector(latent_parameters_)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    model_fluxes = tf.concat([Emulator.fluxes(transform(tf.gather(theta[...,1:], batch_indices[_], axis=0)), tf.gather(N, batch_indices[_], axis=0)) for _ in range(n_latent_batches)], axis=0)        

    # sample hyper-parameters, conditioned on latent parameters
    hyper_samples_ = affine_sample(log_hyperparameter_conditional, n_sub_steps, hyper_current_state, args=[model_fluxes, fluxes, flux_variances, n_sigma_flux_cuts])
    hyper_current_state = tf.split(hyper_samples_[-1,...], (n_hyper_walkers, n_hyper_walkers), axis=0) # set current walkers state
    hyper_parameters_ = hyper_current_state[np.random.randint(0, 2)][np.random.randint(0, n_hyper_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)
    
    # save the chain
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM/latent{}.npy'.format(step), sps_prior.bijector(latent_samples_[-1,...].astype(np.float32)).numpy() )
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM/hyper{}.npy'.format(step), hyper_samples_[-1,...].astype(np.float32))


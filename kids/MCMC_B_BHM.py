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
zspec = tf.constant(zspec.astype(np.float32), dtype=tf.float32)
zprior_sig = tf.constant(zprior_sig.astype(np.float32), dtype=tf.float32)

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
def log_latentparameter_conditional(latentparameters, hyperparameters, fluxes, flux_variances, n_sigma_flux_cuts, zspec, zprior_sig, nz_parameters):
    
    # split the hyper parameters
    zero_points, additive_fractional_errors = tf.split(hyperparameters, (n_bands, n_bands), axis=-1)
    zero_points = tf.expand_dims(zero_points, axis=0)
    additive_fractional_errors = tf.expand_dims(additive_fractional_errors, axis=0)

    # convert latent parameters into physical parameters using bijector
    theta = sps_prior.bijector(latentparameters)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    z = theta[...,-1] # extract redshift

    # model and predicted fluxes (multiplied by zero points)
    model_fluxes = emulator.fluxes(transform(theta[...,1:]), N)
    predicted_fluxes = tf.multiply(model_fluxes, zero_points)
    
    # flux variances (quoted variances, scaled by zero points, plus additional fractional error terms)
    predicted_flux_variances = tf.add(flux_variances, tf.square(tf.multiply(additive_fractional_errors, predicted_fluxes)))

    # log likelihood
    log_likelihood_ = log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances, n_sigma_flux_cuts)
    
    # log-prior
    log_prior_ = sps_prior.log_prob(latentparameters)

    # extra redshift prior from speczs
    log_speczz_prior_ = -tf.multiply(0.5, tf.square(tf.divide(tf.subtract(z, zspec), zprior_sig)))

    # extra redshift prior from n(z)...

    # parameters
    logits, locs, logscales, skewness, tailweight = tf.split(nz_parameters, (3, 3, 3, 3, 3), axis=-1)

    # mixture model
    nz = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits),
                          components_distribution=tfd.SinhArcsinh(loc=locs, scale=tf.exp(logscales), skewness=skewness, tailweight=tailweight))

    log_z_prior_ = nz.log_prob(z)
    
    return log_likelihood_ + log_prior_ + log_specz_prior_ + log_z_prior_

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

@tf.function
def log_nz_conditional(theta, z):
    
    # pull out parameters
    logits, locs, logscales, skewness, tailweight = tf.split(theta, (3, 3, 3, 3, 3), axis=1)

    # mixture model
    nz = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits),
                          components_distribution=tfd.SinhArcsinh(loc=locs, scale=tf.exp(logscales), skewness=skewness, tailweight=tailweight))

    # log prob
    return tf.reduce_sum(nz.log_prob(z), axis=0)

# initial walker states
n_latent_walkers = 300
n_hyper_walkers = 300
n_nz_walkers = 300

# initialize latent
latent_current_state = [tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi.npy')[0:n_latent_walkers,:,:].astype(np.float32), dtype=tf.float32), tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi.npy')[n_latent_walkers:2*n_latent_walkers,:,:].astype(np.float32), dtype=tf.float32)]

# initialize hyper-parameters
hyper_parameters_ = tf.concat([tf.ones(9, dtype=tf.float32), model_error + zp_error], axis=-1)
hyper_current_state = [hyper_parameters_ + tf.random.normal([n_hyper_walkers, hyper_parameters_.shape[0]], 0, 1e-3), hyper_parameters_ + tf.random.normal([n_hyper_walkers, hyper_parameters_.shape[0]], 0, 1e-3)]

# initialize n(z)
logits = np.array([-0.67778176, -1.1752868, -1.6953907]).astype(np.float32)
locs = np.array([0.11383244, 0.28379175, 0.532703]).astype(np.float32)
scales = np.array([0.05216346, 0.10501441, 0.09464115]).astype(np.float32)
skewness = np.array([0.23342754,  0.401639, -0.001292]).astype(np.float32)
tailweight = np.array([0.7333437, 1.6772406, 1.1508114]).astype(np.float32)
nz_parameters_ = tf.concat([logits, locs, tf.math.log(scales), skewness, tailweight], axis=-1)
nz_current_state = [nz_parameters_ + tf.random.normal([n_nz_walkers, nz_parameters_.shape[0]], 0, 1e-3), nz_parameters_ + tf.random.normal([n_nz_walkers, nz_parameters_.shape[0]], 0, 1e-3)]

# set up batching of latent parameters
n_latent = fluxes.shape[0] # number of galaxies
latent_batch_size = 1000
n_latent_batches = n_latent // latent_batch_size + int( (n_latent % latent_batch_size) > 0)
batch_indices = [np.arange(latent_batch_size*i, min(latent_batch_size*(i+1), n_latent)) for i in range(n_latent_batches)]

# how many MCMC steps?
n_steps = 600
n_sub_steps = 5

# loop over batches
for step in range(n_steps):

    # sample latent parameters, conditioned on hyper-parameters (do it in batches and concatenate them together)
    latent_samples_ = tf.concat([affine_sample_batch_state(log_latentparameter_conditional, 
                                                     n_sub_steps, 
                                                     [tf.gather(latent_current_state[0], batch_indices[_], axis=1), tf.gather(latent_current_state[1], batch_indices[_], axis=1)], 
                                                     args=[hyper_parameters_, tf.gather(fluxes, batch_indices[_], axis=0), tf.gather(flux_variances, batch_indices[_], axis=0), n_sigma_flux_cuts, tf.gather(zspec, batch_indices[_], axis=0), tf.gather(zprior_sig, batch_indices[_], axis=0)], tensor=True) for _ in range(n_latent_batches)], axis=1)
    latent_current_state = tf.split(latent_samples_, (n_latent_walkers, n_latent_walkers), axis=0) # set current walkers state
    latent_parameters_ = latent_current_state[np.random.randint(0, 2)][np.random.randint(0, n_latent_walkers),...] # latent-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # compute model fluxes for latent parameters that we'll now condition on (which remain fixed during the hyper-parameter sampling step)
    theta = sps_prior.bijector(latent_parameters_)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    model_fluxes = tf.concat([Emulator.fluxes(transform(tf.gather(theta[...,1:], batch_indices[_], axis=0)), tf.gather(N, batch_indices[_], axis=0)) for _ in range(n_latent_batches)], axis=0)        

    # sample hyper-parameters, conditioned on latent parameters
    hyper_samples_ = affine_sample(log_hyperparameter_conditional, n_sub_steps, hyper_current_state, args=[model_fluxes, fluxes, flux_variances, n_sigma_flux_cuts])
    hyper_current_state = tf.split(hyper_samples_[-1,...], (n_hyper_walkers, n_hyper_walkers), axis=0) # set current walkers state
    hyper_parameters_ = hyper_current_state[np.random.randint(0, 2)][np.random.randint(0, n_hyper_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # sample the nz parameters
    nz_samples_ = affine_sample(log_nz_conditional, n_sub_steps, nz_current_state, args=[tf.expand_dims(latent_parameters_[...,-1], -1)])
    nz_current_state = tf.split(nz_samples_, (n_nz_walkers, n_nz_walkers), axis=0)
    nz_parameters_ = nz_current_state[np.random.randint(0, 2)][np.random.randint(0, n_nz_walkers),...] 
    
    # save the chain
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM/latent{}.npy'.format(step), sps_prior.bijector(latent_samples_[-1,...].astype(np.float32)).numpy() )
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM/hyper{}.npy'.format(step), hyper_samples_[-1,...].astype(np.float32))
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM/nz{}.npy'.format(step), nz_samples_[-1,...].astype(np.float32))


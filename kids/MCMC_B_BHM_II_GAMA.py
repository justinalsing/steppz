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

# thinning
n_thin = 50

# burn-in or not?
burnin = True

# number of bands
n_bands = 9

# assumed fractional model error per band
model_error = tf.constant([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# assumed ZP (fractional) error per band
zp_error = tf.constant([0.05, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# import data
fluxes, flux_sigmas, zspec, specsource, zb, zprior_sig = pickle.load(open('/cfs/home/alju5794/steppz/kids/data/KV1000_GAMA_cut_all.pkl', 'rb'))
zprior_sig[zprior_sig < 1.] = 0.01

# convert to tensors
flux_variances = tf.constant(np.atleast_2d(flux_sigmas**2).astype(np.float32), dtype=tf.float32)
fluxes = tf.constant(np.atleast_2d(fluxes).astype(np.float32), dtype=tf.float32)
zspec = tf.constant(zspec.astype(np.float32), dtype=tf.float32)
zprior_sig = tf.constant(zprior_sig.astype(np.float32), dtype=tf.float32)
zprior_sig_fixed = tf.ones(zprior_sig.shape, dtype=tf.float32)*0.01

# emulator models
n_layers = 4
n_hidden = 128
filternames = ['omegacam_u', 'omegacam_g', 'omegacam_r', 'omegacam_i', 'VISTA_Z', 'VISTA_Y', 'VISTA_J', 'VISTA_H', 'VISTA_Ks']
root_dir = '/cfs/home/alju5794/steppz/sps_models/model_B/trained_models/'
filenames = ['model_{}x{}'.format(n_layers, n_hidden) + filtername for filtername in filternames]
emulator = PhotulatorModelStack(root_dir=root_dir, filenames=filenames)

# prior limits and associated hard priors
log10sSFR_emulator = RegressionNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/sps_models/model_B/DPL_log10sSFR_emulator.pkl')
baseline_SFR_prior_log_prob = RegressionNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/sps_models/model_B/DPL_baseline_SFR_prior_logprob.pkl')
sps_prior = ModelABBaselinePrior(baselineSFRprior=baseline_SFR_prior_log_prob, 
                             log10sSFRemulator=log10sSFR_emulator, 
                             log10sSFRprior=log10sSFRpriorMizuki, 
                             log10sSFRuniformlimits=tfd.Uniform(low=-14, high=-7.5), 
                             redshift_prior=redshift_volume_prior,
                             FMRprior='curti')
n_sps_parameters = sps_prior.n_sps_parameters

# N prior normalization emulator
#log_N_prior_normalization_emulator = RegressionNetwork(restore=True, restore_filename='/cfs/home/alju5794/steppz/kids/log_N_prior_normalization_emulator.pkl')

# bijector from constrained parameter (physical) to unconstrained space for sampling. Note: no bijector for the normalization parameter N
# forward pass takes you from unconstrained space to physical parameter space
bijector = sps_prior.bijector

# declare any transforms that occur on the physical SPS parameters before passing to Speculator models
transforms = [tfb.Identity() for _ in range(n_sps_parameters-1)]
transforms[1] = tfb.Invert(tfb.Square()) # dust2 -> sqrt(dust2)
transform = tfb.Blockwise(transforms)

# hyperparameter prior
#hyper_parameter_prior = tfd.Uniform(low=[12.5, 0.05], high=[14.5, 0.5])

# nuisance parameter prior
#nuisance_parameter_prior = 

hyper_parameters_ = None

# input shape of latent parameters should be (n_walkers, n_galaxies, n_sps_parameters), output shape should be (n_walkers, n_galaxies)
@tf.function
def log_latentparameter_conditional(latentparameters, hyperparameters, nuisanceparameters, fluxes, flux_variances, zspec, zprior_sig):
    
    # split the nuisance parameters
    zero_points, additive_log_fractional_errors = tf.split(nuisanceparameters, (n_bands, n_bands), axis=-1)
    zero_points = tf.expand_dims(zero_points, axis=0)
    additive_fractional_errors = tf.expand_dims(tf.exp(additive_log_fractional_errors), axis=0)

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
    log_likelihood_ = log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances)
    
    # log-prior
    log_prior_ = sps_prior.log_prob(latentparameters)

    # extra prior on N due to selection
    log_prior_ = log_prior_# - tf.math.log(1. + tf.exp((N - hyperparameters[0])/hyperparameters[1])) - tf.squeeze(log_N_prior_normalization_emulator(tf.expand_dims(hyperparameters, 0)), -1)

    # extra redshift prior from speczs
    log_specz_prior_ = -tf.multiply(0.5, tf.square(tf.divide(tf.subtract(z, zspec), zprior_sig)))
    
    return log_likelihood_ + log_prior_ + log_specz_prior_

# input shape of hyperparameters should be (n_walkers, n_hyperparameters), output shape should be (n_walkers)
#@tf.function
#def log_hyperparameter_conditional(hyperparameters, N, z):
    
    # split the hyper parameters
#    N0, sigmaN = tf.split(hyperparameters, (1, 1), axis=-1)
#    log10M = (N - distance_modulus(z)) / (-2.5)

    # prior terms
#    log_prior_ = tf.reduce_sum(mass_function_log_prob(log10M, z) - tf.math.log(1. + tf.exp((N - N0)/sigmaN)), -1) - z.shape[0] * log_N_prior_normalization_emulator(hyperparameters)
    
#    return tf.reduce_sum(log_prior_, -1) + tf.reduce_sum(hyper_parameter_prior.log_prob(hyperparameters), axis=-1)

@tf.function
def log_nuisanceparameter_conditional(nuisanceparameters, model_fluxes, fluxes, flux_variances):

    # split the hyper parameters
    zero_points, additive_log_fractional_errors = tf.split(nuisanceparameters, (n_bands, n_bands), axis=-1)
    zero_points = tf.expand_dims(zero_points, axis=1)
    additive_fractional_errors = tf.expand_dims(tf.exp(additive_log_fractional_errors), axis=1)
    
    # compute predicted fluxes from model fluxes and hyper parameters (ie., scale by zero-points)
    predicted_fluxes = tf.multiply(zero_points, model_fluxes)
    
    # flux variances (quoted variances, scaled by zero points, plus additional fractional error term)
    predicted_flux_variances = tf.add(flux_variances, tf.square(tf.multiply(additive_fractional_errors, predicted_fluxes)))

    # log likelihoods
    log_likelihood_ = log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances)
    
    # log prior
    #log_prior_ = nuisance_parameter_prior.log_prior(hyperparameters)

    # ignore any nans
    log_likelihood_ = tf.where(tf.math.is_nan(log_likelihood_), 0., log_likelihood_)

    return tf.reduce_sum(log_likelihood_, axis=-1)# + log_prior

# initial walker states
n_latent_walkers = 400
n_hyper_walkers = 400
n_nuisance_walkers = 400

# initialize latent
if burnin is False:
    initial_latent_chain = np.load('/cfs/home/alju5794/steppz/kids/initializations/latent1.npy').astype(np.float32)
    latent_current_state = [sps_prior.bijector.inverse(initial_latent_chain[0:n_latent_walkers,...]), sps_prior.bijector.inverse(initial_latent_chain[n_latent_walkers:2*n_latent_walkers,...])] 
else:
    latent_current_state = [tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi_GAMA.npy')[0:n_latent_walkers,:,:].astype(np.float32), dtype=tf.float32), tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/kids/initializations/B_walkers_phi.npy')[n_latent_walkers:2*n_latent_walkers,:,:].astype(np.float32), dtype=tf.float32)]

# initialize hyper-parameters
#if burnin is False:
#    initial_hyper_chain = np.load('/cfs/home/alju5794/steppz/kids/initializations/hyper0.npy').astype(np.float32)
#    hyper_parameters_ = tf.convert_to_tensor(initial_hyper_chain[0,:], dtype=tf.float32)
#    hyper_current_state = [tf.convert_to_tensor(initial_hyper_chain[0:n_hyper_walkers,...], dtype=tf.float32), tf.convert_to_tensor(initial_hyper_chain[n_hyper_walkers:2*n_hyper_walkers,...], dtype=tf.float32)]
#else:
#    hyper_parameters_ = tf.convert_to_tensor(np.array([13.5, 0.2]).astype(np.float32), dtype=tf.float32)
#    hyper_current_state = [hyper_parameters_ + tf.random.normal([n_hyper_walkers, hyper_parameters_.shape[0]], 0, [1e-1, 1e-2]), hyper_parameters_ + tf.random.normal([n_hyper_walkers, hyper_parameters_.shape[0]], 0, [1e-1, 1e-2])]

# initialize nuisance parameters
nuisance_parameters_ = tf.convert_to_tensor(np.concatenate([np.ones([n_bands]), np.log(model_error + zp_error)]), dtype=tf.float32)
nuisance_current_state = [nuisance_parameters_ + tf.random.normal([n_nuisance_walkers, nuisance_parameters_.shape[0]], 0, 5e-3), nuisance_parameters_ + tf.random.normal([n_nuisance_walkers, nuisance_parameters_.shape[0]], 0, 5e-3)]

# set up batching of latent parameters
n_latent = fluxes.shape[0] # number of galaxies
latent_batch_size = 1000
n_latent_batches = n_latent // latent_batch_size + int( (n_latent % latent_batch_size) > 0)
batch_indices = [np.arange(latent_batch_size*i, min(latent_batch_size*(i+1), n_latent)) for i in range(n_latent_batches)]

# how many MCMC steps?
n_steps = 600
n_latent_sub_steps = 10
n_hyper_sub_steps = 10
n_nuisance_sub_steps = 10
n_nz_sub_steps = 10
n_latent_burnin_steps = 500
n_hyper_burnin_steps = 500
n_nuisance_burnin_steps = 500

if burnin is True:

    # burn in with zs fixed...

    # burn in latent parameters, conditioned on hyper-parameters (do it in batches and concatenate them together)
    latent_samples_ = tf.concat([affine_sample_batch_state(log_latentparameter_conditional, 
                                                     n_latent_burnin_steps, 
                                                     [tf.gather(latent_current_state[0], batch_indices[_], axis=1), tf.gather(latent_current_state[1], batch_indices[_], axis=1)], 
                                                     args=[hyper_parameters_, nuisance_parameters_, tf.gather(fluxes, batch_indices[_], axis=0), tf.gather(flux_variances, batch_indices[_], axis=0), tf.gather(zspec, batch_indices[_], axis=0), tf.gather(zprior_sig_fixed, batch_indices[_], axis=0)], tensor=True) for _ in range(n_latent_batches)], axis=1)
    latent_current_state = tf.split(latent_samples_, (n_latent_walkers, n_latent_walkers), axis=0) # set current walkers state
    latent_parameters_ = latent_current_state[np.random.randint(0, 2)][np.random.randint(0, n_latent_walkers),...] # latent-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # compute model fluxes for latent parameters that we'll now condition on (which remain fixed during the hyper-parameter sampling step)
    theta = sps_prior.bijector(latent_parameters_)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    z = theta[...,-1] # redshifts
    model_fluxes = tf.concat([emulator.fluxes(transform(tf.gather(theta[...,1:], batch_indices[_], axis=0)), tf.gather(N, batch_indices[_], axis=0)) for _ in range(n_latent_batches)], axis=0)        

    # sample nuisance-parameters, conditioned on latent parameters
    nuisance_samples_ = affine_sample(log_nuisanceparameter_conditional, n_nuisance_burnin_steps, nuisance_current_state, args=[model_fluxes, fluxes, flux_variances])
    nuisance_current_state = tf.split(nuisance_samples_[-1,...], (n_nuisance_walkers, n_nuisance_walkers), axis=0) # set current walkers state
    nuisance_parameters_ = nuisance_current_state[np.random.randint(0, 2)][np.random.randint(0, n_nuisance_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # hyper-parameter sampling
#    hyper_samples_ = affine_sample(log_hyperparameter_conditional, n_hyper_burnin_steps, hyper_current_state, args=[N, z])
#    hyper_current_state = tf.split(hyper_samples_[-1,...], (n_hyper_walkers, n_hyper_walkers), axis=0) # set current walkers state
#    hyper_parameters_ = hyper_current_state[np.random.randint(0, 2)][np.random.randint(0, n_hyper_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # save the chain
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/latent{}.npy'.format(0), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),...] )
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/z{}.npy'.format(0), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),:,-1] )
#    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/hyper{}.npy'.format(0), hyper_samples_[-1,...].numpy().astype(np.float32))
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/nuisance{}.npy'.format(0), nuisance_samples_[-1,...].numpy().astype(np.float32))

    # burn in with zs free...

    # burn in latent parameters, conditioned on hyper-parameters (do it in batches and concatenate them together)
    latent_samples_ = tf.concat([affine_sample_batch_state(log_latentparameter_conditional, 
                                                     n_latent_burnin_steps, 
                                                     [tf.gather(latent_current_state[0], batch_indices[_], axis=1), tf.gather(latent_current_state[1], batch_indices[_], axis=1)], 
                                                     args=[hyper_parameters_, nuisance_parameters_, tf.gather(fluxes, batch_indices[_], axis=0), tf.gather(flux_variances, batch_indices[_], axis=0), tf.gather(zspec, batch_indices[_], axis=0), tf.gather(zprior_sig, batch_indices[_], axis=0)], tensor=True) for _ in range(n_latent_batches)], axis=1)
    latent_current_state = tf.split(latent_samples_, (n_latent_walkers, n_latent_walkers), axis=0) # set current walkers state
    latent_parameters_ = latent_current_state[np.random.randint(0, 2)][np.random.randint(0, n_latent_walkers),...] # latent-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # compute model fluxes for latent parameters that we'll now condition on (which remain fixed during the hyper-parameter sampling step)
    theta = sps_prior.bijector(latent_parameters_)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    z = theta[...,-1] # redshifts
    model_fluxes = tf.concat([emulator.fluxes(transform(tf.gather(theta[...,1:], batch_indices[_], axis=0)), tf.gather(N, batch_indices[_], axis=0)) for _ in range(n_latent_batches)], axis=0)        

    # sample nuisance-parameters, conditioned on latent parameters
    nuisance_samples_ = affine_sample(log_nuisanceparameter_conditional, n_nuisance_burnin_steps, nuisance_current_state, args=[model_fluxes, fluxes, flux_variances])
    nuisance_current_state = tf.split(nuisance_samples_[-1,...], (n_nuisance_walkers, n_nuisance_walkers), axis=0) # set current walkers state
    nuisance_parameters_ = nuisance_current_state[np.random.randint(0, 2)][np.random.randint(0, n_nuisance_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # hyper-parameter sampling
  #  hyper_samples_ = affine_sample(log_hyperparameter_conditional, n_hyper_burnin_steps, hyper_current_state, args=[N, z])
  #  hyper_current_state = tf.split(hyper_samples_[-1,...], (n_hyper_walkers, n_hyper_walkers), axis=0) # set current walkers state
  #  hyper_parameters_ = hyper_current_state[np.random.randint(0, 2)][np.random.randint(0, n_hyper_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # save the chain
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/latent{}.npy'.format(1), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),...] )
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/z{}.npy'.format(1), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),:,-1] )
  #  np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/hyper{}.npy'.format(1), hyper_samples_[-1,...].numpy().astype(np.float32))
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/nuisance{}.npy'.format(1), nuisance_samples_[-1,...].numpy().astype(np.float32))

# main chain...

# loop over batches
for step in range(n_steps):

    # burn in latent parameters, conditioned on hyper-parameters (do it in batches and concatenate them together)
    latent_samples_ = tf.concat([affine_sample_batch_state(log_latentparameter_conditional, 
                                                     n_latent_sub_steps, 
                                                     [tf.gather(latent_current_state[0], batch_indices[_], axis=1), tf.gather(latent_current_state[1], batch_indices[_], axis=1)], 
                                                     args=[hyper_parameters_, nuisance_parameters_, tf.gather(fluxes, batch_indices[_], axis=0), tf.gather(flux_variances, batch_indices[_], axis=0), tf.gather(zspec, batch_indices[_], axis=0), tf.gather(zprior_sig, batch_indices[_], axis=0)], tensor=True) for _ in range(n_latent_batches)], axis=1)
    latent_current_state = tf.split(latent_samples_, (n_latent_walkers, n_latent_walkers), axis=0) # set current walkers state
    latent_parameters_ = latent_current_state[np.random.randint(0, 2)][np.random.randint(0, n_latent_walkers),...] # latent-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # compute model fluxes for latent parameters that we'll now condition on (which remain fixed during the hyper-parameter sampling step)
    theta = sps_prior.bijector(latent_parameters_)
    N = theta[...,0] # extract normalization parameter N = -2.5log10M + dm(z)
    z = theta[...,-1] # redshifts
    model_fluxes = tf.concat([emulator.fluxes(transform(tf.gather(theta[...,1:], batch_indices[_], axis=0)), tf.gather(N, batch_indices[_], axis=0)) for _ in range(n_latent_batches)], axis=0)        

    # sample nuisance-parameters, conditioned on latent parameters
    nuisance_samples_ = affine_sample(log_nuisanceparameter_conditional, n_nuisance_sub_steps, nuisance_current_state, args=[model_fluxes, fluxes, flux_variances])
    nuisance_current_state = tf.split(nuisance_samples_[-1,...], (n_nuisance_walkers, n_nuisance_walkers), axis=0) # set current walkers state
    nuisance_parameters_ = nuisance_current_state[np.random.randint(0, 2)][np.random.randint(0, n_nuisance_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # hyper-parameter sampling
 #   hyper_samples_ = affine_sample(log_hyperparameter_conditional, n_hyper_sub_steps, hyper_current_state, args=[N, z])
 #   hyper_current_state = tf.split(hyper_samples_[-1,...], (n_hyper_walkers, n_hyper_walkers), axis=0) # set current walkers state
 #   hyper_parameters_ = hyper_current_state[np.random.randint(0, 2)][np.random.randint(0, n_hyper_walkers),...] # hyper-parameters to condition on for next Gibbs step (chosen randomly from walkers)

    # save the chain
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/latent{}.npy'.format(step+2), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),...] )
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/z{}.npy'.format(step+2), sps_prior.bijector(latent_samples_).numpy().astype(np.float32)[np.random.randint(0, 2*n_latent_walkers, n_thin),:,-1] )
 #   np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/hyper{}.npy'.format(step+2), hyper_samples_[-1,...].numpy().astype(np.float32))
    np.save('/cfs/home/alju5794/steppz/kids/chains/B_BHM_II_GAMA/nuisance{}.npy'.format(step+2), nuisance_samples_[-1,...].numpy().astype(np.float32))

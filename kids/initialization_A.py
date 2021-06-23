import numpy as np

import sys
sys.path.append('/cfs/home/alju5794/steppz/code')
from utils import *
from priors import *
from likelihoods import *

# import training data

# import the mags and thetas
training_theta = np.concatenate([np.load('/cfs/home/alju5794/steppz/sps_models/model_A/training_data/parameters/parameters{}.npy'.format(i)) for i in range(32)], axis=0)
training_mags = np.concatenate([np.load('/cfs/home/alju5794/steppz/sps_models/model_A/training_data/photometry/KV_photometry{}.npy'.format(i)).astype(np.float32) for i in range(32)], axis=0) # units: nanomaggies

# transform to normalization parameter
training_theta[:,0] = -2.5*training_theta[:,0] + distance_modulus(training_theta[:,-1].astype(np.float32))

# convert absolute mags to apparent and then flux
training_mags = training_mags + np.expand_dims(training_theta[:,0], -1)
training_flux = 10**(-0.4*training_mags + 9.)
training_flux = training_flux.astype(np.float32)

# puffing factor for gaussian errors
puff = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.0, 1.0])

# assumed fractional model error per band
model_error = tf.constant([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# assumed ZP (fractional) error per band
zp_error = tf.constant([0.05, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=tf.float32)

# import data (with speczs)
flux_sigmas = np.load('/cfs/home/alju5794/steppz/kids/data/flux_errors_no_missing.npy')*(1e9)*puff # units: nanomaggies
fluxes = np.load('/cfs/home/alju5794/steppz/kids/data/fluxes_no_missing.npy')*1e9 # units: nanomaggies
zspec = np.load('/cfs/home/alju5794/steppz/kids/data/zspec_no_missing.npy')
zb = np.load('/cfs/home/alju5794/steppz/kids/data/zb_no_missing.npy')
specsource = np.load('/cfs/home/alju5794/steppz/kids/data/specsource_no_missing.npy')

# cut out dodgy values
cut = (fluxes < 1e9).all(axis=1) * (zspec < 2.0) * (zspec > 1e-3) * (specsource != 'CDFS') * (specsource != 'VVDS')
fluxes = fluxes[cut,:]
flux_sigmas = flux_sigmas[cut,:]
zspec = zspec[cut]
zb = zb[cut]
specsource = specsource[cut]

# training data cuts
fmin = fluxes.min(axis=0)
fmax = fluxes.max(axis=0)
cut = (training_flux < fmin).all(axis=1) + (training_flux > fmax).all(axis=1)
training_flux = training_flux[~cut,:]
training_theta = training_theta[~cut,:]

# prior
prior = ModelABBaselinePrior()

# biject the parameters
training_phi = prior.bijector.inverse(training_theta).numpy()

# cut out nans
cut = np.isnan(training_phi).any(axis=1)
training_flux = training_flux[~cut,:]
training_theta = training_theta[~cut,:]
training_phi = training_phi[~cut,:]

# broadcastable fluxes and variances
extra_flux_variance_ = tf.expand_dims(((model_error + zp_error)*training_flux)**2, 1)
predicted_fluxes_ = tf.expand_dims(training_flux, 1)

# holders for estimators
estimator_phi = np.zeros((fluxes.shape[0], training_phi.shape[-1]))
estimator_theta = np.zeros((fluxes.shape[0], training_phi.shape[-1]))

for i in range(fluxes.shape[0]):
    
    fluxes_ = tf.expand_dims(fluxes[i,:], 0)
    predicted_flux_variances_ = flux_sigmas[i,:]**2 + extra_flux_variance_
    logl = log_likelihood_studentst2(fluxes_, predicted_fluxes_, predicted_flux_variances_, tf.ones(9, dtype=tf.float32))
    argmax = tf.math.argmax(logl, axis=0)
    estimator_phi[i,:] = training_phi[argmax,:]
    estimator_theta[i,:] = training_theta[argmax,:]

np.save('/cfs/home/alju5794/steppz/kids/initializations/A_phi0.npy', estimator_phi)
np.save('/cfs/home/alju5794/steppz/kids/initializations/A_theta0.npy', estimator_theta)

log10M_bijector = tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(13.-7.)), tfb.Shift(-7.)]))
z_bijector = tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(2.5-1e-5)), tfb.Shift(-1e-5)]))

n_walkers = 1000
walkers = np.zeros((n_walkers, estimator_phi.shape[0], estimator_phi.shape[1]))
for i in range(estimator_phi.shape[0]):

	# phi walkers
    walkers[:,i,:] = estimator_phi[i,:] + np.random.normal(0, 0.05, size=(n_walkers, estimator_phi.shape[1]))

    # mass estimator
    log10M_0 = (estimator_theta[i,0] - distance_modulus(tf.math.maximum(1e-5, estimator_theta[i,-1])).numpy())/-2.5

    # mass samples in bijected space
    log10M = log10M_bijector(log10M_bijector.inverse(log10M_0).numpy() + np.random.normal(0, 0.05, n_walkers).astype(np.float32)).numpy()

    # convert back to N
    N = -2.5*log10M + distance_modulus(tf.math.maximum(1e-5, z_bijector(walkers[:,i,-1]))).numpy()

    # put into walkers
    walkers[:,i,0] = N
    
np.save('/cfs/home/alju5794/steppz/kids/initializations/A_walkers_phi.npy', walkers)
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

# prior
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

# initial walker states
n_walkers = 500

# initialize latent
current_state = [bijector.inverse(tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/sps_models/model_B/training_data/parameters/parameters30.npy')[0:n_walkers,:,:].astype(np.float32), dtype=tf.float32)), bijector.inverse(tf.convert_to_tensor(np.load('/cfs/home/alju5794/steppz/sps_models/model_B/training_data/parameters/parameters30.npy')[n_walkers:2*n_walkers,:,:].astype(np.float32), dtype=tf.float32))]

# how many MCMC steps?
n_steps = 2000

# sample hyper-parameters, conditioned on latent parameters
prior_samples = affine_sample(sps_prior.log_prob, n_steps, current_state).numpy()[(n_steps // 2):,:,:].reshape((n_steps* n_walkers, n_sps_parameters))

# save the chain
np.save('/cfs/home/alju5794/steppz/kids/chains/B_prior.npy'.format(0), sps_prior.bijector(samples.astype(np.float32)).numpy().astype(np.float32) )
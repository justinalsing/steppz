import os
import sys
os.environ["SPS_HOME"] = '/home/sinandeger/fsps/'
sys.path.append('/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz/steppz/code')

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import cumtrapz
from scipy.stats import gaussian_kde
import scipy.stats as stats
from getdist import plots, MCSamples
import matplotlib as mpl
from astropy.cosmology import Planck15
from scipy.interpolate import InterpolatedUnivariateSpline
import fsps
import emcee
from scipy.special import hyp2f1

import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm, trange
tfb = tfp.bijectors
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from plotting import triangle_plot
from utils import *
# from priors import *
from population_prior import ModelCBaselinePrior, log10sSFRpriorMizuki, redshift_volume_prior
from affine import affine_sample
from ndes import RegressionNetwork

"""This module samples from the baseline population prior"""
cwd = os.getcwd()
pro_alpha_model_path = '/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz/steppz/sps_models/model_Prospector-alpha/'

# import the relevant models
log10sSFR_emulator = RegressionNetwork(restore=True, restore_filename=pro_alpha_model_path+'ProspectorAlpha_log10sSFR_emulator.pkl')
baseline_SFR_prior_log_prob = RegressionNetwork(restore=True, restore_filename=pro_alpha_model_path+'ProspectorAlpha_baseline_SFR_prior_logprob.pkl')

# set up the prior class
Prior = ModelCBaselinePrior(baselineSFRprior=baseline_SFR_prior_log_prob,
                            log10sSFRemulator=log10sSFR_emulator,
                            SFSprior=None,
                            log10sSFRuniformlimits=tfd.Uniform(low=-14, high=-8),
                            redshift_prior=redshift_volume_prior)


# initialize walkers for sampling
n_walkers = 2000
n_steps = 1000

# baseline prior draws
bijector = tfb.Blockwise([tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()),
                                                tfb.Scale(1./(Prior.upper[_]-Prior.lower[_])),
                                                tfb.Shift(-Prior.lower[_])])) for _ in range(Prior.n_sps_parameters)])
baseline_draws = bijector(Prior.baselinePrior.sample((30000, Prior.n_sps_parameters)))

# reject those outside SFR prior range
sfh = tf.gather(baseline_draws, [2, 3, 4, 5, 6, 7, 1, 11], axis=-1)
log10sSFR = tf.squeeze(log10sSFR_emulator(sfh))
baseline_draws = tf.squeeze(tf.gather(baseline_draws, indices=tf.where((log10sSFR > -14) & (log10sSFR < -8)), axis=0), axis=1)

# convert log10M to N
baseline_draws = baseline_draws.numpy()
baseline_draws[...,0] = -2.5*baseline_draws[...,0] + distance_modulus(tf.math.maximum(1e-5, baseline_draws[...,-1]))
log_prior = Prior.log_prob(Prior.bijector.inverse(baseline_draws)).numpy()
baseline_draws = baseline_draws[~np.isinf(log_prior), :]
# baseline_draws = tf.convert_to_tensor(Prior.bijector.inverse(baseline_draws))

# convert physical to bijected parameter space (to get rid of hard boundaries)
latent_baseline_draws = Prior.bijector.inverse(baseline_draws)

# current state
current_state = [latent_baseline_draws[0:n_walkers,:], latent_baseline_draws[n_walkers:2*n_walkers,:]]

n_batches = 64
n_samples = 100000

# burn in
chain = affine_sample(Prior.log_prob, 1000, current_state)
current_state = [chain[-1, 0:n_walkers, :], chain[-1, n_walkers:, :]]

for batch in tqdm(range(n_batches)):
    chain = affine_sample(Prior.log_prob, 25, current_state)
    current_state = [chain[-1, 0:n_walkers, :], chain[-1, n_walkers:, :]]
    parameters = Prior.bijector(chain).numpy().reshape((25 * 2 * n_walkers, 12))  # need to modify the hard-coded 12 here

    log10M = (parameters[..., 0] - distance_modulus(parameters[..., -1]).numpy()) / -2.5
    plt.hist(log10M, bins=100)
    # plt.yscale('log')
    plt.show()

    # save the parameters
    parameter_output_dir = cwd+'/parameters'
    if not os.path.exists(parameter_output_dir):
        os.makedirs(parameter_output_dir)

    np.save(parameter_output_dir+'/parameters{}.npy'.format(batch), parameters)
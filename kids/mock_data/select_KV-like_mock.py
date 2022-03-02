
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

# merge all the data
for i in range(int(sys.argv[1])):


    f = open('/cfs/home/alju5794/steppz/kids/mock_data/KV-like_mock{}.pkl'.format(i), 'wb')
    photometry, errors, selection, thetas = pickle.load(f)
    f.close()

    if i == 0:
        ind = np.where(selection == 1.)[0]
        fluxes = photometry[ind,:]
        flux_sigmas = errors[ind,:]
        parameters = thetas[ind,:]
        zspec = thetas[ind,-1]
    else:
        ind = np.where(selection == 1.)[0]
        fluxes = np.concatenate([fluxes, photometry[ind,:]])
        flux_sigmas = np.concatenate([flux_sigmas, errors[ind,:]])
        parameters = np.concatenate([parameters, thetas[ind,:]])
        zspec = np.concatenate([zspec, thetas[ind,-1]])

# fir the true n(z)

# paramerers
logits = tf.Variable(np.array([0.67322165, 1.3267788]).astype(np.float32))
locs = tf.Variable(np.array([0.1756928, 0.653366]).astype(np.float32))
scales = tf.Variable(np.array([0.10497721, 0.19987684]).astype(np.float32))
skewness = tf.Variable(np.array([0.5169175 , 0.23451573]).astype(np.float32))
tailweight = tf.Variable(np.array([0.346626 , 1.3105514]).astype(np.float32))

# mixture model
nz = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logits),
                          components_distribution=tfd.SinhArcsinh(loc=locs, scale=scales, skewness=skewness, tailweight=tailweight))

optimizer = tf.keras.optimizers.Adam(lr=0.01)

epochs = 600   
    
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        tape.watch([logits, locs, scales, skewness, tailweight])
        negative_log_likelihood = -tf.reduce_sum(nz.log_prob(zspec) - tf.math.log(1.-nz.cdf(0)))

    gradients = tape.gradient(negative_log_likelihood, [logits, locs, scales, skewness, tailweight])
    optimizer.apply_gradients(zip(gradients, [logits, locs, scales, skewness, tailweight]))        
    t.set_postfix(loss=negative_log_likelihood.numpy())

# construct spec-z subsample

# importance weights
weights = stats.gamma(a=2., scale=0.25).pdf(zspec) / (nz.prob(zspec).numpy() / (1. - nz.cdf(0.).numpy()))
weights = weights / sum(weights)

# make the selection
ind = np.random.choice(np.arange(len(zspec)), p=weights, replace=False, size=int(sys.argv[2]))

# redshift errors
zprior_sigma = 10*np.ones(zspec.shape[0])
zprior_sigma[ind] = 0.001

# now pickle everything
pickle.dump([fluxes, flux_sigmas, zspec, zprior_sigma, parameters], open('/cfs/home/alju5794/steppz/kids/mock_data/KV450-like_mock-selected.pkl', 'wb') )




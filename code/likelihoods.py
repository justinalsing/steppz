import numpy as np
import tensorflow as tf

@tf.function
def log_likelihood_studentst2(fluxes, flux_variances, predicted_fluxes, predicted_flux_variances, n_sigma_flux_cuts):
    
    # variances -> sigmas
    predicted_flux_sigmas = tf.math.sqrt(predicted_flux_variances)
    flux_sigmas = tf.math.sqrt(flux_variances)
    
    # flux cuts
    flux_cuts = tf.multiply(n_sigma_flux_cuts, flux_sigmas)
    
    # log-likelihood (Student's-t with 2 d.o.f)
    log_likelihood_ = tf.reduce_sum(tf.add(tf.multiply(-1.5, tf.math.log(tf.add(1., 0.5 * tf.divide(tf.square(tf.subtract(predicted_fluxes, fluxes)), predicted_flux_variances)))), -tf.math.log(predicted_flux_sigmas) ), axis=-1)
    
    # selection cuts (CDF of the Student's-t for each band given the flux limits)
    log_selection_ = tf.reduce_sum(-tf.math.log(tf.subtract(0.5, tf.divide(tf.divide(tf.subtract(flux_cuts, predicted_fluxes), predicted_flux_sigmas), tf.multiply(tf.math.sqrt(8.), tf.math.sqrt(tf.add(1., tf.multiply(0.5, tf.square(tf.divide(tf.subtract(flux_cuts, predicted_fluxes), predicted_flux_sigmas))))))))), axis=-1)
    
    return log_likelihood_ + log_selection_

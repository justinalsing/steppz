import numpy as np
import tensorflow as tf

@tf.function
def log_likelihood_studentst2(fluxes, predicted_fluxes, predicted_flux_variances):
    
    # variances -> sigmas
    predicted_flux_sigmas = tf.math.sqrt(predicted_flux_variances)
    
    # log-likelihood (Student's-t with 2 d.o.f)
    log_likelihood_ = tf.reduce_sum(tf.add(tf.multiply(-1.5, tf.math.log(tf.add(1., 0.5 * tf.divide(tf.square(tf.subtract(predicted_fluxes, fluxes)), predicted_flux_variances)))), -tf.math.log(predicted_flux_sigmas) ), axis=-1)

    return log_likelihood_

@tf.function
def log_likelihood_normal(fluxes, predicted_fluxes, predicted_flux_variances):
    
    # variances -> sigmas
    predicted_flux_sigmas = tf.math.sqrt(predicted_flux_variances)
    
    # log-likelihood (Student's-t with 2 d.o.f)
    log_likelihood_ = tf.reduce_sum(tf.add(tf.multiply(-0.5, tf.divide(tf.square(tf.subtract(predicted_fluxes, fluxes)), predicted_flux_variances)), -tf.math.log(predicted_flux_sigmas) ), axis=-1)

    return log_likelihood_
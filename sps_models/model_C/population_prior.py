import os
import sys
sys.path.append('/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz/steppz/code')

os.environ["SPS_HOME"] = '/home/sinandeger/fsps/'
# sys.path.insert(0, '/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz/steppz/sps_models/model_B')
# from sfh import sfh
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
"""Import necessary modules from steppz"""
from utils import ln10_, dVdz, halfln2pi_, distance_modulus


"""This module hosts the functions that will be used when sampling from the population prior"""
# double Schechter mass function from Leja+2020

z1_ = tf.constant(0.2)
z2_ = tf.constant(1.6)
z3_ = tf.constant(3.0)

anchor_points_logphi1_ = tf.constant(np.array([-2.44, -3.08, -4.14]).astype(np.float32))
anchor_points_logphi2_ = tf.constant(np.array([-2.89, -3.29, -3.51]).astype(np.float32))
anchor_points_M_star_ = tf.constant(np.array([10.79, 10.88, 10.84]).astype(np.float32))

# Joel
alpha1_ = tf.constant(-0.28)
alpha2_ = tf.constant(-1.48)

# GAMA
#alpha1_ = tf.constant(-0.466)
#alpha2_ = tf.constant(-1.53)

# Joel's SMS prior parameters
joel_SFS_parameters_ = tf.constant(np.array([-0.15040097,  0.9800668 , -0.50802046,  1.0515388 , -0.28611764,
                                             0.02131329,  0.05053138,  1.0766244 , -0.02015052, -0.13125503,
                                             0.7205097 , -0.18212801,  1.5429502 , -1.5872463 , -0.04843145,
                                             0.65359867,  0.92735046, -0.17695354, 10.442122  ,  0.56389964,
                                             0.7500511 ,  2.0604856 , 10.335057  , -0.3050156 ,  0.5491848 ,
                                             10.611108  ,  0.08009169, -0.06575003,  0.3912887 ,  0.54855245,
                                             0.44964817, 11.159543  ,  0.11614972, -1.5658572 ]).astype(np.float32))


# Star forming main sequence SFR prior from Leja+2022
@tf.function
def log10SFRpriorJoel(log10SFR, log10M, z):
    # extract trainable variables
    a_, b_, C_, d_, e_, f_, g_, h_, i_, j_, log10Mt_sf_, log10Mt_q_, sigma_sf, sigma_q0, sigma_q1, sigma_qt, sigma_qs, skew_q = tf.split(
        joel_SFS_parameters_, (3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1))

    Z = tf.stack([tf.ones(z.shape[0]), z, z ** 2], axis=-1)

    a = tf.tensordot(Z, a_, axes=1)
    b = tf.tensordot(Z, b_, axes=1)
    c = tf.tensordot(Z, C_, axes=1)
    log10Mt_sf = tf.tensordot(Z, log10Mt_sf_, axes=1)

    d = tf.tensordot(Z, d_, axes=1)
    e = tf.tensordot(Z, e_, axes=1)
    f = tf.tensordot(Z, f_, axes=1)
    log10Mt_q = tf.tensordot(Z, log10Mt_q_, axes=1)

    # GMM parameters
    sigma_q = sigma_q0 + (sigma_q1 - sigma_q0) * tf.sigmoid((log10M - sigma_qt) / sigma_qs)
    mu_sf = (a * tf.cast(log10M > log10Mt_sf, dtype=tf.float32) + b * tf.cast(log10M < log10Mt_sf,
                                                                              dtype=tf.float32)) * (
                        log10M - log10Mt_sf) + c
    mu_q = (d * tf.cast(log10M > log10Mt_q, dtype=tf.float32) + e * tf.cast(log10M < log10Mt_q, dtype=tf.float32)) * (
                log10M - log10Mt_q) + f - 1.
    r_q = tf.sigmoid(j_) * tf.sigmoid((log10M - (g_ + h_ * z)) / i_)

    p_sf = tfd.Normal(loc=mu_sf, scale=sigma_sf)
    p_q = tfd.SinhArcsinh(loc=mu_q, scale=sigma_q, skewness=skew_q)

    lnp = tf.math.log(r_q * p_q.prob(log10SFR) + (1 - r_q) * p_sf.prob(log10SFR) + 1e-24)

    return lnp


@tf.function
def anchor_points_to_coefficients(anchor_points):
    a = (anchor_points[2] - anchor_points[0] + (anchor_points[1] - anchor_points[0]) * (z1_ - z3_) / (z2_ - z1_)) / (
                z3_ ** 2 - z1_ ** 2 + (z2_ ** 2 - z1_ ** 2) * (z1_ - z3_) / (z2_ - z1_))
    b = (anchor_points[1] - anchor_points[0] - a * (z2_ ** 2 - z1_ ** 2)) / (z2_ - z1_)
    c = anchor_points[0] - a * z1_ ** 2 - b * z1_

    return a, b, c


@tf.function
def compute_phi12_mstar(z):
    a1, b1, c1 = anchor_points_to_coefficients(anchor_points_logphi1_)
    a2, b2, c2 = anchor_points_to_coefficients(anchor_points_logphi2_)
    aM, bM, cM = anchor_points_to_coefficients(anchor_points_M_star_)

    phi1 = 10 ** (a1 * z ** 2 + b1 * z + c1)
    phi2 = 10 ** (a2 * z ** 2 + b2 * z + c2)
    Mstar = aM * z ** 2 + bM * z + cM

    return phi1, phi2, Mstar


@tf.function
def mass_function_log_prob(log10M, z):
    phi1, phi2, Mstar = compute_phi12_mstar(z)

    # normalization: NOTE THIS NORMALIZES THE MASS FUNCTION OVER LOG10M [7, 13] UP TO Z = 2.5
    # log_normalization = tf.math.log(-0.00746909 * z**3 + 0.04891415 * z**2 - 0.13758815*z + 0.20274272)

    return tf.math.log(ln10_ * (
                phi1 * 10 ** ((log10M - Mstar) * (alpha1_ + 1)) * tf.exp(-10 ** (log10M - Mstar)) + phi2 * 10 ** (
                    (log10M - Mstar) * (alpha2_ + 1)) * tf.exp(-10 ** (log10M - Mstar))))  # - log_normalization


# Mass-metallicity relation Gallazzi+
@tf.function
def metallicity_mass_log_prob(log10Z, log10M):
    log10Z_mu = -0.25933163 + 0.38391743 * tf.tanh(2.0229099 * log10M - 20.37321787)
    log10Z_sigma = 0.6883885 - 0.37122853 * tf.tanh(2.47629773 * log10M - 25.74109587)

    return - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(log10Z, log10Z_mu), log10Z_sigma))) - tf.add(halfln2pi_,
                                                                                                           tf.math.log(
                                                                                                               log10Z_sigma)) - tf.math.log(
        tf.math.erf((0.19 - log10Z) / (tf.sqrt(2.) * log10Z_sigma)) - tf.math.erf(
            (-1.98 - log10Z) / (tf.sqrt(2.) * log10Z_sigma)))


# Star forming main sequence SFR prior from Mizuki (Tanaka+2015)
@tf.function
def log10sSFRpriorMizuki(log10sSFR, z):
    # star forming galaxies
    mu_log10sSFR = tf.cast(z <= 2., tf.float32) * (2.1 * tf.math.log(1. + z) / ln10_ - 10.) + tf.cast(z > 2.,
                                                                                                      tf.float32) * (
                               1.5 * tf.math.log(1 + z) / ln10_ - 11. + tf.math.log(19.) / ln10_)
    sigma_log10sSFR = 0.3

    # quiescent galaxies (fixed peak)
    mu_log10sSFR_Q = -11.8
    sigma_log10sSFR_Q = 0.36

    return tf.math.log(0.5 * tf.exp(-0.5 * (log10sSFR - mu_log10sSFR) ** 2 / sigma_log10sSFR ** 2 - 0.5 * tf.math.log(
        2 * np.pi * sigma_log10sSFR ** 2)) + 0.5 * tf.exp(
        -0.5 * (log10sSFR - mu_log10sSFR_Q) ** 2 / sigma_log10sSFR_Q ** 2 - 0.5 * tf.math.log(
            2 * np.pi * sigma_log10sSFR_Q ** 2)) + 1e-32)


# volume redshift prior
@tf.function
def redshift_volume_prior(z):
    return tf.math.log(dVdz(z))


@tf.function
def curtiFMR(log10Z, log10M, log10sSFR):
    Z0 = 8.78
    gamma = 0.3
    m0 = 10.1
    m1 = 0.56
    beta = 2.1
    mean = Z0 - (gamma / beta) * tf.math.log(
        1 + 10 ** (-beta * (log10M - (m0 + m1 * (log10sSFR + log10M))))) / ln10_ - 8.69
    sigma = 0.06

    return -0.5 * (log10Z - mean) ** 2 / sigma ** 2


class ModelCBaselinePrior:

    def __init__(self, baselineSFRprior=None, log10sSFRemulator=None, log10sSFRuniformlimits=None, redshift_prior=None,
                 SFSprior=None, FMRprior=None):

        # parameters and limits
        self.parameter_names = ['N', 'gaslog10Z'] +\
                               ['logsfr_ratio{}'.format(i) for i in range(1, 7)] +\
                               ['dust2', 'dust_index', 'dust1_fraction', 'z']

        self.n_sps_parameters = len(self.parameter_names)

        self.lower = tf.constant([7., -1.98, -5., -5., -5., -5., -5., -5., 0., -1., 0., 0.], dtype=tf.float32)
        self.upper = tf.constant([13., 0.19, 5., 5., 5., 5., 5., 5., 4., 0.4, 2., 2.5], dtype=tf.float32)

        # bijector from constrained parameter (physical) to unconstrained space for sampling.
        # note: no bijector for the normalization parameter N (since it is already unconstrained)
        self.bijector = tfb.Blockwise([tfb.Identity()] + [tfb.Invert(tfb.Chain(
            [tfb.Invert(tfb.NormalCDF()), tfb.Scale(1. / (self.upper[_] - self.lower[_])), tfb.Shift(-self.lower[_])]))
                                                          for _ in range(1, self.n_sps_parameters)])

        # baseline prior on unconstrained latent parameters
        # note this does not apply to the normalization parameter N
        self.baselinePrior = tfd.Normal(loc=0., scale=1.)

        # mass limits prior
        self.massLimitsPrior = tfd.Uniform(low=7., high=13.)

        # star formation history parameters prior: import conditional density estimator model for P(SFH | z)
        self.baselineSFRprior = baselineSFRprior
        self.log10sSFRemulator = log10sSFRemulator
        self.log10sSFRuniformlimits = log10sSFRuniformlimits

        # redshift prior
        self.redshift_prior = redshift_prior
        self.SFSprior = SFSprior
        self.FMRprior = FMRprior

    # @tf.function
    def log_prob(self, latentparameters):

        # convert parameters...

        # biject unconstrained latent parameters to physical parameters
        theta = self.bijector(latentparameters)

        # split up the parameters to make things easier to read
        N, gas_logz, logsfr_ratios, dust2, dust_index, dust1_fraction, z = tf.split(theta, (1, 1, 6, 1, 1, 1, 1), axis=-1)

        # convert normalization and redshift to logmass
        log10M = (N - distance_modulus(tf.math.maximum(1e-5, z))) / -2.5

        # stacked SFH parameters
        sfh = tf.concat([logsfr_ratios, gas_logz, z], axis=-1)

        # compute prior log density...

        # initialise log target density to baseline prior (unit normal prior on unconstrained parameters:
        # NB not applied to normalization parameter N which is not bijected)
        logp = tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[..., 1:]), axis=-1, keepdims=True)

        # logmass prior
        logp = logp + mass_function_log_prob(log10M, z) + self.massLimitsPrior.log_prob(log10M)

        # metallicity prior
        logp = logp + metallicity_mass_log_prob(gas_logz, log10M)

        # logsfr ratio priors (student's-t with 2 d.o.f)
        logp = logp + tf.reduce_sum(
            tf.multiply(-1.5, tf.math.log(tf.add(1., tf.multiply(0.5, tf.square(tf.divide(logsfr_ratios, 0.3)))))),
            axis=-1, keepdims=True)

        # dust2 prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust2, 0.3), 1.0)))

        # dust index prior
        logp = logp - tf.multiply(0.5, tf.square(
            tf.divide(tf.subtract(dust_index, -0.095 + 0.111 * dust2 - 0.0066 * tf.square(dust2)), 0.4)))

        # dust1 fraction prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust1_fraction, 1.), 0.3)))

        # log fagn prior
        ### uniform only ###

        # log agn tau prior
        ### uniform only ###

        # gas logZ prior
        ### uniform only ###

        # squeeze
        logp = tf.squeeze(logp, axis=-1)

        # SFH prior
        if self.SFSprior is not None:

            if self.SFSprior == 'mizuki':
                log10sSFR = tf.squeeze(self.log10sSFRemulator(sfh), -1)
                baseline_SFR_prior_logprob = tf.squeeze(self.baselineSFRprior(tf.expand_dims(log10sSFR, -1)), -1)  # returns log prob
                target_SFR_prior_logprob = log10sSFRpriorMizuki(log10sSFR, tf.squeeze(z))  # returns log prob
                uniform_SFR_limits = self.log10sSFRuniformlimits.log_prob(log10sSFR)

                logp = logp + target_SFR_prior_logprob - baseline_SFR_prior_logprob + uniform_SFR_limits

            if self.SFSprior == 'leja':
                log10sSFR = tf.squeeze(self.log10sSFRemulator(sfh), -1)
                log10SFR = tf.squeeze(log10M) + log10sSFR
                baseline_SFR_prior_logprob = tf.squeeze(self.baselineSFRprior(tf.expand_dims(log10sSFR, -1)), -1)  # returns log prob
                target_SFR_prior_logprob = log10SFRpriorJoel(log10SFR, tf.squeeze(log10M), tf.squeeze(z))  # returns log prob
                uniform_SFR_limits = self.log10sSFRuniformlimits.log_prob(log10sSFR)

                logp = logp + target_SFR_prior_logprob - baseline_SFR_prior_logprob + uniform_SFR_limits

        if self.FMRprior == 'curti':
            log10sSFR = self.log10sSFRemulator(sfh)
            logp = logp + tf.squeeze(curtiFMR(gas_logz, log10M, log10sSFR), -1)

        # z prior
        if self.redshift_prior is not None:
            logp = logp + tf.squeeze(self.redshift_prior(z), -1)

        return logp






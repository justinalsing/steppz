import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import *
tfd = tfp.distributions
tfb = tfp.bijectors

# double Schechter mass function from Leja+2020

z1_ = tf.constant(0.2)
z2_ = tf.constant(1.6)
z3_ = tf.constant(3.0)

anchor_points_logphi1_ = tf.constant(np.array([-2.44, -3.08, -4.14]).astype(np.float32))
anchor_points_logphi2_ = tf.constant(np.array([-2.89, -3.29, -3.51]).astype(np.float32))
anchor_points_M_star_ = tf.constant(np.array([10.79, 10.88, 10.84]).astype(np.float32))

alpha1_ = tf.constant(-0.28)
alpha2_ = tf.constant(-1.48)

@tf.function
def anchor_points_to_coefficients(anchor_points):
    
    a = (anchor_points[2]-anchor_points[0] + (anchor_points[1] - anchor_points[0])*(z1_-z3_)/(z2_-z1_))/(z3_**2 - z1_**2 + (z2_**2-z1_**2)*(z1_-z3_)/(z2_-z1_))
    b = (anchor_points[1]-anchor_points[0] - a*(z2_**2-z1_**2))/(z2_-z1_)
    c = anchor_points[0] - a*z1_**2 - b*z1_
    
    return a, b, c

@tf.function
def compute_phi12_mstar(z):
    
    a1, b1, c1 = anchor_points_to_coefficients(anchor_points_logphi1_)
    a2, b2, c2 = anchor_points_to_coefficients(anchor_points_logphi2_)
    aM, bM, cM = anchor_points_to_coefficients(anchor_points_M_star_)

    phi1 = 10**(a1*z**2 + b1*z + c1)
    phi2 = 10**(a2*z**2 + b2*z + c2)
    Mstar = aM*z**2 + bM*z + cM
    
    return phi1, phi2, Mstar

@tf.function
def mass_function_log_prob(log10M, z):
    
    phi1, phi2, Mstar = compute_phi12_mstar(z)

    # normalization: NOTE THIS NORMALIZES THE MASS FUNCTION OVER LOG10M [7, 13] UP TO Z = 2.5
    log_normalization = tf.math.log(-0.00746909 * z**3 + 0.04891415 * z**2 - 0.13758815*z + 0.20274272)

    return tf.math.log(ln10_ * (phi1*10**((log10M - Mstar)*(alpha1_+1))*tf.exp(-10**(log10M - Mstar)) + phi2*10**((log10M - Mstar)*(alpha2_+1))*tf.exp(-10**(log10M - Mstar)) ) ) - log_normalization


# Mass-metallicity relation Gallazzi+
@tf.function
def metallicity_mass_log_prob(log10Z, log10M):

    log10Z_mu = -0.25933163 + 0.38391743*tf.tanh(2.0229099*log10M - 20.37321787)
    log10Z_sigma = 0.6883885 - 0.37122853*tf.tanh(2.47629773*log10M - 25.74109587)

    return - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(log10Z, log10Z_mu), log10Z_sigma))) - tf.add(halfln2pi_, tf.math.log(log10Z_sigma)) - tf.math.log(tf.math.erf((0.19 - log10Z)/(tf.sqrt(2.)*log10Z_sigma)) - tf.math.erf((-1.98 - log10Z)/(tf.sqrt(2.)*log10Z_sigma)) )


class ProspectorAlphaBaselinePrior:
    
    def __init__(self):
        
        # parameters and limits
        self.n_sps_parameters = 15
        self.parameter_names = ['N', 'log10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1,7)] + ['dust2', 'dust_index', 'dust1_fraction', 'lnfagn', 'lnagntau', 'gaslog10Z', 'z']
        self.lower = tf.constant([7., -1.98, -5., -5., -5., -5., -5., -5., 0., -1., 0., np.log(1e-5),  np.log(5.), -2., 0.], dtype=tf.float32)
        self.upper = tf.constant([13., 0.19, 5., 5., 5., 5., 5., 5., 4., 0.4, 2., np.log(3.), np.log(150.), 0.5, 2.5], dtype=tf.float32)

        # bijector from constrained parameter (physical) to unconstrained space for sampling. 
        # note: no bijector for the normalization parameter N (since it is already unconstrained)
        self.bijector = tfb.Blockwise([tfb.Identity()] + [tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(self.upper[_]-self.lower[_])), tfb.Shift(-self.lower[_])])) for _ in range(1, self.n_sps_parameters)])

        # baseline prior on unconstrained latent parameters
        # note this does not apply to the normalization parameter N
        self.baselinePrior = tfd.Normal(loc=0., scale=1.)

        # mass limits prior
        self.massLimitsPrior = tfd.Uniform(low=7., high=13.)
        
    @tf.function
    def log_prob(self, latentparameters):
        
        # convert parameters...
        
        # biject unconstrained latent parameters to physical parameters
        theta = self.bijector(latentparameters)

        # split up the parameters to make things easier to read
        N, log10Z, logsfr_ratios, dust2, dust_index, dust1_fraction, lnfagn, lnagn_tau, gas_logz, z = tf.split(theta, (1, 1, 6, 1, 1, 1, 1, 1, 1, 1), axis=-1)
        
        # convert normalization and redshift to logmass
        log10M = (N - distance_modulus(tf.math.maximum(1e-5, z)))/-2.5

        # compute prior log density...
        
        # initialise log target density to baseline prior (unit normal prior on unconstrained parameters: NB not applied to normalization parameter N which is not bijected)
        logp = tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[...,1:]), axis=-1, keepdims=True)
        
        # logmass prior
        logp = logp + mass_function_log_prob(log10M, z) + self.massLimitsPrior.log_prob(log10M)

        # metallicity prior
        logp = logp + metallicity_mass_log_prob(log10Z, log10M)

        # logsfr ratio priors (student's-t with 2 d.o.f)
        logp = logp + tf.reduce_sum(tf.multiply(-1.5, tf.math.log(tf.add(1., tf.multiply(0.5, tf.square(tf.divide(logsfr_ratios, 0.3)))))), axis=-1)

        # dust2 prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust2, 0.3), 1.0)))

        # dust index prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust_index, -0.095 + 0.111*dust2 - 0.0066*tf.square(dust2)), 0.4)))

        # dust1 fraction prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust1_fraction, 1.), 0.3)))

        # log fagn prior
        ### uniform only ###

        # log agn tau prior
        ### uniform only ###

        # gas logZ prior
        ### uniform only ###

        # z prior
        ### uniform only ###

        return tf.squeeze(logp, axis=-1)
        

class ModelHMIBaselinePrior:
    
    def __init__(self):
        
        # parameters and limits
        self.n_sps_parameters = 6
        self.parameter_names = ['N', 'log10Z', 'dust2', 'lntau', 'tmax', 'z']
        self.lower = tf.constant([7., -1.98, 0., np.log(0.316), 7.3e-5, 1e-5], dtype=tf.float32)
        self.upper = tf.constant([13., 0.19, 4., np.log(100.), 1.0, 2.5], dtype=tf.float32)

        # bijector from constrained parameter (physical) to unconstrained space for sampling. 
        # note: no bijector for the normalization parameter N (since it is already unconstrained)
        self.bijector = tfb.Blockwise([tfb.Identity()] + [tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(self.upper[_]-self.lower[_])), tfb.Shift(-self.lower[_])])) for _ in range(1, self.n_sps_parameters)])

        # baseline prior on unconstrained latent parameters
        # note this does not apply to the normalization parameter N
        self.baselinePrior = tfd.Normal(loc=0., scale=1.)

        # mass limits prior
        self.massLimitsPrior = tfd.Uniform(low=7., high=13.)
        
    @tf.function
    def log_prob(self, latentparameters):
        
        # convert parameters...
        
        # biject unconstrained latent parameters to physical parameters
        theta = self.bijector(latentparameters)

        # split up the parameters to make things easier to read
        N, log10Z, dust2, lntau, tmax, z = tf.split(theta, (1, 1, 1, 1, 1, 1), axis=-1)
        
        # convert normalization and redshift to logmass
        log10M = (N - distance_modulus(tf.math.maximum(1e-5, z)))/-2.5

        # compute prior log density...
        
        # initialise log target density to baseline prior (unit normal prior on unconstrained parameters: NB not applied to normalization parameter N which is not bijected)
        logp = tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[...,1:]), axis=-1, keepdims=True)
        
        # logmass prior
        logp = logp + mass_function_log_prob(log10M, z) + self.massLimitsPrior.log_prob(log10M)

        # metallicity prior
        logp = logp + metallicity_mass_log_prob(log10Z, log10M)

        # dust2 prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust2, 0.3), 1.0)))

        # log tau prior
        ### uniform only ###

        # tmax prior
        ### uniform only ###

        # z prior
        ### uniform only ###

        return tf.squeeze(logp, axis=-1)


class ModelHMIIBaselinePrior:
    
    def __init__(self):
        
        # parameters and limits
        self.n_sps_parameters = 7
        self.parameter_names = ['N', 'log10Z', 'dust2', 'lntau', 'tmax', 'dust_index', 'z']
        self.lower = tf.constant([7., -1.98, 0., np.log(0.316), 7.3e-5, -1., 1e-5], dtype=tf.float32)
        self.upper = tf.constant([13., 0.19, 4., np.log(100.), 1.0, 0.4, 2.5], dtype=tf.float32)

        # bijector from constrained parameter (physical) to unconstrained space for sampling. 
        # note: no bijector for the normalization parameter N (since it is already unconstrained)
        self.bijector = tfb.Blockwise([tfb.Identity()] + [tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(self.upper[_]-self.lower[_])), tfb.Shift(-self.lower[_])])) for _ in range(1, self.n_sps_parameters)])

        # baseline prior on unconstrained latent parameters
        # note this does not apply to the normalization parameter N
        self.baselinePrior = tfd.Normal(loc=0., scale=1.)

        # mass limits prior
        self.massLimitsPrior = tfd.Uniform(low=7., high=13.)
        
    @tf.function
    def log_prob(self, latentparameters):
        
        # convert parameters...
        
        # biject unconstrained latent parameters to physical parameters
        theta = self.bijector(latentparameters)

        # split up the parameters to make things easier to read
        N, log10Z, dust2, lntau, tmax, dust_index, z = tf.split(theta, (1, 1, 1, 1, 1, 1, 1), axis=-1)
        
        # convert normalization and redshift to logmass
        log10M = (N - distance_modulus(tf.math.maximum(1e-5, z)))/-2.5

        # compute prior log density...
        
        # initialise log target density to baseline prior (unit normal prior on unconstrained parameters: NB not applied to normalization parameter N which is not bijected)
        logp = tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[...,1:]), axis=-1, keepdims=True)
        
        # logmass prior
        logp = logp + mass_function_log_prob(log10M, z) + self.massLimitsPrior.log_prob(log10M)

        # metallicity prior
        logp = logp + metallicity_mass_log_prob(log10Z, log10M)

        # dust2 prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust2, 0.3), 1.0)))

        # dust index prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust_index, -0.095 + 0.111*dust2 - 0.0066*tf.square(dust2)), 0.4)))

        # log tau prior
        ### uniform only ###

        # tmax prior
        ### uniform only ###

        # z prior
        ### uniform only ###

        return tf.squeeze(logp, axis=-1)


class ModelABBaselinePrior:
    
    def __init__(self, SFHPrior=None):
        
        # parameters and limits
        self.n_sps_parameters = 9
        self.parameter_names = ['N', 'log10Z', 'dust2', 'dust1_fraction', 'dust_index', 'log10alpha', 'log10beta', 'tau', 'z']
        self.lower = tf.constant([7., -1.98, 0., 0., -1., -1., -1., 0.1, 1e-5], dtype=tf.float32)
        self.upper = tf.constant([13., 0.19, 4., 2., 0.4, 3., 3., 13.78, 2.5], dtype=tf.float32)

        # bijector from constrained parameter (physical) to unconstrained space for sampling. 
        # note: no bijector for the normalization parameter N (since it is already unconstrained)
        self.bijector = tfb.Blockwise([tfb.Identity()] + [tfb.Invert(tfb.Chain([tfb.Invert(tfb.NormalCDF()), tfb.Scale(1./(self.upper[_]-self.lower[_])), tfb.Shift(-self.lower[_])])) for _ in range(1, self.n_sps_parameters)])

        # baseline prior on unconstrained latent parameters
        # note this does not apply to the normalization parameter N
        self.baselinePrior = tfd.Normal(loc=0., scale=1.)

        # mass limits prior
        self.massLimitsPrior = tfd.Uniform(low=7., high=13.)

        # star formation history parameters prior: import conditional density estimator model for P(SFH | z)
        self.SFHPrior = SFHPrior
        
    @tf.function
    def log_prob(self, latentparameters):
        
        # convert parameters...
        
        # biject unconstrained latent parameters to physical parameters
        theta = self.bijector(latentparameters)

        # split up the parameters to make things easier to read
        N, log10Z, dust2, dust1_fraction, dust_index, sfh, z = tf.split(theta, (1, 1, 1, 1, 1, 3, 1), axis=-1)

        # latent version of SFH parameters
        latent_sfh = latentparameters[..., 5:8]
        
        # convert normalization and redshift to logmass
        log10M = (N - distance_modulus(tf.math.maximum(1e-5, z)))/-2.5

        # compute prior log density...
        
        # initialise log target density to baseline prior (unit normal prior on unconstrained parameters: NB not applied to normalization parameter N which is not bijected)
        logp = tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[...,1:5]), axis=-1, keepdims=True) + tf.reduce_sum(self.baselinePrior.log_prob(latentparameters[...,8:]), axis=-1, keepdims=True)
        
        # logmass prior
        logp = logp + mass_function_log_prob(log10M, z) + self.massLimitsPrior.log_prob(log10M)

        # metallicity prior
        logp = logp + metallicity_mass_log_prob(log10Z, log10M)

        # dust2 prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust2, 0.3), 1.0)))

        # dust index prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust_index, -0.095 + 0.111*dust2 - 0.0066*tf.square(dust2)), 0.4)))

        # dust1 fraction prior
        logp = logp - tf.multiply(0.5, tf.square(tf.divide(tf.subtract(dust1_fraction, 1.), 0.3)))

        # squeeze
        logp = tf.squeeze(logp, axis=-1)

        # SFH prior
        if self.SFHPrior is not None:
            if len(latent_sfh.shape) > 2:
                dims = latent_sfh.shape[:-1]
                size = tf.math.reduce_prod(dims)
                logp = logp + tf.reshape(self.SFHPrior.log_prob(tf.reshape(latent_sfh, [size, 3]), tf.reshape(z, [size, 1]) ), dims)
            else:
                logp = logp + self.SFHPrior.log_prob(latent_sfh, z)

        return logp
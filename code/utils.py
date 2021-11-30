import numpy as np
import tensorflow as tf

# natural log of 10 and ln(2pi)/2
ln10_ = tf.constant(np.log(10), dtype=tf.float32)
halfln2pi_ = tf.constant(0.5*np.log(2*np.pi), dtype=tf.float32)

# set up distance modulus (fitting function) parameters
Om_ = tf.constant(0.286, dtype=tf.float32) # omega matter
H0_ = tf.constant(69.32, dtype=tf.float32) # Hubble constant
c_ = tf.constant(299792.458, dtype=tf.float32) # speed of light
A0_ = tf.constant(c_/H0_, dtype=tf.float32)
s_ = tf.constant(((1-Om_)/Om_)**(1./3.), dtype=tf.float32)
B0_ = tf.constant(2*np.sqrt((1-Om_)/Om_ +1), dtype=tf.float32)
B1_ = tf.constant(-0.154*s_, dtype=tf.float32)
B2_ = tf.constant(0.4304*s_**2, dtype=tf.float32)
B3_ = tf.constant(0.19097*s_**3, dtype=tf.float32)
B4_ = tf.constant(0.066941*s_**4, dtype=tf.float32)
eta0_ = tf.constant(B0_*(1 + B1_ + B2_ + B3_ + B4_)**(-0.125), dtype=tf.float32)

# distance modulus fitting function
@tf.function
def distance_modulus(z):

  return 5*tf.math.log(1e6*A0_*(1+z)*(eta0_ - (B0_*((1+z)**4 + B1_*(1+z)**3 + B2_*(1+z)**2 + B3_*(1+z) + B4_)**(-0.125) )))/ln10_ - 5

  # distance modulus fitting function
@tf.function
def dVdz(z):

    return A0_ * z**2 * (eta0_ + (0.125 * B0_* ((1+z)**4 + B1_*(1+z)**3 + B2_*(1+z)**2 + B3_*(1+z) + B4_)**(-1.125) *  (4*(1+z)**3 + 3*B1_*(1+z)**2 + 2*B2_*(1+z) + B3_) ))
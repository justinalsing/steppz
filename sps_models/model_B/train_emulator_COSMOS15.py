import sys
import numpy as np
import tensorflow as tf
from speculator import *

# import training data

# root directory
root_dir = sys.argv[1]

# import the mags and thetas
training_theta = np.row_stack([np.load(root_dir + 'training_data/parameters/parameters{}.npy'.format(_))[:,1:] for _ in range(64)]).astype(np.float32)
training_mag = np.row_stack([np.load(root_dir + 'training_data/photometry/photometry{}.npy'.format(_))[:,9:] for _ in range(64)]).astype(np.float32)

# re-parameterization
training_theta[:,1] = np.sqrt(training_theta[:,1]) # dust2 -> sqrt(dust2)

# parameters shift and scale
parameters_shift = np.mean(training_theta, axis=0)
parameters_scale = np.std(training_theta, axis=0)
magnitudes_shift = np.mean(training_mag, axis=0)
magnitudes_scale = np.std(training_mag, axis=0)

# convert training data to tensors
training_theta = tf.convert_to_tensor(training_theta)
training_mag = tf.convert_to_tensor(training_mag)

# filter names
filters = ['ip_cosmos', 'v_cosmos', 'uvista_y_cosmos', 'r_cosmos', 'hsc_y',
       'zpp', 'b_cosmos', 'uvista_h_cosmos', 'wircam_H', 'ia484_cosmos',
       'ia527_cosmos', 'ia624_cosmos', 'ia679_cosmos', 'ia738_cosmos',
       'ia767_cosmos', 'ia427_cosmos', 'ia464_cosmos', 'ia505_cosmos',
       'ia574_cosmos', 'ia709_cosmos', 'ia827_cosmos', 'uvista_j_cosmos',
       'uvista_ks_cosmos', 'wircam_Ks', 'NB711.SuprimeCam',
       'NB816.SuprimeCam']

# training set up
validation_split = 0.1
lr = [1e-3, 1e-4, 1e-5, 1e-6]
batch_size = [1000, 10000, 50000, int((1-validation_split) * training_theta.shape[0])]
gradient_accumulation_steps = [1, 1, 1, 10]
epochs = 1000

# early stopping set up
patience = 20

# architecture
n_layers = 4
n_units = 128

# do the training
train_photulator_stack(training_theta, 
                       training_mag, 
                       parameters_shift, 
                       parameters_scale, 
                       magnitudes_shift, 
                       magnitudes_scale,
                       filters=filters,
                       n_layers=n_layers,
                       n_units=n_units,
                       validation_split=validation_split,
                       lr=lr,
                       batch_size=batch_size,
                       gradient_accumulation_steps=gradient_accumulation_steps,
                       epochs=epochs,
                       patience=patience,
                       verbose=True, 
                       root_dir=root_dir + 'trained_models/')

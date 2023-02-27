import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from speculator import *

# import training data

# root directory
# root_dir = sys.argv[1]
root_dir = '/home/sinandeger/PycharmProjects/tfenv_cosmicexplorer/steppz_sfh_testing/'

# import the mags and thetas
training_theta_init = np.row_stack([np.load(root_dir + 'training_data/parameters/parameters{}.npy'.format(_))[:, 1:] for _ in range(1)]).astype(np.float32)

n_time_bins = 7
parameter_names = ['gaslog10Z'] + ['logsfr_ratio{}'.format(i) for i in range(1, n_time_bins)] + ['dust2', 'dust_index', 'dust1_fraction', 'z'] # N is discarded during the definition of training_theta
"""Populate a pandas dataframe with the parameters array & parameter names"""
parameter_df = pd.DataFrame(data=training_theta_init, columns=parameter_names)
# re-parameterization
parameter_df['dust2'] = np.sqrt(parameter_df['dust2']) # dust2 -> sqrt(dust2)

training_mag = np.row_stack([np.load(root_dir + 'photometry/COSMOS15_photometry{}.npy'.format(_)) for _ in range(1)]).astype(np.float32)

training_theta = parameter_df.to_numpy()

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

# filter range to run per stack
# first = sys.argv[2]
# last = sys.argv[3]

first = 24
last = 26

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

print(training_mag[:, first:last])

train_photulator_stack(training_theta,
                       training_mag[:, first:last],
                       parameters_shift,
                       parameters_scale,
                       magnitudes_shift[first:last],
                       magnitudes_scale[first:last],
                       filters=filters[first:last],
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
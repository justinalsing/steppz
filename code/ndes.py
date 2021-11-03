import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import pickle
tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

class NeuralSplineFlow1D(tf.Module):
    
    def __init__(self, n_spline_bins=32, n_conditional=1, n_hidden=[10, 10], activation=tf.tanh, base_loc=0., base_scale=0.25, spline_min=-1., spline_range=2.):
        
        # spline bins
        self._n_spline_bins = n_spline_bins
        
        # conditional dimensions
        self._n_conditional = n_conditional
        
        # hidden units and activation function
        self._architecture = [n_conditional] + n_hidden
        self._activation = activation
        
        # loc and scale for the (normal) base density, spline set-up
        self._base_loc = base_loc
        self._base_scale = base_scale
        self._spline_min = spline_min
        self._spline_range = spline_range
        
        # networks parameterizing bin widths, heights and knot slopes
        self._bin_widths = tfk.Sequential([tfkl.Dense(self._architecture[i+1], activation=self._activation) for i in range(len(n_hidden))] + [tfkl.Dense(self._n_spline_bins)] + [tfkl.Lambda(lambda x: tf.math.softmax(x, axis=-1) * (self._spline_range - self._n_spline_bins * 1e-2) + 1e-2)])
        self._bin_heights = tfk.Sequential([tfkl.Dense(self._architecture[i+1], activation=self._activation) for i in range(len(n_hidden))] + [tfkl.Dense(self._n_spline_bins)] + [tfkl.Lambda(lambda x: tf.math.softmax(x, axis=-1) * (self._spline_range - self._n_spline_bins * 1e-2) + 1e-2)])
        self._knot_slopes = tfk.Sequential([tfkl.Dense(self._architecture[i+1], activation=self._activation) for i in range(len(n_hidden))] + [tfkl.Dense(self._n_spline_bins - 1)] + [tfkl.Lambda(lambda x: tf.math.softplus(x) + 1e-2)])

    # construct spline bijector given inputs x
    def spline(self, x):
   
        return tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(x),
            bin_heights=self._bin_heights(x),
            knot_slopes=self._knot_slopes(x),
            range_min=self._spline_min)

    # construct transformed distribution given conditional inputs x
    def __call__(self, x):
        
        return tfd.TransformedDistribution(tfd.Normal(loc=self._base_loc, scale=self._base_scale), bijector=self.spline(x))
    
    # log probability for inputs y and conditionals x, ie., P(y | x)
    #@tf.function
    def log_prob(self, y, x):
        
        # construct spline
        rqspline = tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(x),
            bin_heights=self._bin_heights(x),
            knot_slopes=self._knot_slopes(x),
            range_min=self._spline_min)
        
        # construct distribution
        distribution = tfd.TransformedDistribution(tfd.Normal(loc=self._base_loc, scale=self._base_scale), bijector=rqspline)
        
        return distribution.log_prob(y)


class MixtureDensityNetwork(tfd.Distribution):

	def __init__(self, n_dimensions=None, n_conditionals=None, n_components=3, conditional_shift=None, conditional_scale=None, output_shift=None, output_scale=None, n_hidden=[50,50], activation=tf.keras.layers.LeakyReLU(0.01), optimizer=tf.keras.optimizers.Adam(), dtype=tf.float32, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), restore=False, restore_filename=None):

		super(MixtureDensityNetwork, self).__init__(dtype=tf.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True)

		# load parameters if restoring saved model
		if restore is True:
			n_dimensions, n_conditionals, n_components, conditional_shift, conditional_scale, output_shift, output_scale, n_hidden, activation, kernel_initializer, loaded_trainable_variables = pickle.load(open(restore_filename, 'rb'))

		# dimension of data and parameter spaces
		self.n_dimensions = n_dimensions
		self.n_conditionals = n_conditionals

		# number of mixture components and network architecture
		self.n_components = n_components

		# required size of output layer for a Gaussian mixture density network
		self.n_hidden = n_hidden
		self.activation = activation
		self.architecture = [self.n_conditionals] + self.n_hidden
		self.kernel_initializer = kernel_initializer

		# shift and scale
		self.conditional_shift = tf.convert_to_tensor(conditional_shift, dtype=tf.float32) if conditional_shift is not None else tf.zeros(self.n_conditionals, dtype=tf.float32)
		self.conditional_scale = tf.convert_to_tensor(conditional_scale, dtype=tf.float32) if conditional_scale is not None else tf.ones(self.n_conditionals, dtype=tf.float32)
		self.output_shift = tf.convert_to_tensor(output_shift, dtype=tf.float32) if output_shift is not None else tf.zeros(self.n_dimensions, dtype=tf.float32)
		self.output_scale = tf.convert_to_tensor(output_scale, dtype=tf.float32) if output_scale is not None else tf.ones(self.n_dimensions, dtype=tf.float32)

		# construce network model
		self._network = self.build_network(kernel_initializer)

		# optimizer
		self.optimizer = optimizer

		# load in the saved weights if restoring
		if restore is True:
			for model_variable, loaded_variable in zip(self._network.trainable_variables, loaded_trainable_variables):
				model_variable.assign(loaded_variable)

	def build_network(self, kernel_initializer):

		model = tf.keras.models.Sequential([tf.keras.layers.Dense(self.architecture[layer + 1],
																  input_shape=(size,),
																  activation=self.activation,
																  kernel_initializer=kernel_initializer)
																  for layer, size in enumerate(self.architecture[:-1])])

		# output layer (mixture component parts)
		model.add(tf.keras.layers.Dense(tfp.layers.MixtureSameFamily.params_size(self.n_components, component_params_size=tfp.layers.MultivariateNormalTriL.params_size(self.n_dimensions)), kernel_initializer=kernel_initializer))

		# transform output layer into distribution
		model.add(tfp.layers.MixtureSameFamily(self.n_components, tfp.layers.MultivariateNormalTriL(self.n_dimensions)))

		return model

	def log_prob(self, x, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).log_prob((x - self.output_shift)/self.output_scale )

	def prob(self, x, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).prob((x - self.output_shift)/self.output_scale)

	def sample(self, n, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).sample(n) * self.output_scale + self.output_shift

	def save(self, filename):

		pickle.dump([self.n_dimensions, self.n_conditionals, self.n_components, self.conditional_shift, self.conditional_scale, self.output_shift, self.output_scale, self.n_hidden, self.activation, self.kernel_initializer] + [tuple(variable.numpy() for variable in self._network.trainable_variables)], open(filename, 'wb'))

	#@tf.function
	def loss(self, x, conditional=None):

		return -tf.reduce_mean(self.log_prob(x, conditional=conditional))

	#@tf.function
	def training_step(self, x, conditional=None):

		with tf.GradientTape() as tape:

			loss = self.loss(x, conditional=conditional)

		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return loss

	def fit(self, training_data, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):

		# training data
		training_conditionals, training_outputs = training_data

		# split into validation and training sub-sets
		n_validation = int(training_conditionals.shape[0] * validation_split)
		n_training = training_conditionals.shape[0] - n_validation
		training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

		# create iterable dataset (given batch size)
		training_data = tf.data.Dataset.from_tensor_slices((training_conditionals[training_selection], training_outputs[training_selection])).shuffle(n_training).batch(batch_size)

		# set up training loss
		training_loss = [np.infty]
		validation_loss = [np.infty]
		best_loss = np.infty
		early_stopping_counter = 0

		with trange(epochs) as t:
			for epoch in t:

				# loop over batches for a single epoch
				for conditionals, outputs in training_data:

					loss = self.training_step(outputs, conditional=conditionals)
					t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# compute total loss and validation loss
				validation_loss.append(self.loss(training_outputs[~training_selection], conditional=training_conditionals[~training_selection]).numpy())
				training_loss.append(loss.numpy())

				# update progress bar
				t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# early stopping condition
				if validation_loss[-1] < best_loss:
					best_loss = validation_loss[-1]
					early_stopping_counter = 0
					if save:
						self.save(filename)
				else:
					early_stopping_counter += 1
				if early_stopping_counter >= patience:
					break
		return training_loss, validation_loss


class MixtureDensityNetworkDiag(tfd.Distribution):

	def __init__(self, n_dimensions=None, n_conditionals=None, n_components=3, conditional_shift=None, conditional_scale=None, output_shift=None, output_scale=None, n_hidden=[50,50], activation=tf.keras.layers.LeakyReLU(0.01), optimizer=tf.keras.optimizers.Adam(), dtype=tf.float32, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), restore=False, restore_filename=None):

		super(MixtureDensityNetworkDiag, self).__init__(dtype=tf.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True)

		# load parameters if restoring saved model
		if restore is True:
			n_dimensions, n_conditionals, n_components, conditional_shift, conditional_scale, output_shift, output_scale, n_hidden, activation, kernel_initializer, loaded_trainable_variables = pickle.load(open(restore_filename, 'rb'))

		# dimension of data and parameter spaces
		self.n_dimensions = n_dimensions
		self.n_conditionals = n_conditionals

		# number of mixture components and network architecture
		self.n_components = n_components

		# required size of output layer for a Gaussian mixture density network
		self.n_hidden = n_hidden
		self.activation = activation
		self.architecture = [self.n_conditionals] + self.n_hidden
		self.kernel_initializer = kernel_initializer

		# shift and scale
		self.conditional_shift = tf.convert_to_tensor(conditional_shift, dtype=tf.float32) if conditional_shift is not None else tf.zeros(self.n_conditionals, dtype=tf.float32)
		self.conditional_scale = tf.convert_to_tensor(conditional_scale, dtype=tf.float32) if conditional_scale is not None else tf.ones(self.n_conditionals, dtype=tf.float32)
		self.output_shift = tf.convert_to_tensor(output_shift, dtype=tf.float32) if output_shift is not None else tf.zeros(self.n_dimensions, dtype=tf.float32)
		self.output_scale = tf.convert_to_tensor(output_scale, dtype=tf.float32) if output_scale is not None else tf.ones(self.n_dimensions, dtype=tf.float32)

		# construce network model
		self._network = self.build_network(kernel_initializer)

		# optimizer
		self.optimizer = optimizer

		# load in the saved weights if restoring
		if restore is True:
			for model_variable, loaded_variable in zip(self._network.trainable_variables, loaded_trainable_variables):
				model_variable.assign(loaded_variable)

	def build_network(self, kernel_initializer):

		model = tf.keras.models.Sequential([tf.keras.layers.Dense(self.architecture[layer + 1],
																  input_shape=(size,),
																  activation=self.activation,
																  kernel_initializer=kernel_initializer)
																  for layer, size in enumerate(self.architecture[:-1])])

		# output layer (mixture component parts)
		model.add(tf.keras.layers.Dense(tfp.layers.MixtureSameFamily.params_size(self.n_components, component_params_size=tfp.layers.IndependentNormal.params_size(self.n_dimensions)), kernel_initializer=kernel_initializer))

		# transform output layer into distribution
		model.add(tfp.layers.MixtureSameFamily(self.n_components, tfp.layers.IndependentNormal(self.n_dimensions)))

		return model

	def log_prob(self, x, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).log_prob((x - self.output_shift)/self.output_scale )

	def prob(self, x, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).prob((x - self.output_shift)/self.output_scale)

	def sample(self, n, conditional=None):

		return self._network((conditional - self.conditional_shift)/self.conditional_scale).sample(n) * self.output_scale + self.output_shift

	def save(self, filename):

		pickle.dump([self.n_dimensions, self.n_conditionals, self.n_components, self.conditional_shift, self.conditional_scale, self.output_shift, self.output_scale, self.n_hidden, self.activation, self.kernel_initializer] + [tuple(variable.numpy() for variable in self._network.trainable_variables)], open(filename, 'wb'))

	#@tf.function
	def loss(self, x, conditional=None):

		return -tf.reduce_mean(self.log_prob(x, conditional=conditional))

	#@tf.function
	def training_step(self, x, conditional=None):

		with tf.GradientTape() as tape:

			loss = self.loss(x, conditional=conditional)

		gradients = tape.gradient(loss, self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return loss

	def fit(self, training_data, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):

		# training data
		training_conditionals, training_outputs = training_data

		# split into validation and training sub-sets
		n_validation = int(training_conditionals.shape[0] * validation_split)
		n_training = training_conditionals.shape[0] - n_validation
		training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

		# create iterable dataset (given batch size)
		training_data = tf.data.Dataset.from_tensor_slices((training_conditionals[training_selection], training_outputs[training_selection])).shuffle(n_training).batch(batch_size)

		# set up training loss
		training_loss = [np.infty]
		validation_loss = [np.infty]
		best_loss = np.infty
		early_stopping_counter = 0

		with trange(epochs) as t:
			for epoch in t:

				# loop over batches for a single epoch
				for conditionals, outputs in training_data:

					loss = self.training_step(outputs, conditional=conditionals)
					t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# compute total loss and validation loss
				validation_loss.append(self.loss(training_outputs[~training_selection], conditional=training_conditionals[~training_selection]).numpy())
				training_loss.append(loss.numpy())

				# update progress bar
				t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

				# early stopping condition
				if validation_loss[-1] < best_loss:
					best_loss = validation_loss[-1]
					early_stopping_counter = 0
					if save:
						self.save(filename)
				else:
					early_stopping_counter += 1
				if early_stopping_counter >= patience:
					break
		return training_loss, validation_loss



class AutoregressiveNeuralSplineFlow(tf.Module):
    
    def __init__(self, n_bins=32, n_dimensions=3, n_conditional=3, n_hidden=[10, 10], activation=tf.tanh, base_loc=0., base_scale=0.25, restore=False, restore_filename=None):
        
        # set up variables...

        # load parameters if restoring saved model
        if restore:
          n_bins, n_dimensions, n_conditional, base_loc, base_scale, n_hidden, activation, loaded_trainable_variables = pickle.load(open(restore_filename, 'rb'))

        # spline bins
        self._n_bins = n_bins
        
        # density and conditional dimensions
        self._n_dimensions = n_dimensions
        self._n_conditional = n_conditional
        
        # hidden units and activation function
        self._n_hidden = n_hidden
        self._activation = activation
        
        # loc and scale for the (normal) base density
        self._base_loc = base_loc
        self._base_scale = base_scale

        # construct the model...

        # conditional autoregressive network parameterizing the bin widths
        self._bin_widths_ = tfb.AutoregressiveNetwork(params=self._n_bins, 
                                                     event_shape=self._n_dimensions, 
                                                     conditional=True, 
                                                     conditional_event_shape=self._n_conditional,
                                                     hidden_units=self._n_hidden,
                                                     activation=self._activation)

        # conditional autoregressive network parameterizing the bin heights
        self._bin_heights_ = tfb.AutoregressiveNetwork(params=self._n_bins, 
                                                     event_shape=self._n_dimensions, 
                                                     conditional=True, 
                                                     conditional_event_shape=self._n_conditional,
                                                     hidden_units=self._n_hidden,
                                                     activation=self._activation)
        
        # conditional autoregressive network parameterizing the slopes
        self._knot_slopes_ = tfb.AutoregressiveNetwork(params=self._n_bins-1, 
                                                     event_shape=self._n_dimensions, 
                                                     conditional=True, 
                                                     conditional_event_shape=self._n_conditional,
                                                     hidden_units=self._n_hidden,
                                                     activation=self._activation)
        
        # call to initialize trainable variables
        _ = self.__call__(tf.zeros((1, self._n_dimensions)), tf.zeros((1, self._n_conditional)))
        
        if restore:
			      for model_variable, loaded_variable in zip(self.trainable_variables, loaded_trainable_variables):
				        model_variable.assign(loaded_variable)

    # softmax the bin widths
    def bin_widths(self, x, y):
        
        return tf.math.softmax(self._bin_widths_(x, conditional_input=y), axis=-1) * (2 - self._n_bins * 1e-2) + 1e-2
    
    # softmax the bin heights
    def bin_heights(self, x, y):
        
        return tf.math.softmax(self._bin_heights_(x, conditional_input=y), axis=-1) * (2 - self._n_bins * 1e-2) + 1e-2

    # softplus the knot slopes
    def knot_slopes(self, x, y):
        
        return tf.math.softplus(self._knot_slopes_(x, conditional_input=y)) + 1e-2

    # construct spline bijector given inputs x and conditional inputs y
    def spline(self, x, y):
   
        return tfb.RationalQuadraticSpline(
            bin_widths=self.bin_widths(x, y),
            bin_heights=self.bin_heights(x, y),
            knot_slopes=self.knot_slopes(x, y))

    # construct transformed distribution given inputs x and conditional inputs y
    def __call__(self, x, y):
        
        return tfd.TransformedDistribution(tfd.Normal(loc=self._base_loc, scale=self._base_scale), bijector=self.spline(x, y))
    
    @tf.function
    def log_prob(self, x, y):
        
        distribution_ = tfd.TransformedDistribution(tfd.Normal(loc=self._base_loc, scale=self._base_scale), bijector=tfb.RationalQuadraticSpline(bin_widths=self.bin_widths(x, y), bin_heights=self.bin_heights(x, y), knot_slopes=self.knot_slopes(x, y)))
        
        return tf.math.reduce_sum(distribution_.log_prob(x), axis=-1)

    # save and restore
    def save(self, filename):

		    pickle.dump([self._n_bins, self._n_dimensions, self._n_conditional, self._base_loc, self._base_scale, self._n_hidden, self._activation] + [tuple(variable.numpy() for variable in self.trainable_variables)], open(filename, 'wb'))

class BinomialNetwork(tf.Module):
    
    def __init__(self, n_inputs=18, n_hidden=[10, 10], activation=tf.tanh, optimizer=tf.keras.optimizers.Adam(lr=1e-3), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5, seed=None), restore=False, restore_filename=None):
        
        # set up variables...

        # load parameters if restoring saved model
        if restore:
            n_inputs, n_hidden, activation, loaded_trainable_variables = pickle.load(open(restore_filename, 'rb'))

        # architecture
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.architecture = [n_inputs] + n_hidden + [1]
        self.n_layers = len(self.architecture) - 1
        self.activation = activation
        self.activations = [activation for _ in range(self.n_layers-1)] + [tf.nn.sigmoid]
        self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        
        # model
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(self.architecture[layer+1],
                                                                  input_shape=(self.architecture[layer],),
                                                                  activation=self.activations[layer],
                                                                  kernel_initializer=self.kernel_initializer) for layer in range(self.n_layers)])


        # call to initialize trainable variables
        _ = self.__call__(tf.zeros((1, self.n_inputs)))
        
        # restore trainable variables if needed
        if restore:
            for model_variable, loaded_variable in zip(self.trainable_variables, loaded_trainable_variables):
                model_variable.assign(loaded_variable)

    # call model (log_prob)
    @tf.function
    def __call__(self, x):
        
        return self.model(x)

    # loss
    @tf.function
    def loss(self, inputs, selected):
        
        p = tf.squeeze(self.__call__(inputs), -1)
        return - tf.reduce_mean(selected * tf.math.log(p) + (1 - selected) * tf.math.log(1-p))
    
    # training step
    @tf.function
    def training_step(self, inputs, selected):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, selected)
            gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # fit
    def fit(self, data, validation_split=0.1, epochs=1000, batch_size=128, patience=20, progress_bar=True, save=False, filename=None):

        # training data
        inputs, selected = data

        # split into validation and training sub-sets
        n_validation = int(inputs.shape[0] * validation_split)
        n_training = inputs.shape[0] - n_validation
        training = tf.random.shuffle([True] * n_training + [False] * n_validation)
        training_inputs = inputs[training]
        training_selected = selected[training]
        validation_inputs = inputs[~training]
        validation_selected = selected[~training]

        # create iterable dataset (given batch size)
        training_data = tf.data.Dataset.from_tensor_slices((training_inputs, training_selected)).shuffle(n_training).batch(batch_size)

        # set up training loss
        training_loss = [np.infty]
        validation_loss = [np.infty]
        best_loss = np.infty
        early_stopping_counter = 0

        with trange(epochs) as t:
            for epoch in t:

                # loop over batches for a single epoch
                for inputs_, selected_ in training_data:

                    loss = self.training_step(inputs_, selected_)
                    t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                # compute total loss and validation loss
                validation_loss.append(self.loss(validation_inputs, validation_selected).numpy())
                training_loss.append(loss.numpy())

                # update progress bar
                t.set_postfix(ordered_dict={'training loss':loss.numpy(), 'validation_loss':validation_loss[-1]}, refresh=True)

                # early stopping condition
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    early_stopping_counter = 0
                    if save:
                        self.save(filename)
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    break
        return training_loss, validation_loss
    
    # save and restore
    def save(self, filename):
        pickle.dump([self.n_inputs, self.n_hidden, self.activation] + [tuple(variable.numpy() for variable in self.trainable_variables)], open(filename, 'wb'))
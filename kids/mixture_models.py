import tensorflow as tf
import numpy as np
import pickle

class StudentNormalMixture(tf.keras.Model):
    
    def __init__(self, n_hidden=64, df_student=2., optimizer=tf.keras.optimizers.Adam(lr=0.01)):
        
        super(StudentNormalMixture, self).__init__()
        
        self.n_hidden = n_hidden # n hidden units for network
        self.df_student = df_student # df of the Student's-t part of the convolution
        self.n_params = 4 # number of parameters of Student's-t Normal mixture
        
        # weights, biases and activation parameters
        self.W1 = tf.Variable(tf.random.normal([1, self.n_hidden], 0, 1e-8), trainable=True)
        self.b1 = tf.Variable(tf.random.normal([self.n_hidden], 0, 1e-8), trainable=True)
        self.W2 = tf.Variable(tf.random.normal([self.n_hidden, self.n_params], 0, 1e-8), trainable=True)
        self.b2 = tf.Variable(tf.random.normal([self.n_params], 0, 1e-8), trainable=True)
        self.alphas = tf.Variable(tf.random.normal([self.n_hidden], 0, 1e-8), trainable=True)
        self.betas = tf.Variable(tf.random.normal([self.n_hidden], 0, 1e-8), trainable=True)
        
        # constants
        self.logtwopi = tf.constant(np.log(2.*np.pi), dtype=tf.float32)
        self.pi = tf.constant(np.pi, dtype=tf.float32)
        
        # optimizer
        self.optimizer = optimizer
        
    # custom activation function
    def activation(self, x, alpha, beta):
        
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)
    
    # pass through network to get parameters of mixture model for given logr
    @tf.function
    def call(self, logr):
        
        # neural network
        hidden = tf.matmul(logr, self.W1) + self.b1
        output = tf.matmul(self.activation(hidden, self.alphas, self.betas), self.W2) + self.b2
        
        # split the network output
        x1, x2, x3, x4 = tf.split(output, (1, 1, 1, 1), axis=-1)

        # transform outputs into mixture parameters (satisfying required constraints)
        alpha = tf.squeeze(1./(1. + tf.exp(-x1))) # component weight ([0, 1])
        scale_normal = tf.squeeze(1. + tf.exp(x2)) # scale of normal (>= 1.)
        scale_student = tf.squeeze((1. + tf.exp(x3))*tf.exp(logr)) # scale of the Student's-t (>= r)
        df = tf.squeeze(self.df_student + tf.exp(x4)) # df of the Student's-t (>= df of student in the convolution)
        
        return alpha, scale_normal, scale_student, df
    
    # log probability of mixture model for given logr
    @tf.function
    def log_prob(self, x, logr):
        
        # mixture model parameters for the given logr
        alpha, scale_normal, scale_student, df = self.call(logr)
        
        # log prob of the normal component (including component weight)
        log_prob_normal = -0.5*tf.square(x/scale_normal) - 0.5*self.logtwopi - tf.math.log(scale_normal) 

        # log prob student
        log_prob_student = -( (df + 1.)/2. )*tf.math.log(1. + (1./df)*tf.square(x/scale_student)) + tf.math.lgamma((df + 1.)/2.) - tf.math.lgamma(df/2.) - 0.5*tf.math.log(self.pi*df) - tf.math.log(scale_student)

        # stack the probs
        log_prob = tf.math.reduce_logsumexp(tf.stack([log_prob_normal + tf.math.log(alpha), log_prob_student + tf.math.log(1. - alpha)], axis=-1), axis=-1)

        return log_prob
    
    # MSE between model and target log prob
    @tf.function
    def loss(self, x, logr, target_log_prob):
        
        return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.log_prob(x, logr), target_log_prob)))
    
    # gradient update step
    @tf.function
    def training_step(self, x, logr, target_log_prob):
        
        with tf.GradientTape() as tape:
            loss = self.loss(x, logr, target_log_prob)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss
    
    # save model
    def save(self, filename):
        
        f = open(filename, 'wb')
        pickle.dump([variable.numpy() for variable in self.trainable_variables], f)
        f.close()
        
    # load model
    def load(self, filename):
        
        f = open(filename, 'rb')
        trained_variables = pickle.load(f)
        f.close()
        
        for variable, trained_variable in zip(self.trainable_variables, trained_variables):
            variable.assign(trained_variable)
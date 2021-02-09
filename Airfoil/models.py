import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange
import math

tf.autograph.set_verbosity(0)
################################################################################
#################################### Layers ####################################
################################################################################
class ConditionalBatchNormalization2D(keras.layers.Layer):
    def __init__(self,channels):
        super(ConditionalBatchNormalization2D, self).__init__()
        
        self.batch_norm = keras.layers.BatchNormalization(trainable=False)
        self.gamma_dense = keras.layers.Dense(channels)
        self.beta_dense = keras.layers.Dense(channels)
        
    def call(self, inputs, condition):
        
        x = self.batch_norm(inputs)
        
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)
        
        x = x + x*tf.expand_dims(tf.expand_dims(gamma,1),1) + tf.expand_dims(tf.expand_dims(beta,1),1)
        
        return x

class Residual_Block(keras.layers.Layer):
    def __init__(self,channels, kernel_size, kernel_initializer, kernel_regularizer, downsample=False):
        super(Residual_Block, self).__init__()
        self. downsample = downsample

        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU1 = keras.layers.ReLU()

        if self.downsample:
            self.Conv1 = keras.layers.Conv2D(channels, kernel_size, strides = (2,1), padding = 'same',
                                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
            self.Conv2 = keras.layers.Conv2D(channels, kernel_size = 1, strides = (2,1), padding = 'same',
                                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        else:
            self.Conv1 = keras.layers.Conv2D(channels, kernel_size, strides = 1, padding = 'same',
                                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU2 = keras.layers.ReLU()

        self.Conv3 = keras.layers.Conv2D(channels, kernel_size, strides = 1, padding = 'same',
                                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    def call(self, inputs, training = True):
        x_init = inputs

        x = self.batchnorm1(x_init, training=training)
        x = self.LReLU1(x)


        if self.downsample:
            x = self.Conv1(x)
            x_init = self.Conv2(x_init)

        else :
            x = self.Conv1(x)

        x = self.batchnorm2(x, training=training)
        x = self.LReLU2(x)
        x = self.Conv3(x)
        
        return x + x_init


################################################################################
######################## Replicated PaDGAN as Reference ########################
################################################################################
class Discriminator(keras.Model):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        self.latent_dim = latent_dim
        
        
        self.Conv1 = keras.layers.Conv2D(depth*1, kernel_size, strides=(2,1), padding='same')
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU1 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout1 = keras.layers.Dropout(dropout)

        self.Conv2 = keras.layers.Conv2D(depth*2, kernel_size, strides=(2,1), padding='same')
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU2 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout2 = keras.layers.Dropout(dropout)

        self.Conv3 = keras.layers.Conv2D(depth*4, kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU3 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout3 = keras.layers.Dropout(dropout)

        self.Conv4 = keras.layers.Conv2D(depth*8, kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU4 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout4 = keras.layers.Dropout(dropout)

        self.Conv5 = keras.layers.Conv2D(depth*16, kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU5 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout5 = keras.layers.Dropout(dropout)

        self.Conv6 = keras.layers.Conv2D(depth*32, kernel_size, strides=(2,1), padding='same')
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU6 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout6 = keras.layers.Dropout(dropout)

        self.flatten = keras.layers.Flatten()
        
        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(1)

        self.dense3 = keras.layers.Dense(128)

        self.LReLU8 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense3 = keras.layers.Dense(128)
        self.dense4 = keras.layers.Dense(latent_dim)
        self.dense5 = keras.layers.Dense(latent_dim)
        

    def call(self, inputs, training = True):
        
        x = inputs
        
        x = self.Conv1(x)
        x = self.batchnorm1(x, training=training)
        x = self.LReLU1(x)
        x = self.dropout1(x, training=training)
        
        x = self.Conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.LReLU2(x)
        x = self.dropout2(x, training=training)
        
        x = self.Conv3(x)
        x = self.batchnorm3(x, training=training)
        x = self.LReLU3(x)
        x = self.dropout3(x, training=training)
        
        x = self.Conv4(x)
        x = self.batchnorm4(x, training=training)
        x = self.LReLU4(x)
        x = self.dropout4(x, training=training)
        
        x = self.Conv5(x)
        x = self.batchnorm5(x, training=training)
        x = self.LReLU5(x)
        x = self.dropout5(x, training=training)
        
        x = self.Conv6(x)
        x = self.batchnorm6(x, training=training)
        x = self.LReLU6(x)
        x = self.dropout6(x, training=training)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.batchnorm7(x, training=False)
        x = self.LReLU7(x)
        
        d = self.dense2(x)
        
        q = self.dense3(x)

        q = self.LReLU8(q)
        q_mean = self.dense4(q)
        q_logstd = self.dense5(q)
        q_logstd = tf.math.maximum(q_logstd, -16)
        q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
        q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
        q = tf.concat([q_mean, q_logstd], axis=1)
        
        return d, q


class Generator(keras.Model):
    def __init__(self, bezier_degree, output_size):
        super(Generator, self).__init__()
        depth_cpw = 32*8
        dim_cpw = int((bezier_degree+1)/8)
        kernel_size = (4,3)
        self.dim_cpw = dim_cpw
        self.depth_cpw = depth_cpw
        self.bezier_degree = bezier_degree
        self.EPSILON = 1e-7
        
        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu1 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(dim_cpw*3*depth_cpw)
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu2 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT1 = keras.layers.Conv2DTranspose(int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu3 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT2 = keras.layers.Conv2DTranspose(int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu4 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT3 = keras.layers.Conv2DTranspose(int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu5 = keras.layers.LeakyReLU(alpha=0.2)

        self.Conv1 = keras.layers.Conv2D(1, (1,2), padding='valid')

        self.Conv2 = keras.layers.Conv2D(1, (1,3), padding='valid')
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

        self.dense3 = keras.layers.Dense(1024)
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu6 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense4 = keras.layers.Dense(256)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense5 = keras.layers.Dense(output_size - 1)
        self.softmax = keras.layers.Softmax()

    def call(self, c, z, training=True):
        
        if z == None:
                cz = c
        else:
            cz = tf.concat([c, z], axis=-1)
        
        cpw = self.dense1(cz)
        cpw = self.batchnorm1(cpw, training=training)
        cpw = self.LReLu1(cpw)

        cpw = self.dense2(cpw)
        cpw = self.batchnorm2(cpw, training=training)
        cpw = self.LReLu2(cpw)
        cpw = tf.reshape(cpw, (-1, self.dim_cpw, 3, self.depth_cpw))

        cpw = self.ConvT1(cpw)
        cpw = self.batchnorm3(cpw, training=training)
        cpw = self.LReLu3(cpw)
        
        cpw = self.ConvT2(cpw)
        cpw = self.batchnorm4(cpw, training=training)
        cpw = self.LReLu4(cpw)
        
        cpw = self.ConvT3(cpw)
        cpw = self.batchnorm5(cpw, training=training)
        cpw = self.LReLu5(cpw)
        
        # Control points
        cp = self.Conv1(cpw)
        cp = tf.math.tanh(cp)
        cp = tf.squeeze(cp, axis=-1)
        
        # Weights
        w = self.Conv2(cpw)
        w = self.sigmoid(w)
        w = tf.squeeze(w, axis=-1)
        
        # Parameters at data points
        db = self.dense3(cz)
        db = self.batchnorm6(db, training=training)
        db = self.LReLu6(db)
        
        db = self.dense4(db)
        db = self.batchnorm7(db, training=training)
        db = self.LReLu7(db)
        
        db = self.dense5(db)
        db = self.softmax(db)
        
        ub = tf.pad(db, [[0,0],[1,0]], constant_values=0)
        ub = tf.math.cumsum(ub, axis=1)
        ub = tf.math.minimum(ub, 1)
        ub = tf.expand_dims(ub, axis=-1)
        
        # Bezier layer
        # Compute values of basis functions at data points
        num_control_points = self.bezier_degree + 1
        lbs = tf.tile(ub, [1, 1, num_control_points])
        pw1 = tf.range(0, num_control_points, dtype=tf.float32)
        pw1 = tf.reshape(pw1, [1, 1, -1])
        pw2 = tf.reverse(pw1, axis=[-1])
        lbs = tf.math.add(tf.math.multiply(pw1, tf.math.log(lbs+self.EPSILON)), tf.math.multiply(pw2, tf.math.log(1-lbs+self.EPSILON)))
        lc = tf.add(tf.math.lgamma(pw1+1), tf.math.lgamma(pw2+1))
        lc = tf.math.subtract(tf.math.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc)
        lbs = tf.math.add(lbs, lc)
        bs = tf.math.exp(lbs)
        # Compute data points
        cp_w = tf.math.multiply(cp, w)
        dp = tf.matmul(bs, cp_w)
        bs_w = tf.matmul(bs, w)
        dp = tf.math.divide(dp, bs_w)
        dp = tf.expand_dims(dp, axis=-1)
        
        return dp, cp, w, ub, db


################################################################################
################################### Estimator ##################################
################################################################################
class Estimator(keras.Model):
    def __init__(self):
        super(Estimator, self).__init__()
        
        depth = 16
        kernel_size = (4,2)
        residual_list = [2, 2, 2, 2]
        weight_init = keras.initializers.he_uniform()
        weight_regularizer = keras.regularizers.l2(0.0001)

        self.optimizer = keras.optimizers.Adam(beta_1 = 0.5)


        self.Conv1 = keras.layers.Conv2D(depth*1, kernel_size, strides=1, padding='same', 
                                 kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)
        
        self.ResBlocks = []
        for _ in range(residual_list[0]):
            self.ResBlocks.append(Residual_Block(channels=depth, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=False))
        
        self.ResBlocks.append(Residual_Block(channels=depth*2, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=True))       
        for _ in range(1, residual_list[1]):
            self.ResBlocks.append(Residual_Block(channels=depth*2, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=False))
        
        self.ResBlocks.append(Residual_Block(channels=depth*4, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=True))
        for _ in range(1, residual_list[2]):
            self.ResBlocks.append(Residual_Block(channels=depth*4, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=False))
        
        self.ResBlocks.append(Residual_Block(channels=depth*8, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=True))
        for _ in range(1, residual_list[3]):
            self.ResBlocks.append(Residual_Block(channels=depth*8, kernel_size=kernel_size, kernel_initializer=weight_init, 
                                                 kernel_regularizer=weight_regularizer, downsample=False))
        
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU1 = keras.layers.ReLU()

        self.flatten = keras.layers.Flatten()

        self.dense1 = keras.layers.Dense(128, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU2 = keras.layers.ReLU()

        self.dense2 = keras.layers.Dense(1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer)
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

    def call(self, inputs, training=True):
        
        x = inputs
        x = self.Conv1(x)
        
        for i in range(len(self.ResBlocks)):
            x = self.ResBlocks[i](x, training = training)

        x = self.batchnorm1(x, training=training)
        x = self.LReLU1(x)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm2(x, training=training)
        x = self.LReLU2(x)
        
        y = self.dense2(x)
        y = self.sigmoid(y)
        
        return y, x


################################################################################
#################################### CcGAN #####################################
################################################################################
class CcDiscriminator(keras.Model):
    def __init__(self, latent_dim):
        super(CcDiscriminator, self).__init__()
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        self.latent_dim = latent_dim
        
        self.cond_Dense_norm = tfa.layers.SpectralNormalization(keras.layers.Dense(12288))
        
        self.Conv1 = keras.layers.Conv2D(depth*1, kernel_size, strides=(2,1), padding='same')
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU1 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout1 = keras.layers.Dropout(dropout)

        self.Conv2 = keras.layers.Conv2D(depth*2, kernel_size, strides=(2,1), padding='same')
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU2 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout2 = keras.layers.Dropout(dropout)

        self.Conv3 = keras.layers.Conv2D(depth*4, kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU3 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout3 = keras.layers.Dropout(dropout)

        self.Conv4 = keras.layers.Conv2D(depth*8, kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU4 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout4 = keras.layers.Dropout(dropout)

        self.Conv5 = keras.layers.Conv2D(depth*16, kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU5 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout5 = keras.layers.Dropout(dropout)

        self.Conv6 = keras.layers.Conv2D(depth*32, kernel_size, strides=(2,1), padding='same')
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU6 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout6 = keras.layers.Dropout(dropout)

        self.flatten = keras.layers.Flatten()
        
        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(1)

        self.dense3 = keras.layers.Dense(128)

        self.LReLU8 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense3 = keras.layers.Dense(128)
        self.dense4 = keras.layers.Dense(latent_dim)
        self.dense5 = keras.layers.Dense(latent_dim)
        

    def call(self, inputs, condition, training = True):
        
        cond = self.cond_Dense_norm(condition)
        
        x = inputs
        
        x = self.Conv1(x)
        x = self.batchnorm1(x, training=training)
        x = self.LReLU1(x)
        x = self.dropout1(x, training=training)
        
        x = self.Conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.LReLU2(x)
        x = self.dropout2(x, training=training)
        
        x = self.Conv3(x)
        x = self.batchnorm3(x, training=training)
        x = self.LReLU3(x)
        x = self.dropout3(x, training=training)
        
        x = self.Conv4(x)
        x = self.batchnorm4(x, training=training)
        x = self.LReLU4(x)
        x = self.dropout4(x, training=training)
        
        x = self.Conv5(x)
        x = self.batchnorm5(x, training=training)
        x = self.LReLU5(x)
        x = self.dropout5(x, training=training)
        
        x = self.Conv6(x)
        x = self.batchnorm6(x, training=training)
        x = self.LReLU6(x)
        x = self.dropout6(x, training=training)
        
        x = self.flatten(x)
        
        cond = tf.expand_dims(tf.reduce_sum(cond*x, axis=-1),-1)
        
        x = self.dense1(x)
        x = self.batchnorm7(x, training=False)
        x = self.LReLU7(x)
        
        d = self.dense2(x) + cond
        
        q = self.dense3(x)

        q = self.LReLU8(q)
        q_mean = self.dense4(q)
        q_logstd = self.dense5(q)
        q_logstd = tf.math.maximum(q_logstd, -16)
        q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
        q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
        q = tf.concat([q_mean, q_logstd], axis=1)
        
        return d, q
    
    
class CcGenerator(keras.Model):
    def __init__(self, bezier_degree, output_size):
        super(CcGenerator, self).__init__()
        depth_cpw = 32*8
        dim_cpw = int((bezier_degree+1)/8)
        kernel_size = (4,3)
        self.dim_cpw = dim_cpw
        self.depth_cpw = depth_cpw
        self.bezier_degree = bezier_degree
        self.EPSILON = 1e-7

        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu1 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(dim_cpw*3*depth_cpw)
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu2 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT1 = keras.layers.Conv2DTranspose(int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = ConditionalBatchNormalization2D(int(depth_cpw/2))
        self.LReLu3 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT2 = keras.layers.Conv2DTranspose(int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = ConditionalBatchNormalization2D(int(depth_cpw/4))
        self.LReLu4 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT3 = keras.layers.Conv2DTranspose(int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = ConditionalBatchNormalization2D(int(depth_cpw/8))
        self.LReLu5 = keras.layers.LeakyReLU(alpha=0.2)

        self.Conv1 = keras.layers.Conv2D(1, (1,2), padding='valid')

        self.Conv2 = keras.layers.Conv2D(1, (1,3), padding='valid')
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

        self.dense3 = keras.layers.Dense(1024)
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu6 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense4 = keras.layers.Dense(256)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense5 = keras.layers.Dense(output_size - 1)
        self.softmax = keras.layers.Softmax()

    def call(self, c, z, condition, training=True):
        
        if z == None:
                cz = c
        else:
            cz = tf.concat([c, z], axis=-1)
        


        cpw = self.dense1(cz)
        cpw = self.batchnorm1(cpw, training=training)
        cpw = self.LReLu1(cpw)

        cpw = self.dense2(cpw)
        cpw = self.batchnorm2(cpw, training=training)
        cpw = self.LReLu2(cpw)
        cpw = tf.reshape(cpw, (-1, self.dim_cpw, 3, self.depth_cpw))

        cpw = self.ConvT1(cpw)
        cpw = self.batchnorm3(cpw, condition)
        cpw = self.LReLu3(cpw)
        
        cpw = self.ConvT2(cpw)
        cpw = self.batchnorm4(cpw, condition)
        cpw = self.LReLu4(cpw)
        
        cpw = self.ConvT3(cpw)
        cpw = self.batchnorm5(cpw, condition)
        cpw = self.LReLu5(cpw)
        
        # Control points
        cp = self.Conv1(cpw)
        cp = tf.math.tanh(cp)
        cp = tf.squeeze(cp, axis=-1)
        
        # Weights
        w = self.Conv2(cpw)
        w = self.sigmoid(w)
        w = tf.squeeze(w, axis=-1)
        
        # Parameters at data points
        db = self.dense3(cz)
        db = self.batchnorm6(db, training=training)
        db = self.LReLu6(db)
        
        db = self.dense4(db)
        db = self.batchnorm7(db, training=training)
        db = self.LReLu7(db)
        
        db = self.dense5(db)
        db = self.softmax(db)
        
        ub = tf.pad(db, [[0,0],[1,0]], constant_values=0)
        ub = tf.math.cumsum(ub, axis=1)
        ub = tf.math.minimum(ub, 1)
        ub = tf.expand_dims(ub, axis=-1)
        
        # Bezier layer
        # Compute values of basis functions at data points
        num_control_points = self.bezier_degree + 1
        lbs = tf.tile(ub, [1, 1, num_control_points])
        pw1 = tf.range(0, num_control_points, dtype=tf.float32)
        pw1 = tf.reshape(pw1, [1, 1, -1])
        pw2 = tf.reverse(pw1, axis=[-1])
        lbs = tf.math.add(tf.math.multiply(pw1, tf.math.log(lbs+self.EPSILON)), tf.math.multiply(pw2, tf.math.log(1-lbs+self.EPSILON)))
        lc = tf.add(tf.math.lgamma(pw1+1), tf.math.lgamma(pw2+1))
        lc = tf.math.subtract(tf.math.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc)
        lbs = tf.math.add(lbs, lc)
        bs = tf.math.exp(lbs)
        # Compute data points
        cp_w = tf.math.multiply(cp, w)
        dp = tf.matmul(bs, cp_w)
        bs_w = tf.matmul(bs, w)
        dp = tf.math.divide(dp, bs_w)
        dp = tf.expand_dims(dp, axis=-1)
        
        return dp, cp, w, ub, db


################################################################################
################################ CcGAN Embedder ################################
################################################################################
class Embedder(keras.Model):
    def __init__(self, size):
        super(Embedder, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(size)
        self.gnorm1 = tfa.layers.GroupNormalization(8)
        self.ReLU1 = tf.keras.layers.ReLU()
        
        self.dense2 = tf.keras.layers.Dense(size)
        self.gnorm2 = tfa.layers.GroupNormalization(8)
        self.ReLU2 = tf.keras.layers.ReLU()
        
        self.dense3 = tf.keras.layers.Dense(size)
        self.gnorm3 = tfa.layers.GroupNormalization(8)
        self.ReLU3 = tf.keras.layers.ReLU()
        
        self.dense4 = tf.keras.layers.Dense(size)
        self.gnorm4 = tfa.layers.GroupNormalization(8)
        self.ReLU4 = tf.keras.layers.ReLU()
        
        self.dense5 = tf.keras.layers.Dense(size)
        self.gnorm5 = tfa.layers.GroupNormalization(8)
        self.ReLU5 = tf.keras.layers.ReLU()
        
        self.dense6 = tf.keras.layers.Dense(size)
        self.gnorm6 = tfa.layers.GroupNormalization(8)
        self.ReLU6 = tf.keras.layers.ReLU()
        
        self.dense7 = tf.keras.layers.Dense(size)
        self.ReLU7 = tf.keras.layers.ReLU()
        
    def call(self, inputs):
        
        x = self.dense1(inputs)
        x = self.gnorm1(x)
        x = self.ReLU1(x)
        
        x = self.dense2(x)
        x = self.gnorm2(x)
        x = self.ReLU2(x)
        
        x = self.dense3(x)
        x = self.gnorm3(x)
        x = self.ReLU3(x)
        
        x = self.dense4(x)
        x = self.gnorm4(x)
        x = self.ReLU4(x)
        
        x = self.dense5(x)
        x = self.gnorm5(x)
        x = self.ReLU5(x)
        
        x = self.dense6(x)
        x = self.gnorm6(x)
        x = self.ReLU6(x)
        
        x = self.dense7(x)
        x = self.ReLU7(x)
        
        return x


################################################################################
#################################### PcDGAN ####################################
################################################################################
class PcDGenerator(keras.Model):
    def __init__(self, bezier_degree, output_size):
        super(PcDGenerator, self).__init__()
        depth_cpw = 32*8
        dim_cpw = int((bezier_degree+1)/8)
        kernel_size = (4,3)
        self.dim_cpw = dim_cpw
        self.depth_cpw = depth_cpw
        self.bezier_degree = bezier_degree
        self.EPSILON = 1e-7
        cond_embedding_size = 10
        
        self.cond_emb = tf.Variable(tf.random.normal([1,cond_embedding_size],stddev=0.02))
        
        self.cond_dense = keras.layers.Dense(128)
        self.cond_LReLu = keras.layers.LeakyReLU(alpha=0.2)
        
        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu1 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(dim_cpw*3*depth_cpw)
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu2 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT1 = keras.layers.Conv2DTranspose(int(depth_cpw/2), kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = ConditionalBatchNormalization2D(int(depth_cpw/2))
        self.LReLu3 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT2 = keras.layers.Conv2DTranspose(int(depth_cpw/4), kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = ConditionalBatchNormalization2D(int(depth_cpw/4))
        self.LReLu4 = keras.layers.LeakyReLU(alpha=0.2)

        self.ConvT3 = keras.layers.Conv2DTranspose(int(depth_cpw/8), kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = ConditionalBatchNormalization2D(int(depth_cpw/8))
        self.LReLu5 = keras.layers.LeakyReLU(alpha=0.2)

        self.Conv1 = keras.layers.Conv2D(1, (1,2), padding='valid')

        self.Conv2 = keras.layers.Conv2D(1, (1,3), padding='valid')
        self.sigmoid = keras.layers.Activation(keras.activations.sigmoid)

        self.dense3 = keras.layers.Dense(1024)
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu6 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense4 = keras.layers.Dense(256)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLu7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense5 = keras.layers.Dense(output_size - 1)
        self.softmax = keras.layers.Softmax()

    def call(self, c, z, condition, training=True):
        
        if z == None:
                cz = c
        else:
            cz = tf.concat([c, z], axis=-1)
        
        cond = tf.tile(self.cond_emb,[tf.shape(condition)[0],1])+condition
        cond = self.cond_dense(cond)
        cond = self.cond_LReLu(cond)
        
        cpw = self.dense1(cz)
        cpw = self.batchnorm1(cpw, training=training)
        cpw = self.LReLu1(cpw)

        cpw = self.dense2(cpw)
        cpw = self.batchnorm2(cpw, training=training)
        cpw = self.LReLu2(cpw)
        cpw = tf.reshape(cpw, (-1, self.dim_cpw, 3, self.depth_cpw))

        cpw = self.ConvT1(cpw)
        cpw = self.batchnorm3(cpw, cond)
        cpw = self.LReLu3(cpw)
        
        cpw = self.ConvT2(cpw)
        cpw = self.batchnorm4(cpw, cond)
        cpw = self.LReLu4(cpw)
        
        cpw = self.ConvT3(cpw)
        cpw = self.batchnorm5(cpw, cond)
        cpw = self.LReLu5(cpw)
        
        # Control points
        cp = self.Conv1(cpw)
        cp = tf.math.tanh(cp)
        cp = tf.squeeze(cp, axis=-1)
        
        # Weights
        w = self.Conv2(cpw)
        w = self.sigmoid(w)
        w = tf.squeeze(w, axis=-1)
        
        # Parameters at data points
        db = self.dense3(cz)
        db = self.batchnorm6(db, training=training)
        db = self.LReLu6(db)
        
        db = self.dense4(db)
        db = self.batchnorm7(db, training=training)
        db = self.LReLu7(db)
        
        db = self.dense5(db)
        db = self.softmax(db)
        
        ub = tf.pad(db, [[0,0],[1,0]], constant_values=0)
        ub = tf.math.cumsum(ub, axis=1)
        ub = tf.math.minimum(ub, 1)
        ub = tf.expand_dims(ub, axis=-1)
        
        # Bezier layer
        # Compute values of basis functions at data points
        num_control_points = self.bezier_degree + 1
        lbs = tf.tile(ub, [1, 1, num_control_points])
        pw1 = tf.range(0, num_control_points, dtype=tf.float32)
        pw1 = tf.reshape(pw1, [1, 1, -1])
        pw2 = tf.reverse(pw1, axis=[-1])
        lbs = tf.math.add(tf.math.multiply(pw1, tf.math.log(lbs+self.EPSILON)), tf.math.multiply(pw2, tf.math.log(1-lbs+self.EPSILON)))
        lc = tf.add(tf.math.lgamma(pw1+1), tf.math.lgamma(pw2+1))
        lc = tf.math.subtract(tf.math.lgamma(tf.cast(num_control_points, dtype=tf.float32)), lc)
        lbs = tf.math.add(lbs, lc)
        bs = tf.math.exp(lbs)
        # Compute data points
        cp_w = tf.math.multiply(cp, w)
        dp = tf.matmul(bs, cp_w)
        bs_w = tf.matmul(bs, w)
        dp = tf.math.divide(dp, bs_w)
        dp = tf.expand_dims(dp, axis=-1)
        
        return dp, cp, w, ub, db
        
class PcDDiscriminator(keras.Model):
    def __init__(self, latent_dim):
        super(PcDDiscriminator, self).__init__()
        depth = 64
        dropout = 0.4
        kernel_size = (4,2)
        self.latent_dim = latent_dim
        cond_embedding_size = 10
        
        self.cond_Dense_norm = tfa.layers.SpectralNormalization(keras.layers.Dense(12288))
        
        self.cond_emb = tf.Variable(tf.random.normal([1,cond_embedding_size],stddev=0.02))
        
        self.cond_dense = keras.layers.Dense(128)
        self.cond_LReLu = keras.layers.LeakyReLU(alpha=0.2)

        self.Conv1 = keras.layers.Conv2D(depth*1, kernel_size, strides=(2,1), padding='same')
        self.batchnorm1 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU1 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout1 = keras.layers.Dropout(dropout)

        self.Conv2 = keras.layers.Conv2D(depth*2, kernel_size, strides=(2,1), padding='same')
        self.batchnorm2 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU2 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout2 = keras.layers.Dropout(dropout)

        self.Conv3 = keras.layers.Conv2D(depth*4, kernel_size, strides=(2,1), padding='same')
        self.batchnorm3 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU3 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout3 = keras.layers.Dropout(dropout)

        self.Conv4 = keras.layers.Conv2D(depth*8, kernel_size, strides=(2,1), padding='same')
        self.batchnorm4 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU4 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout4 = keras.layers.Dropout(dropout)

        self.Conv5 = keras.layers.Conv2D(depth*16, kernel_size, strides=(2,1), padding='same')
        self.batchnorm5 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU5 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout5 = keras.layers.Dropout(dropout)

        self.Conv6 = keras.layers.Conv2D(depth*32, kernel_size, strides=(2,1), padding='same')
        self.batchnorm6 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU6 = keras.layers.LeakyReLU(alpha=0.2)
        self.dropout6 = keras.layers.Dropout(dropout)

        self.flatten = keras.layers.Flatten()
        
        self.dense1 = keras.layers.Dense(1024)
        self.batchnorm7 = keras.layers.BatchNormalization(momentum=0.9)
        self.LReLU7 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense2 = keras.layers.Dense(1)

        self.dense3 = keras.layers.Dense(128)

        self.LReLU8 = keras.layers.LeakyReLU(alpha=0.2)

        self.dense3 = keras.layers.Dense(128)
        self.dense4 = keras.layers.Dense(latent_dim)
        self.dense5 = keras.layers.Dense(latent_dim)
        

    def call(self, inputs, condition, training = True):
        
        cond = tf.tile(self.cond_emb,[tf.shape(condition)[0],1])+condition
        cond = self.cond_dense(cond)
        cond = self.cond_LReLu(cond)
        cond = self.cond_Dense_norm(cond)
        
        x = inputs
        
        x = self.Conv1(x)
        x = self.batchnorm1(x, training=training)
        x = self.LReLU1(x)
        x = self.dropout1(x, training=training)
        
        x = self.Conv2(x)
        x = self.batchnorm2(x, training=training)
        x = self.LReLU2(x)
        x = self.dropout2(x, training=training)
        
        x = self.Conv3(x)
        x = self.batchnorm3(x, training=training)
        x = self.LReLU3(x)
        x = self.dropout3(x, training=training)
        
        x = self.Conv4(x)
        x = self.batchnorm4(x, training=training)
        x = self.LReLU4(x)
        x = self.dropout4(x, training=training)
        
        x = self.Conv5(x)
        x = self.batchnorm5(x, training=training)
        x = self.LReLU5(x)
        x = self.dropout5(x, training=training)
        
        x = self.Conv6(x)
        x = self.batchnorm6(x, training=training)
        x = self.LReLU6(x)
        x = self.dropout6(x, training=training)
        
        x = self.flatten(x)
        
        cond = tf.expand_dims(tf.reduce_sum(cond*x, axis=-1),-1)
        
        x = self.dense1(x)
        x = self.batchnorm7(x, training=False)
        x = self.LReLU7(x)
        
        d = self.dense2(x) + cond
        
        q = self.dense3(x)

        q = self.LReLU8(q)
        q_mean = self.dense4(q)
        q_logstd = self.dense5(q)
        q_logstd = tf.math.maximum(q_logstd, -16)
        q_mean = tf.reshape(q_mean, (-1, 1, self.latent_dim))
        q_logstd = tf.reshape(q_logstd, (-1, 1, self.latent_dim))
        q = tf.concat([q_mean, q_logstd], axis=1)
        
        return d, q
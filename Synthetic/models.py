import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange
import math


class CcDiscriminator(keras.Model):
    def __init__(self):
        super(CcDiscriminator, self).__init__()
        
        self.Dense1 = keras.layers.Dense(128)
        self.LReLU1 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense2 = keras.layers.Dense(128)
        self.LReLU2 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense3 = keras.layers.Dense(128)
        self.LReLU3 = keras.layers.LeakyReLU(alpha = 0.2)

        self.Dense4 = keras.layers.Dense(128)
        self.LReLU4 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense5 = keras.layers.Dense(1)
    
    def call(self, inputs, condition):
        
        x = tf.concat([inputs,condition],-1)
         
        x = self.Dense1(x)
        x = self.LReLU1(x)
        
        x = self.Dense2(x)
        x = self.LReLU2(x)
        
        x = self.Dense3(x)
        x = self.LReLU3(x)

        x = self.Dense4(x)
        x = self.LReLU4(x)
        
        x = self.Dense5(x)
        
        return x
    
class CcGenerator(keras.Model):
    def __init__(self, data_dim):
        super(CcGenerator, self).__init__()
        
        self.Dense1 = keras.layers.Dense(128)
        self.LReLU1 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense2 = keras.layers.Dense(128)
        self.LReLU2 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense3 = keras.layers.Dense(128)
        self.LReLU3 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense4 = keras.layers.Dense(128)
        self.LReLU4 = keras.layers.LeakyReLU(alpha = 0.2)

        self.Dense5 = keras.layers.Dense(128)
        self.LReLU5 = keras.layers.LeakyReLU(alpha = 0.2)
        
        self.Dense6 = keras.layers.Dense(data_dim)
        self.out_activation = keras.layers.Activation(keras.activations.tanh)
        
    def call(self, inputs, condition):
        
        x = tf.concat([inputs,condition],-1)

        x = self.Dense1(x)
        x = self.LReLU1(x)
        
        x = self.Dense2(x)
        x = self.LReLU2(x)
        
        x = self.Dense3(x)
        x = self.LReLU3(x)
        
        x = self.Dense4(x)
        x = self.LReLU4(x)

        x = self.Dense5(x)
        x = self.LReLU5(x)
        
        x = self.Dense6(x)
        x = self.out_activation(x)
        
        return x
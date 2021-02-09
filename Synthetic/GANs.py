import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange
import math
from models import CcDiscriminator, CcGenerator
from utils import lambert_w_log_exp_score
            

class CcGAN(keras.Model):
    def __init__(self, noise_dim = 2, data_dim = 2, sigma=-1, kappa=-2, strategy='soft', nonzero_soft_weight_threshold=1e-3,Y=None):
        super(CcGAN, self).__init__()
        
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.EPSILON = 1e-7
        self.sigma = sigma
        self.kappa = kappa
        self.strategy = strategy
        self.nonzero_soft_weight_threshold = nonzero_soft_weight_threshold

        self.generator = CcGenerator(data_dim)
        self.discriminator = CcDiscriminator()
        
        if kappa<0.0 and type(Y)!=type(None): 
            y_sorted = np.sort(np.unique(Y))
            kappa_base = abs(kappa)*np.max(y_sorted[1:] - y_sorted[0:-1])

            if strategy == 'soft':
                self.kappa = 1/kappa_base**2
            else:
                self.kappa = kappa_base
            
            print('kappa: %f'%(self.kappa))
        
        elif kappa<0.0:
            self.kappa = 0.02
            print('kappa: %f'%(self.kappa))

        else:
            self.kappa = kappa
            print('kappa: %f'%(self.kappa))
        
        if sigma<0.0 and type(Y)!=type(None):
            std = np.std(Y)
            self.sigma = 1.06*std*(len(Y))**(-1/5)

            print('sigma: %f'%(self.sigma))
        
        elif sigma<0.0:
            self.sigma = 0.05
            print('sigma: %f'%(self.sigma))

        else:
            self.sigma = sigma
            print('sigma: %f'%(self.sigma))
        
    def get_batch(self, X, Y, batch_size):
        
        #Vicinal Batch
        sigma = self.sigma
        kappa = self.kappa
        
        batch_labels = np.random.choice(np.squeeze(np.unique(Y)),size=batch_size,replace=True)
        batch_epsilons = np.random.normal(0.0, sigma, batch_size)
        batch_target_labels = np.clip(batch_labels + batch_epsilons, 0.0, 1.0)
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        batch_fake_labels = np.zeros(batch_size)
        
        for j in range(batch_size):
            if self.strategy == 'hard':
                indx_real_in_vicinity = np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]
            else:
                indx_real_in_vicinity = np.where((Y-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]

            while indx_real_in_vicinity.shape[0] < 1:
                batch_target_labels[j] = np.clip(batch_labels[j] + np.random.normal(0.0, sigma), 0.0, 1.0)

                if self.strategy == 'hard':
                    indx_real_in_vicinity = np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]
                else:
                    indx_real_in_vicinity = np.where((Y-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]
            
            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
            
            if self.strategy == 'hard':
                lb = batch_target_labels[j] - kappa
                ub = batch_target_labels[j] + kappa
            else:
                lb = batch_target_labels[j] - np.sqrt(-np.log(self.nonzero_soft_weight_threshold)/kappa)
                ub = batch_target_labels[j] + np.sqrt(-np.log(self.nonzero_soft_weight_threshold)/kappa)

            lb = max(0.0, lb)
            ub = min(ub, 1.0)

            batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
        
        X_real = X[batch_real_indx]
        Y_real = Y[batch_real_indx]
        batch_fake_labels = np.expand_dims(batch_fake_labels,-1)
        batch_target_labels = np.expand_dims(batch_target_labels,-1)
        
        return X_real, Y_real, batch_fake_labels, batch_target_labels
    
    @tf.function
    def train_step(self, X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer, g_optimizer,equation):
        
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = X_real.shape[0]
        
        if self.strategy == "soft":
            real_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
            fake_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
        else:
            real_weights = tf.ones(batch_size)
            fake_weights = tf.ones(batch_size)
        
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        X_fake = self.generator(z, batch_fake_labels, training=False)
        
        #Discriminator Training
        with tf.GradientTape() as tape:
            d_real = self.discriminator(X_real, batch_target_labels)
            d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real,sample_weight=real_weights)
            d_fake = self.discriminator(X_fake, batch_target_labels)
            d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake,sample_weight=fake_weights)
            loss = d_loss_fake + d_loss_real
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(loss, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        
        #Generator Training
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        
        with tf.GradientTape() as tape:
            x_fake_train = self.generator(z, batch_target_labels)
            d_fake = self.discriminator(x_fake_train, batch_target_labels)
            g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)
            loss_total = g_loss
        
        variables = self.generator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        g_optimizer.apply_gradients(zip(gradients, variables))
        
        return d_loss_real, d_loss_fake, g_loss
    
    def train(self, X, Y, equation, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = train_steps//10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps//10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)
        
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")
        for step in steps:
            X_real, Y_real, batch_fake_labels, batch_target_labels = self.get_batch(X, Y, batch_size)
            X_real, Y_real, batch_fake_labels, batch_target_labels = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(batch_fake_labels, tf.float32), tf.cast(batch_target_labels, tf.float32)
            d_loss_real, d_loss_fake, g_loss = self.train_step(X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer,g_optimizer,equation)
            log_mesg = "%d: [D] real %+.7f fake %+.7f" % (step+1, d_loss_real, d_loss_fake)
            log_mesg = "%s  [G] fake %+.7f lr %+.7f" % (log_mesg, g_loss, g_optimizer._decayed_lr('float32'))
               
            steps.set_postfix_str(log_mesg)
            
class PcDGAN(keras.Model):
    def __init__(self, noise_dim = 2, data_dim = 2, lambda0=2.0, lambda1=0.5, sigma=-1, kappa=-1, lambert_cutoff=4.7, strategy='soft', nonzero_soft_weight_threshold=1e-3,Y=None):
        super(PcDGAN, self).__init__()
        
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.EPSILON = 1e-7
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambert_cutoff = lambert_cutoff
        self.nonzero_soft_weight_threshold = nonzero_soft_weight_threshold
        self.strategy = strategy

        self.generator = CcGenerator(data_dim)
        self.discriminator = CcDiscriminator()

        if kappa<0.0 and type(Y)!=type(None): 
            y_sorted = np.sort(np.unique(Y))
            kappa_base = abs(kappa)*np.max(y_sorted[1:] - y_sorted[0:-1])

            if strategy == 'soft':
                self.kappa = 1/kappa_base**2
            else:
                self.kappa = kappa_base
            
            print('kappa: %f'%(self.kappa))
        
        elif kappa<0.0:
            self.kappa = 0.02
            print('kappa: %f'%(self.kappa))

        else:
            self.kappa = kappa
            print('kappa: %f'%(self.kappa))
        
        if sigma<0.0 and type(Y)!=type(None):
            std = np.std(Y)
            self.sigma = 1.06*std*(len(Y))**(-1/5)*abs(sigma)

            print('sigma: %f'%(self.sigma))
        
        elif sigma<0.0:
            self.sigma = 0.05
            print('sigma: %f'%(self.sigma))

        else:
            self.sigma = sigma
            print('sigma: %f'%(self.sigma))

        
    def compute_diversity_loss(self, x, y):
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5 * tf.math.square(D))
        y = tf.squeeze(y)
        
        if self.lambda0 == 'inf':

            eig_val, _ = tf.linalg.eigh(S)
            loss = -10 * tf.reduce_mean(y)

            Q = None
            L = None

        elif self.lambda0 == 'naive':

            eig_val, _ = tf.linalg.eigh(S)
            loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, self.EPSILON))) - 10 * tf.reduce_mean(y)

            Q = None
            L = None

        else:
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1)
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.math.pow(Q, self.lambda0)

            eig_val, _ = tf.linalg.eigh(L)
            loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, self.EPSILON)))

        return loss, D, S, Q, L, y
    
    def get_batch(self, X, Y, batch_size):
        
        #GAN batch
        sigma = self.sigma
        kappa = self.kappa
        
        #get singular vicinity or SSVDL or SHVDL
        batch_labels = np.ones([batch_size]) * np.random.uniform(low=np.min(Y),high=np.max(Y))
        
        if self.strategy == 'hard':
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_labels[0])<= kappa)[0]
        else:
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_labels[0])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]
            
        while indx_real_in_vicinity.shape[0] == 0:
            batch_labels = np.ones([batch_size]) * np.random.uniform(low=np.min(Y),high=np.max(Y))
            if self.strategy == 'hard':
                indx_real_in_vicinity =  np.where(np.abs(Y-batch_labels[0])<= kappa)[0]
            else:
                indx_real_in_vicinity =  np.where((Y-batch_labels[0])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]
        
        batch_target_labels = np.clip(batch_labels + np.random.normal(0.0, sigma, batch_size),0.0,1.0)
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        batch_fake_labels = np.zeros(batch_size)
        
        for j in range(batch_size):
            if self.strategy == 'hard':
                indx_real_in_vicinity = np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]
            else:
                indx_real_in_vicinity = np.where((Y-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]

            while indx_real_in_vicinity.shape[0] < 1:
                batch_target_labels[j] = np.clip(batch_labels[j] + np.random.normal(0.0, sigma), 0.0, 1.0)

                if self.strategy == 'hard':
                    indx_real_in_vicinity = np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]
                else:
                    indx_real_in_vicinity = np.where((Y-batch_target_labels[j])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]
            
            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
            
            if self.strategy == 'hard':
                lb = batch_target_labels[j] - kappa
                ub = batch_target_labels[j] + kappa
            else:
                lb = batch_target_labels[j] - np.sqrt(-np.log(self.nonzero_soft_weight_threshold)/kappa)
                ub = batch_target_labels[j] + np.sqrt(-np.log(self.nonzero_soft_weight_threshold)/kappa)

            lb = max(0.0, lb)
            ub = min(ub, 1.0)

            batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
        
        X_real = X[batch_real_indx]
        Y_real = Y[batch_real_indx]
        batch_fake_labels = np.expand_dims(batch_fake_labels,-1)
        batch_target_labels = np.expand_dims(batch_target_labels,-1)
        
        return X_real, Y_real, batch_fake_labels, batch_target_labels
    
    @tf.function
    def train_step(self, X_real, Y_real, equation, batch_fake_labels, batch_target_labels, d_optimizer, g_optimizer):
        
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = X_real.shape[0]

        if self.strategy == "soft":
            real_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
            fake_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
        else:
            real_weights = tf.ones(batch_size)
            fake_weights = tf.ones(batch_size)
        
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        X_fake = self.generator(z, batch_fake_labels, training=False)
        
        #Discriminator Training
        with tf.GradientTape() as tape:
            d_fake = self.discriminator(X_fake,batch_target_labels)
            d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake,sample_weight=fake_weights)
            d_real = self.discriminator(X_real,batch_target_labels)
            d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real,sample_weight=real_weights)
            loss = d_loss_fake + d_loss_real
            
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(loss, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        
        #Generator Training
        with tf.GradientTape() as tape:
            x_fake_train = self.generator(z, batch_target_labels)
            d_fake = self.discriminator(x_fake_train, batch_target_labels)
            g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)

            y = tf.expand_dims(equation(x_fake_train),-1)
            cond_score = lambert_w_log_exp_score(tf.math.abs(y-batch_target_labels),self.lambert_cutoff)
            
            dpp_loss = self.compute_diversity_loss(x_fake_train,cond_score)[0]
            
            loss_total = g_loss + self.lambda1 * dpp_loss
        
        variables = self.generator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        g_optimizer.apply_gradients(zip(gradients, variables))
        
        return d_loss_real, d_loss_fake, g_loss, tf.reduce_mean(cond_score), dpp_loss
    
    def train(self, X, equation, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = 2*train_steps//10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps//10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)
        
        Y = np.expand_dims(equation(X),-1)
        
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")
        for step in steps:
            X_real, Y_real, batch_fake_labels, batch_target_labels = self.get_batch(X, Y, batch_size)
            X_real, Y_real, batch_fake_labels, batch_target_labels = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(batch_fake_labels, tf.float32), tf.cast(batch_target_labels, tf.float32)
            d_loss_real, d_loss_fake, g_loss, cond_score, dpp_loss = self.train_step(X_real, Y_real, equation, batch_fake_labels, batch_target_labels, d_optimizer,g_optimizer)
            log_mesg = "%d: [D] real %+.7f fake %+.7f" % (step+1, d_loss_real, d_loss_fake)
            log_mesg = "%s  [G] fake %+.7f dpp %+.7f [C] cs %+.7f lr %+.7f" % (log_mesg, g_loss, dpp_loss, cond_score, g_optimizer._decayed_lr('float32'))
               
            steps.set_postfix_str(log_mesg)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm import trange
import math
from models import CcGenerator, CcDiscriminator, Estimator, Embedder, PcDGenerator, PcDDiscriminator
from utils import lambert_w_log_exp_score
import pickle
from glob import glob

tf.autograph.set_verbosity(0)

class CcGAN(keras.Model):
    def __init__(self, latent_dim=5, noise_dim=10, n_points=192, bezier_degree=31, bounds=(0.0, 1.0), sigma=-1, kappa=-1, strategy='soft', nonzero_soft_weight_threshold=1e-3,Y=None):
        super(CcGAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.n_points = n_points
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        self.EPSILON = 1e-7
        self.sigma = sigma
        self.kappa = kappa
        self.strategy = strategy
        self.nonzero_soft_weight_threshold = nonzero_soft_weight_threshold

        self.generator = CcGenerator(bezier_degree,n_points)
        self.discriminator = CcDiscriminator(latent_dim)
        self.estimator = Estimator()
        self.embedder = Embedder(128)
        
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

    def get_balanced_batch(self, X, Y, batch_size):
        
        kappa = 0.02
        
        batch_target_labels = np.random.uniform(low=np.min(Y),high=np.max(Y),size=[batch_size])
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        
        for j in range(batch_size):
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            while indx_real_in_vicinity.shape[0] == 0:
                batch_target_labels[j] = np.random.uniform(low=np.min(Y), high=np.max(Y))
                indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
            
        X_batch = X[batch_real_indx]
        Y_batch = Y[batch_real_indx]
        
        return X_batch, Y_batch
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def estimator_train_step(self, X_batch, Y_batch, optimizer):
        
        loss_fn = keras.losses.MeanAbsoluteError()
        loss_fn_mse = keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            y_pred = self.estimator(X_batch)[0]
            loss = loss_fn_mse(Y_batch,y_pred)
 
            L1 = loss_fn(Y_batch,y_pred)
            
        variables = self.estimator.trainable_weights
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        return loss, L1
    
    
    def train_estimator(self, X_train, Y_train, X_test, Y_test, batch_size=256, train_steps=10000, lr=1e-4, balanced_training=True, early_stop_save=None):
        
        lr = keras.optimizers.schedules.ExponentialDecay(lr,decay_steps = train_steps//4, decay_rate = 0.4642, staircase=True)
        optimizer = keras.optimizers.Adam(lr,beta_1 = 0.5)
        steps = trange(train_steps, desc='Training estimator Model', leave=True, ascii ="         =")
        
        validation_metric1 = keras.losses.MeanAbsoluteError()
        validation_metric2 = keras.losses.MeanSquaredError()
        
        Y_pred = self.estimator(X_test)[0]
        m1 = validation_metric1(Y_test,Y_pred)
        m2 = validation_metric2(Y_test,Y_pred)
        
        best = m1
        best_train = -1.0
        
        
        for step in steps:
            if balanced_training:
                X_batch,Y_batch = self.get_balanced_batch(X_train,Y_train,batch_size)
            
            else:
                ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                X_batch = X_train[ind]
                Y_batch = Y_train[ind]
            
            loss, L1 = self.estimator_train_step(X_batch,Y_batch,optimizer)
            
            if (step+1)%50 == 0:
                Y_pred = self.estimator(X_test,training=False)[0]
                m1 = validation_metric1(Y_test,Y_pred)
                m2 = validation_metric2(Y_test,Y_pred)
                if early_stop_save and (m1<=best or m1<0.015):
                    best = m1
                    best_train = L1
                    self.estimator.save_weights(early_stop_save)

            
            steps.set_postfix_str('Train L2: %f | L1: %f, Validation L1: %f | L2: %f, lr: %f' % (loss,L1,m1,m2,optimizer._decayed_lr('float32')))
        print('Best Estimator Saved With: Validation_L1 = %f, Train_L1 = %f' % (best, best_train))
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def embedder_train_step(self, X_batch, Y_batch, optimizer):
        loss_fn = keras.losses.MeanSquaredError()
        loss_fn_mae = keras.losses.MeanAbsoluteError()
        with tf.GradientTape() as tape:
            h = self.estimator(X_batch)[1]
            h_pred = self.embedder(Y_batch)
            loss = loss_fn(h,h_pred)
            L1 = loss_fn_mae(h,h_pred)
            
        variables = self.embedder.trainable_weights
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        return loss, L1
    
    def train_embedder(self, X_train, Y_train, X_test, Y_test, batch_size=256, train_steps=10000, lr=1e-4, balanced_training=True, early_stop_save=None):
        
        lr = keras.optimizers.schedules.ExponentialDecay(lr,decay_steps = train_steps//4, decay_rate = 0.4642, staircase=True)
        optimizer = keras.optimizers.Adam(lr,beta_1 = 0.5)
        steps = trange(train_steps, desc='Training Embedder Model', leave=True, ascii ="         =")
        
        validation_metric1 = keras.losses.MeanAbsoluteError()
        validation_metric2 = keras.losses.MeanSquaredError()
        
        h = self.estimator(X_test)[1]
        h_pred = self.embedder(Y_test)
        m1 = validation_metric1(h,h_pred)
        m2 = validation_metric2(h,h_pred)
        
        best = m1
        best_train = -1.0

        for step in steps:
            if balanced_training:
                X_batch,Y_batch = self.get_balanced_batch(X_train,Y_train,batch_size)
            
            else:
                ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                X_batch = X_train[ind]
                Y_batch = Y_train[ind]
            
            loss,L1 = self.embedder_train_step(X_batch,Y_batch,optimizer)
            
            if (step+1)%50 == 0:
                h = self.estimator(X_test)[1]
                h_pred = self.embedder(Y_test)
                m1 = validation_metric1(h,h_pred)
                m2 = validation_metric2(h,h_pred)
                if early_stop_save and m1<=best:
                    best = m1
                    best_train = L1
                    self.embedder.save_weights(early_stop_save)
            
            steps.set_postfix_str('Train L2: %f | L1: %f, Validation L1: %f | L2: %f, lr: %f' % (loss,L1,m1,m2,optimizer._decayed_lr('float32')))
        print('Best Embedder Saved With: Validation_L1 = %f, Train_L1 = %f' % (best, best_train))
    
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
    @tf.autograph.experimental.do_not_convert
    def train_step(self, X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer, g_optimizer):
        
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = X_real.shape[0]

        if self.strategy == "soft":
            real_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
            fake_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
        else:
            real_weights = tf.ones(batch_size)
            fake_weights = tf.ones(batch_size)
        
        c = tf.random.uniform(minval=self.bounds[0], maxval = self.bounds[1], shape=(batch_size, self.latent_dim))
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        X_fake = self.generator(c, z, self.embedder(batch_fake_labels,training=False), training=False)[0]
        
        ############ Discriminator Training ############
        with tf.GradientTape() as tape:
            d_real, _ = self.discriminator(X_real, self.embedder(batch_target_labels,training=False))
            d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real,sample_weight=real_weights)
            d_fake, q_fake_train = self.discriminator(X_fake, self.embedder(batch_target_labels,training=False))
            d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake,sample_weight=fake_weights)
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
            q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
            q_loss = tf.reduce_mean(q_loss)
            loss_total = d_loss_fake + q_loss + d_loss_real
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        q_loss_d = q_loss


        ############ Generator Training ############
        
        c = tf.random.uniform(minval=self.bounds[0], maxval = self.bounds[1], shape=(batch_size, self.latent_dim))
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        
        with tf.GradientTape() as tape:
            x_fake_train, cp_train, w_train, _, db_train = self.generator(c, z, self.embedder(batch_target_labels,training=False))
            d_fake, q_fake_train = self.discriminator(x_fake_train, self.embedder(batch_target_labels,training=False))
            
            g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)
            
            r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
            cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
            r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
            ends = cp_train[:,0] - cp_train[:,-1]
            r_ends_loss = tf.norm(ends, axis=-1) + tf.math.maximum(0.0, -10*ends[:,1])
            r_loss = r_w_loss + r_cp_loss + r_ends_loss
            r_loss = tf.reduce_mean(r_loss)

            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
            q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
            q_loss = tf.reduce_mean(q_loss)
            
            loss_total = g_loss + 10*r_loss + q_loss
        
        variables = self.generator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        g_optimizer.apply_gradients(zip(gradients, variables))
        
        return d_loss_real, d_loss_fake, q_loss_d, g_loss, r_loss, q_loss
    
    def train(self, X, Y, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = train_steps/10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps/10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)
        
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")
        for step in steps:
            X_real, Y_real, batch_fake_labels, batch_target_labels = self.get_batch(X, Y, batch_size)
            X_real, Y_real, batch_fake_labels, batch_target_labels = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(batch_fake_labels, tf.float32), tf.cast(batch_target_labels, tf.float32)
            d_loss_real, d_loss_fake, q_loss_d, g_loss, r_loss, q_loss = self.train_step(X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer,g_optimizer)
            log_mesg = "%d: [D] real %+.7f fake %+.7f q %+.7f lr %+.7f" % (step+1, d_loss_real, d_loss_fake, q_loss_d, d_optimizer._decayed_lr('float32'))
            log_mesg = "%s  [G] fake %+.7f reg %+.7f q %+.7f lr %+.7f" % (log_mesg, g_loss, r_loss, q_loss, g_optimizer._decayed_lr('float32'))
               
            steps.set_postfix_str(log_mesg)
            
            
class PcDGAN(keras.Model):
    def __init__(self, latent_dim=5, noise_dim=10, n_points=192, bezier_degree=31, bounds=(0.0, 1.0), lambda0=2.0, lambda1=0.2, kappa=-1, sigma=-1, lambert_cutoff=4.7, strategy='soft', nonzero_soft_weight_threshold=1e-3, Y=None):
        super(PcDGAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.n_points = n_points
        self.bezier_degree = bezier_degree
        self.bounds = bounds
        self.EPSILON = 1e-7
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambert_cutoff = lambert_cutoff
        self.strategy=strategy
        self.nonzero_soft_weight_threshold=nonzero_soft_weight_threshold
        self.generator = PcDGenerator(bezier_degree,n_points)
        self.discriminator = PcDDiscriminator(latent_dim)
        self.estimator = Estimator()
        
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
        
    def get_balanced_batch(self, X, Y, batch_size):
        
        kappa = 0.02
        
        batch_target_labels = np.random.uniform(low=np.min(Y),high=np.max(Y),size=[batch_size])
        
        batch_real_indx = np.zeros(batch_size, dtype=int)
        
        for j in range(batch_size):
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            while indx_real_in_vicinity.shape[0] == 0:
                batch_target_labels[j] = np.random.uniform(low=np.min(Y), high=np.max(Y))
                indx_real_in_vicinity =  np.where(np.abs(Y-batch_target_labels[j])<= kappa)[0]

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]
            
        X_batch = X[batch_real_indx]
        Y_batch = Y[batch_real_indx]
        
        return X_batch, Y_batch
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def compute_diversity_loss(self, x, y):

        flatten = keras.layers.Flatten()    
        x = flatten(x)
        y = tf.squeeze(y)
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.math.exp(-0.5*tf.square(D))
        
        if self.lambda0 == 'naive':

            eig_val, _ = tf.linalg.eigh(S)
            loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, self.EPSILON)))-10*tf.reduce_mean(y)
            
            Q = None
            L = None
            
        else:
            
            Q = tf.tensordot(tf.expand_dims(y, 1), tf.expand_dims(y, 0), 1)
            if self.lambda0 == 0.:
                L = S
            else:
                L = S * tf.math.pow(Q, self.lambda0)
            
            eig_val, _ = tf.linalg.eigh(L)
            loss = -tf.reduce_mean(tf.math.log(tf.maximum(eig_val, self.EPSILON)))
        
        return loss, D, S, Q, L
    
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def estimator_train_step(self, X_batch, Y_batch, optimizer):
        
        loss_fn = keras.losses.MeanAbsoluteError()
        loss_fn_mse = keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            y_pred = self.estimator(X_batch)[0]
            loss = loss_fn_mse(Y_batch,y_pred)
 
            L1 = loss_fn(Y_batch,y_pred)
            
        variables = self.estimator.trainable_weights
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        return loss, L1
    
    
    def train_estimator(self, X_train, Y_train, X_test, Y_test, batch_size=256, train_steps=10000, lr=1e-4, balanced_training=True, early_stop_save=None):
        
        lr = keras.optimizers.schedules.ExponentialDecay(lr,decay_steps = train_steps//4, decay_rate = 0.4642, staircase=True)
        optimizer = keras.optimizers.Adam(lr,beta_1 = 0.5)
        steps = trange(train_steps, desc='Training estimator Model', leave=True, ascii ="         =")
        
        validation_metric1 = keras.losses.MeanAbsoluteError()
        validation_metric2 = keras.losses.MeanSquaredError()
        
        Y_pred = self.estimator(X_test)[0]
        m1 = validation_metric1(Y_test,Y_pred)
        m2 = validation_metric2(Y_test,Y_pred)
        
        best = m1
        best_train = -1.0
        
        
        for step in steps:
            if balanced_training:
                X_batch,Y_batch = self.get_balanced_batch(X_train,Y_train,batch_size)
            
            else:
                ind = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                X_batch = X_train[ind]
                Y_batch = Y_train[ind]
            
            loss, L1 = self.estimator_train_step(X_batch,Y_batch,optimizer)
            
            if (step+1)%50 == 0:
                Y_pred = self.estimator(X_test,training=False)[0]
                m1 = validation_metric1(Y_test,Y_pred)
                m2 = validation_metric2(Y_test,Y_pred)
                if early_stop_save and (m1<=best or m1<0.015):
                    best = m1
                    best_train = L1
                    self.estimator.save_weights(early_stop_save)

            
            steps.set_postfix_str('Train L2: %f | L1: %f, Validation L1: %f | L2: %f, lr: %f' % (loss,L1,m1,m2,optimizer._decayed_lr('float32')))
        print('Best Estimator Saved With: Validation_L1 = %f, Train_L1 = %f' % (best, best_train))
    
    def get_batch(self, X, Y, batch_size):
        
        #GAN batch
        sigma = self.sigma
        kappa = self.kappa
        
        #get singular vicinity or SSVDL or SHVDL
        batch_labels = np.ones([batch_size]) * np.random.uniform()
        
        if self.strategy == 'hard':
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_labels[0])<= kappa)[0]
        else:
            indx_real_in_vicinity =  np.where(np.abs(Y-batch_labels[0])**2 <= -np.log(self.nonzero_soft_weight_threshold)/kappa)[0]
            
        while indx_real_in_vicinity.shape[0] == 0:
            batch_labels = np.ones([batch_size]) * np.random.uniform()
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
    @tf.autograph.experimental.do_not_convert
    def train_step(self, X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer, g_optimizer,lambda1):
        
        binary_cross_entropy_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        batch_size = X_real.shape[0]

        if self.strategy == "soft":
            real_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
            fake_weights = tf.squeeze(tf.math.exp(-self.kappa*(Y_real-batch_target_labels)**2))
        else:
            real_weights = tf.ones(batch_size)
            fake_weights = tf.ones(batch_size)
        
        c = tf.random.uniform(minval=self.bounds[0], maxval = self.bounds[1], shape=(batch_size, self.latent_dim))
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        X_fake = self.generator(c, z, batch_fake_labels, training=False)[0]
        
        with tf.GradientTape() as tape:
            d_fake, q_fake_train = self.discriminator(X_fake,batch_target_labels)
            d_loss_fake = binary_cross_entropy_loss_fn(tf.zeros_like(d_fake),d_fake,sample_weight=fake_weights)
            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
            q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
            q_loss = tf.reduce_mean(q_loss)
            loss_total = d_loss_fake + q_loss
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        q_loss_d = q_loss
        
        with tf.GradientTape() as tape:
            d_real, _ = self.discriminator(X_real,batch_target_labels)
            d_loss_real = binary_cross_entropy_loss_fn(tf.ones_like(d_real),d_real,sample_weight=real_weights)
            loss_total = d_loss_real
        variables = self.discriminator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        d_optimizer.apply_gradients(zip(gradients, variables))
        
        c = tf.random.uniform(minval=self.bounds[0], maxval = self.bounds[1], shape=(batch_size, self.latent_dim))
        z = tf.random.normal(stddev=0.5,shape=(batch_size, self.noise_dim))
        
        with tf.GradientTape() as tape:
            x_fake_train, cp_train, w_train, _, db_train = self.generator(c, z, batch_target_labels)
            d_fake, q_fake_train = self.discriminator(x_fake_train, batch_target_labels)
            
            
            g_loss = binary_cross_entropy_loss_fn(tf.ones_like(d_fake),d_fake)
            
            r_w_loss = tf.reduce_mean(w_train[:,1:-1], axis=[1,2])
            cp_dist = tf.norm(cp_train[:,1:]-cp_train[:,:-1], axis=-1)
            r_cp_loss = tf.reduce_mean(cp_dist, axis=-1)
            ends = cp_train[:,0] - cp_train[:,-1]
            r_ends_loss = tf.norm(ends, axis=-1) + tf.math.maximum(0.0, -10*ends[:,1])
            r_loss = r_w_loss + r_cp_loss + r_ends_loss
            r_loss = tf.reduce_mean(r_loss)

            q_mean = q_fake_train[:, 0, :]
            q_logstd = q_fake_train[:, 1, :]
            epsilon = (c - q_mean) / (tf.math.exp(q_logstd) + self.EPSILON)
            q_loss = q_logstd + 0.5 * tf.math.square(epsilon)
            q_loss = tf.reduce_mean(q_loss)
            
            y = self.estimator(x_fake_train,training = False)[0]
            cond_score = lambert_w_log_exp_score(tf.math.abs(y-batch_target_labels),self.lambert_cutoff)
            
            dpp_loss = self.compute_diversity_loss(x_fake_train, tf.math.sigmoid(d_fake)*cond_score)[0]

            loss_total = g_loss + 10*r_loss + q_loss + lambda1 * dpp_loss
        
        variables = self.generator.trainable_weights
        gradients = tape.gradient(loss_total, variables)
        g_optimizer.apply_gradients(zip(gradients, variables))
        
        return d_loss_real, d_loss_fake, q_loss_d, g_loss, r_loss, q_loss, tf.reduce_mean(cond_score), dpp_loss
    
    def train(self, X, Y, train_steps=10000, batch_size=32, disc_lr=1e-4, gen_lr=1e-4):
        
        disc_lr = keras.optimizers.schedules.ExponentialDecay(disc_lr,decay_steps = 2*train_steps/10, decay_rate = 0.8, staircase=True)
        gen_lr = keras.optimizers.schedules.ExponentialDecay(gen_lr,decay_steps = train_steps/10, decay_rate = 0.8, staircase=True)
        
        g_optimizer = keras.optimizers.Adam(gen_lr,beta_1 = 0.5)

        d_optimizer = keras.optimizers.Adam(disc_lr,beta_1 = 0.5)
        
        steps = trange(train_steps, desc='Training', leave=True, ascii ="         =")
        for step in steps:
            X_real, Y_real, batch_fake_labels, batch_target_labels = self.get_batch(X, Y, batch_size)
            X_real, Y_real, batch_fake_labels, batch_target_labels = tf.cast(X_real, tf.float32), tf.cast(Y_real, tf.float32), tf.cast(batch_fake_labels, tf.float32), tf.cast(batch_target_labels, tf.float32)
            
            p = tf.constant(5.0,dtype=tf.float32)
            lambda1 = self.lambda1 * tf.cast((step/(train_steps-1))**p,tf.float32)
            lambda1 = tf.constant(lambda1,dtype=tf.float32)
            
            d_loss_real, d_loss_fake, q_loss_d, g_loss, r_loss, q_loss, cond_score, dpp_loss = self.train_step(X_real, Y_real, batch_fake_labels, batch_target_labels, d_optimizer,g_optimizer,lambda1)
            log_mesg = "%d: [D] real %+.7f fake %+.7f q %+.7f lr %+.7f" % (step+1, d_loss_real, d_loss_fake, q_loss_d, d_optimizer._decayed_lr('float32'))
            log_mesg = "%s  [G] fake %+.7f reg %+.7f q %+.7f lr %+.7f dpp %+.7f l1 %+.7f [C] cs %+.7f" % (log_mesg, g_loss, r_loss, q_loss, g_optimizer._decayed_lr('float32'),dpp_loss, lambda1, cond_score)
            
            steps.set_postfix_str(log_mesg)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange, tqdm
import tensorflow_probability as tfp
import os
import subprocess
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from matplotlib import cm
from sklearn.metrics import pairwise_distances
import matplotlib.animation as animation

def compute_diversity_loss(x): 
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.exp(-0.5 * tf.math.square(D))
        L = S

        eig_val, _ = tf.linalg.eigh(L)
        loss = -tf.reduce_mean(tf.math.log(tf.math.maximum(eig_val, 1e-7)))

        
        return loss, D, S
    
    
def lambert_w_log_exp_score(abserror,a=3.5):
    
    w = tfp.math.lambertw(-1/2/a)
    threshold = tf.math.exp(-a * tf.math.exp(w))
    s = tf.math.exp(-a * tf.math.exp(w))/tf.math.sqrt(-2*w)
    
    return tf.where(abserror>threshold,-tf.math.log(abserror)/a,tf.math.exp(-tf.math.square(abserror)/2/s**2))


def diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1)
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times

def hist_anim(ys,conds,save_location='./Evaluation/CcGAN_hist.mp4'):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=6800)
    def update_hist(n, data):
        plt.cla()
        plt.hist(np.squeeze(data[n]),np.array(list(range(21)))/20)
        plt.title("Conditioned On: %f" % conds[n])
        plt.xlabel('Condition')

    fig = plt.figure()
    hist = plt.hist(np.squeeze(ys[0]),np.array(list(range(21)))/20)
    plt.title("Conditioned On: %f" % conds[0])
    plt.xlabel('Condition')

    anim = animation.FuncAnimation(fig, update_hist, len(ys), fargs=(ys, ) )

    anim.save(save_location, writer=writer)


def data_plot(samples,equation,save_location):

    xlim = (-0.6,0.6)
    ylim = (-0.6,0.6)
    plt.rc('font', size=30)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                        np.linspace(ylim[0], ylim[1], 50))
    grid = np.vstack((xx.ravel(), yy.ravel())).T
    q = equation(grid)
    x_fake = samples
    fig = plt.figure(figsize=(14,10))
    contours = plt.contourf(np.linspace(xlim[0], xlim[1],50),np.linspace(ylim[0], ylim[1], 50),np.reshape(q,(xx.shape)),levels=20,cmap='Blues_r',alpha=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Performance Metric')
    x_fake = np.squeeze(x_fake)
    plt.scatter(x_fake[:,0],x_fake[:,1],color='#FFA600',marker='o', label='Data')
    plt.savefig(save_location,dpi=300, bbox_inches='tight')



def dist_anim(samples,conds,equation,save_location='./Evaluation/CcGAN_output.mp4'):

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=10000)

    xlim = (-0.6,0.6)
    ylim = (-0.6,0.6)
    plt.rc('font', size=30)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                        np.linspace(ylim[0], ylim[1], 50))
    grid = np.vstack((xx.ravel(), yy.ravel())).T
    q = equation(grid)
    x_fake = samples[0]

    def update_hist(n, data):
        plt.cla()
        contours = plt.contourf(np.linspace(xlim[0], xlim[1], 50),np.linspace(ylim[0], ylim[1], 50),np.reshape(q,(xx.shape)),levels=20,cmap='Blues_r',alpha=1)
        plt.title("PcDGAN Conditioned On: %f" % (conds[n]))
        x_fake = np.squeeze(data[n])
        plt.scatter(x_fake[:,0],x_fake[:,1],color='#FFA600',marker='o', label='Output samples')
        

    fig = plt.figure(figsize=(14,10))
    contours = plt.contourf(np.linspace(xlim[0], xlim[1], 50),np.linspace(ylim[0], ylim[1], 50),np.reshape(q,(xx.shape)),levels=20,cmap='Blues_r',alpha=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Performance Metric')
    plt.title("PcDGAN Conditioned On: %f" % (conds[0]))
    x_fake = np.squeeze(x_fake)
    plt.scatter(x_fake[:,0],x_fake[:,1],color='#FFA600',marker='o', label='Output samples')

    anim = animation.FuncAnimation(fig, update_hist, len(samples), fargs=(samples, ) )

    anim.save(save_location, writer=writer)
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tqdm.autonotebook import trange
import tensorflow_probability as tfp
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.spatial.distance import directed_hausdorff
from sklearn.decomposition import PCA

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
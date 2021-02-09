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

#Get OS Info For Simulation
if os.name == 'nt':
    current_os = 'win'
    xfoil_command = '.\\XFOIL_Windows\\xfoil.exe'
    new_ln = '\\r\\n'
else:
    current_os = 'osx'
    new_ln = '\\n'
    xfoil_command = 'xfoil'

#No Quality Diversity
def compute_diversity_loss(x):

        flatten = keras.layers.Flatten()    
        x = flatten(x)
        
        r = tf.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        S = tf.math.exp(-0.5*tf.square(D))
        

        eig_val, _ = tf.linalg.eigh(S)
        loss = -tf.reduce_mean(tf.math.log(tf.maximum(eig_val, 1e-7)))

        
        return loss, D, S
    

#LLETS Function
def lambert_w_log_exp_score(abserror,a=3.5):
    
    w = tfp.math.lambertw(-1/2/a)
    threshold = tf.math.exp(-a * tf.math.exp(w))
    s = tf.math.exp(-a * tf.math.exp(w))/tf.math.sqrt(-2*w)
    
    return tf.where(abserror>threshold,-tf.math.log(abserror)/a,tf.math.exp(-tf.math.square(abserror)/2/s**2))


#Fast XFOIL Communication
def simulate(file, re=1.8e6, mach=0.01, max_iter=200, alpha=0.0, verbose=False):
    ps = subprocess.Popen([xfoil_command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:
        results = ps.communicate(bytes('PLOP\nG F\n\n\nload ' + file + '\naf\nOPER\nVISC ' + str(re) + '\nMACH ' + str(mach) + '\nITER ' + str(max_iter) + '\nPACC\n\n\na ' + str(alpha) + '\n','utf-8'), timeout = 2)
    except subprocess.TimeoutExpired:
        ps.kill()
        if verbose:
            print('Failed')
        return np.nan,np.nan,np.nan
    conv_check = ' '.join(str(results[0]).split('\\n')[-10:])
    if conv_check.find('Convergence failed') != -1:
        ps.kill()
        if verbose:
            print('Failed')
        return np.nan,np.nan,np.nan
    else:
        try:
            
            CD = float(str(results[0])[str(results[0]).rfind('CD ')+5:].split('=')[0])
            CL = float(str(results[0])[str(results[0]).rfind('CL ')+5:].split(new_ln)[0])
            LDR = CL/CD
            ps.kill()
            if verbose:
                print('CD = %f | CL = %f | L/D = %f' % (CD, CL, LDR))
            return CD, CL, LDR
        except:
            ps.kill()
            if verbose:
                print('Failed')
            return np.nan,np.nan,np.nan
    
#Parallel Processing for Faster Simulations
def run_imap_multiprocessing(func, argument_list):

    pool = mp.Pool(processes=mp.cpu_count())

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list),position=0, leave=True):
        result_list_tqdm.append(result)

    return result_list_tqdm

#Mainstream Simulation Process
def batch_simulate(Airfoils, txt_folder):
    files = []
    print('Generating Text Airfoil Files For Simulation ...')
    for i in trange(Airfoils.shape[0],position=0, leave=True):
        np.savetxt(txt_folder + str(i) + '.dat',np.squeeze(Airfoils[i]))
        files.append(txt_folder + str(i) + '.dat')
    
    print('Simulating Samples')
    LDRs = run_imap_multiprocessing(simulate,files)
    LDRs = np.squeeze(np.array(LDRs)[:,2])
    
    print('Removing Text Airfoil Files ...')
    for i in trange(Airfoils.shape[0],position=0, leave=True):
        os.remove(txt_folder + str(i) + '.dat')
    
    return LDRs

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
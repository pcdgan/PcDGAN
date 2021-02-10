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


def plot_airfoil_samples(PcDGAN_samples,CcGAN_samples,y_PcD,y_Cc, conds, n_plot = 5, save_path = './Evaluation/Sample Airfoils With Error.png'):
    
    n_conds = conds.shape[0]
    
    fig, axs = plt.subplots(n_plot,n_conds*2,figsize=(64,16))

    batch_size = CcGAN_samples.shape[0]//n_conds

    for i in range(n_conds):
        Cc_airfoils = CcGAN_samples[i*batch_size:(i+1)*batch_size]
        PcD_airfoils = PcDGAN_samples[i*batch_size:(i+1)*batch_size]
        cond = conds[i]
        Cc_ys = y_Cc[i*batch_size:(i+1)*batch_size]
        PcD_ys = y_PcD[i*batch_size:(i+1)*batch_size]
        inds_PcD = np.argsort(np.abs(PcD_ys[0:n_plot]-conds[i]),0)
        inds_Cc = np.argsort(np.abs(Cc_ys[0:n_plot]-conds[i]),0)

        for k in range(n_plot):
            j = inds_PcD[k,0]
            axs[k,i*2].plot(PcD_airfoils[j,:,0,0],PcD_airfoils[j,:,1,0],color="#003F5C",linewidth=4)
            axs[k,i*2].axis('equal')
            axs[k,i*2].set_axis_off()
            axs[k,i*2].text(0.3,-0.20,r"%.2f" % (np.abs(PcD_ys[j]-conds[i])),fontsize=70*3/n_plot)
            j = inds_Cc[k,0]
            axs[k,i*2+1].plot(Cc_airfoils[j,:,0,0],Cc_airfoils[j,:,1,0],color="#FFA600",linewidth=4)
            axs[k,i*2+1].axis('equal')
            axs[k,i*2+1].set_axis_off()
            axs[k,i*2+1].text(0.3,-0.20,r"%.2f" % (np.abs(Cc_ys[j]-conds[i])),fontsize=70*3/n_plot)
            
        axs[0,2*i].text(0.45,0.2,'Input Label: %.2f' % (conds[i]),fontsize=80 * 3/n_plot)
    axs[-1,-1].legend(['CcGAN'],bbox_to_anchor=(-3 * n_conds/4, -0.5),fancybox=False, shadow=False, framealpha=0.0, ncol=2,fontsize=70*3/n_plot)
    axs[-1,-2].legend(['PcDGAN'],bbox_to_anchor=(-3 * n_conds/4, -0.5),fancybox=False, shadow=False, framealpha=0.0, ncol=2,fontsize=70*3/n_plot)

    plt.savefig(save_path,dpi=300, bbox_inches='tight')

def plot_airfoils(airfoils, ys, zs, ax, norm, cmap, zs_data=None):
        n = airfoils.shape[0]
        ys[np.isnan(ys)] = 0.
        for i in range(n):
            plot_shape(airfoils[i]+np.array([[-.5,0]]), zs[i, 0], zs[i, 1], ax, 1./n**.5, False, None, lw=1.2, alpha=.7, c=cmap(norm(ys[i])))
        if zs_data is not None:
            ax.scatter(zs_data[:,0], zs_data[:,1], s=100, marker='o', edgecolors='none', c='#FFA600')
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False,
            labelbottom=False)
        ax.set_xlim([-.01-.5/n**.5, 1.01+.5/n**.5])
        ax.set_ylim([-.05, 1.05])
        ax.set_aspect('equal')
        
def plot_shape(xys, z1, z2, ax, scale, scatter, symm_axis, **kwargs):
#    mx = max([y for (x, y) in m])
#    mn = min([y for (x, y) in m])
    xscl = scale# / (mx - mn)
    yscl = scale# / (mx - mn)
#    ax.scatter(z1, z2)
    if scatter:
        if 'c' not in kwargs:
            kwargs['c'] = cm.rainbow(np.linspace(0,1,xys.shape[0]))
#        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        ax.scatter( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), edgecolors='none', **kwargs)
    else:
        ax.plot( *zip(*[(x * xscl + z1, y * yscl + z2) for (x, y) in xys]), **kwargs)
        
    if symm_axis == 'y':
#        ax.plot( *zip(*[(-x * xscl + z1, y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_betweenx( *zip(*[(y * yscl + z2, -x * xscl + z1, x * xscl + z1)
                          for (x, y) in xys]), color='gray', alpha=.2)
    elif symm_axis == 'x':
#        ax.plot( *zip(*[(x * xscl + z1, -y * yscl + z2) for (x, y) in xys]), lw=.2, c='b')
        plt.fill_between( *zip(*[(x * xscl + z1, -y * yscl + z2, y * yscl + z2)
                          for (x, y) in xys]), color='gray', alpha=.2)

def hist_anim(ys,conds,save_location='./Evaluation/CcGAN_hist.mp4'):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, bitrate=6800)
    def update_hist(n, data):
        plt.cla()
        plt.hist(np.squeeze(data[n]),np.array(list(range(21)))/20)
        plt.title("Conditioned On: %f" % conds[n])

    fig = plt.figure()
    hist = plt.hist(np.squeeze(ys[0]),np.array(list(range(21)))/20)
    plt.title("Conditioned On: %f" % conds[0])


    anim = animation.FuncAnimation(fig, update_hist, len(ys), fargs=(ys, ) )

    anim.save(save_location, writer=writer)

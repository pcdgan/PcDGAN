import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
from GANs import CcGAN, PcDGAN
from utils import compute_diversity_loss, diversity_score
import matplotlib.animation as animation
from glob import glob
import argparse
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from data import Donut2D, ThinDonut2D, Uniform2D, MixRing6

parser = argparse.ArgumentParser(description='Eval Parameters')
parser.add_argument('dataset', type=str, default='Uneven', help='Set which dataset to use. Default: Uneven, Options: Uneven, Donut, Uniform')
parser.add_argument('--estimator', type=str, help='Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--embedder', type=str, help='Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')
args = parser.parse_args()
folder = args.dataset
if __name__ == "__main__":
    print("\nLoading Data ...\n")
    N = 10000
    func_obj = MixRing6()
    if args.dataset == 'Uneven':
        N = 1000000
        data_obj = Uniform2D(N,-0.7,0.7)
        X = data_obj.data
        X = np.concatenate([X[np.linalg.norm(X-func_obj.centers[1],axis=1)<=0.2,:][0:5000],X[np.linalg.norm(X-func_obj.centers[1],axis=1)>=0.2,:][0:5000]],0)
        data_obj.data = X
    elif args.dataset == 'Donut':
        data_obj = Donut2D(N)
    else:
        data_obj = Uniform2D(N,-0.6,0.6)
    
    ind = np.random.choice(data_obj.data.shape[0], size=1000, replace=False)
    X = data_obj.data
    Y = np.expand_dims(func_obj.equation(X),-1)
    miny = np.min(Y)
    maxy = np.max(Y)
    rangey = maxy - miny
    Y = (Y-miny)/rangey
    equation = lambda x: (func_obj.equation(x)-miny)/rangey

    model_CcGAN = CcGAN(Y=Y)
    model_PcDGAN = PcDGAN(Y=Y)

    CcGAN_paths = glob('./'+folder+'/Weights/Generator/CcGAN/*.index')
    PcDGAN_paths = glob('./'+folder+'/Weights/Generator/PcDGAN/*.index')

    print('\nFound %i CcGAN Checkpoints' % (len(CcGAN_paths)))
    print('Found %i PcDGAN Checkpoints\n' % (len(PcDGAN_paths)))

    def get_generator(generator, estimator):

        @tf.function
        @tf.autograph.experimental.do_not_convert
        def generate_samples(condition, size=1000):
            z = tf.random.normal(stddev=0.5,shape=(batch_size, 2))
            samples = generator(z, condition, training=False)
            y = tf.expand_dims(estimator(samples),-1)
            good_samples = tf.gather(samples,tf.where(tf.math.abs(y-condition)<0.05)[:,0])
            return samples, y, good_samples 
        return generate_samples

    print("\nEvaluating CcGAN Checkpoints ...\n")
    divers = []
    KDEs = []
    MAEs = []
    CcGAN_sim_samples = []
    batch_size = 500
    progress_Cc = trange(1000*len(CcGAN_paths),position=0, leave=True)
    for CcGAN_path in CcGAN_paths:
        model_CcGAN.generator.load_weights(CcGAN_path[0:-6])
        sample_generator = get_generator(model_CcGAN.generator,equation)
        conds = np.linspace(0.05,0.95,100)
        for j in range(10):
            diversity = []
            KDE = []
            ys = []
            MAE = []
            for i in range(100):
                samples, y, good_samples = sample_generator(conds[i] * tf.ones([batch_size,1]),batch_size)
                ind = np.random.choice(samples.shape[0],size=100,replace=False)
                diversity.append(diversity_score(samples.numpy()))
                bandwidth = np.arange(0.01, 0.1, .01)
                kde = KernelDensity(kernel='gaussian')
                grid = GridSearchCV(kde, {'bandwidth': bandwidth})
                grid.fit(y.numpy())
                kde = grid.best_estimator_
                KDE.append(np.exp(kde.score_samples([[conds[i]]])[0]))
                ys.append(y)
                MAE.append(np.mean(np.abs(y-conds[i])))
                progress_Cc.update(1)
            divers.append(diversity)
            KDEs.append(KDE)
            MAEs.append(MAE)

    CcGAN_KDE = np.nanmean(KDEs,0)
    CcGAN_KDE_overall = np.nanmean(KDEs)
    CcGAN_diver = np.nanmean(divers,0)
    CcGAN_diver_overall = np.nanmean(divers)
    CcGAN_MAE = np.nanmean(MAEs,0)
    CcGAN_MAE_overall = np.nanmean(MAEs)
    CcGAN_KDE_std = np.nanstd(KDEs,0)
    CcGAN_KDE_std_overall = np.nanstd(KDEs)
    CcGAN_diver_std = np.nanstd(divers,0)
    CcGAN_diver_std_overall = np.nanstd(divers)
    CcGAN_MAE_std = np.nanstd(MAEs,0)
    CcGAN_MAE_std_overall = np.nanstd(MAE)

    print("\nEvaluating PcDGAN Checkpoints ...\n")
    divers = []
    KDEs = []
    MAEs = []
    PcDGAN_sim_samples = []
    progress_PcD = trange(1000*len(PcDGAN_paths),position=0, leave=True)
    for PcDGAN_path in PcDGAN_paths:
        model_PcDGAN.generator.load_weights(PcDGAN_path[0:-6])
        sample_generator = get_generator(model_PcDGAN.generator,equation)
        conds = np.linspace(0.05,0.95,100)
        for j in range(10):
            diversity = []
            KDE = []
            ys = []
            MAE = []
            for i in range(100):
                samples, y, good_samples = sample_generator(conds[i] * tf.ones([batch_size,1]),batch_size)
                #make sure there is enough samples for diversity compute
                ind = np.random.choice(samples.shape[0],size=100,replace=False)
                diversity.append(diversity_score(samples.numpy()))
                bandwidth = np.arange(0.01, 0.1, .01)
                kde = KernelDensity(kernel='gaussian')
                grid = GridSearchCV(kde, {'bandwidth': bandwidth})
                grid.fit(y.numpy())
                kde = grid.best_estimator_
                KDE.append(np.exp(kde.score_samples([[conds[i]]])[0]))
                MAE.append(np.mean(np.abs(y-conds[i])))
                progress_PcD.update(1)
            divers.append(diversity)
            KDEs.append(KDE)
            MAEs.append(MAE)
    PcDGAN_KDE = np.nanmean(KDEs,0)
    PcDGAN_KDE_overall = np.nanmean(KDEs)
    PcDGAN_diver = np.nanmean(divers,0)
    PcDGAN_diver_overall = np.nanmean(divers)
    PcDGAN_MAE = np.nanmean(MAEs,0)
    PcDGAN_MAE_overall = np.nanmean(MAEs)
    PcDGAN_KDE_std = np.nanstd(KDEs,0)
    PcDGAN_KDE_std_overall = np.nanstd(KDEs)
    PcDGAN_diver_std = np.nanstd(divers,0)
    PcDGAN_diver_std_overall = np.nanstd(divers)
    PcDGAN_MAE_std = np.nanstd(MAEs,0)
    PcDGAN_MAE_std_overall = np.nanstd(MAE)


    plt.figure(figsize=(18,12))
    plt.rc('font', size=45)
    plt.plot(conds,PcDGAN_diver,color='#003F5C')
    plt.fill_between(conds,PcDGAN_diver-PcDGAN_diver_std,PcDGAN_diver+PcDGAN_diver_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
    plt.plot(conds,CcGAN_diver,color='#FFA600')
    plt.fill_between(conds,CcGAN_diver-CcGAN_diver_std,CcGAN_diver+CcGAN_diver_std,facecolor='#FFA600',edgecolor="#FFA600",alpha=0.3)
    plt.legend(['PcDGAN','CcGAN'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=2)

    plt.title('Diversity vs Condition')
    plt.xlabel('Condition')
    plt.ylabel('Diversity')
    plt.savefig('./'+folder+'/Evaluation/Diversity Comparison.png',dpi=300, bbox_inches='tight')

    plt.figure(figsize=(18,12))
    plt.rc('font', size=45)
    plt.plot(conds,PcDGAN_KDE,color='#003F5C')
    plt.fill_between(conds,PcDGAN_KDE-PcDGAN_KDE_std,PcDGAN_KDE+PcDGAN_KDE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
    plt.plot(conds,CcGAN_KDE,color='#FFA600')
    plt.fill_between(conds,CcGAN_KDE-CcGAN_KDE_std,CcGAN_KDE+CcGAN_KDE_std,facecolor='#FFA600',edgecolor="#FFA600",alpha=0.3)
    plt.legend(['PcDGAN','CcGAN'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=2)

    plt.title('KDE vs Condition')
    plt.xlabel('Condition')
    plt.ylabel('KDE of Condition Label in Output Distribution')
    plt.savefig('./'+folder+'/Evaluation/Probability Comparison.png',dpi=300, bbox_inches='tight')

    plt.figure(figsize=(18,12))
    plt.rc('font', size=45)
    plt.plot(conds,PcDGAN_MAE,color='#003F5C')
    plt.fill_between(conds,PcDGAN_MAE-PcDGAN_MAE_std,PcDGAN_MAE+PcDGAN_MAE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
    plt.plot(conds,CcGAN_MAE,color='#FFA600')
    plt.fill_between(conds,CcGAN_MAE-CcGAN_MAE_std,CcGAN_MAE+CcGAN_MAE_std,facecolor='#FFA600',edgecolor="#FFA600",alpha=0.3)
    plt.legend(['PcDGAN','CcGAN'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=2)

    plt.title('Label Score vs Condition')
    plt.xlabel('Condition')
    plt.ylabel('Label Score')
    plt.savefig('./'+folder+'/Evaluation/MAE Comparison.png',dpi=300, bbox_inches='tight')

    print("\nSummary of Performance:\n")
    print(tabulate([['CcGAN', str(CcGAN_MAE_overall) + '+/-' + str(CcGAN_MAE_std_overall),str(CcGAN_KDE_overall) + '+/-' + str(CcGAN_KDE_std_overall),str(CcGAN_diver_overall) + '+/-' + str(CcGAN_diver_std_overall)], ['PcDGAN',  str(PcDGAN_MAE_overall) + '+/-' + str(PcDGAN_MAE_std_overall),str(PcDGAN_KDE_overall) + '+/-' + str(PcDGAN_KDE_std_overall),str(PcDGAN_diver_overall) + '+/-' + str(PcDGAN_diver_std_overall)]], headers=['Model', 'Label Score', 'Probability Density', 'Diversity']))
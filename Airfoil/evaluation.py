import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
from GANs import CcGAN, PcDGAN
from utils import compute_diversity_loss, batch_simulate, diversity_score
import matplotlib.animation as animation
from glob import glob
import argparse
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Eval Parameters')
parser.add_argument('--estimator', type=str, help='Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--embedder', type=str, help='Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--simonly', type=int, help='Only evaluate based on simluation. Default: 0(False). Set to 1 for True', default=0)
args = parser.parse_args()

if __name__ == "__main__":
    print("\nLoading Data ...\n")
    X_train = np.load('./data/xs_train.npy').astype('float32')
    X_test = np.load('./data/xs_test.npy').astype('float32')
    Y_train = np.load('./data/ys_train.npy').astype('float32')
    Y_test = np.load('./data/ys_test.npy').astype('float32')
    Y = np.concatenate((Y_train,Y_test))
    min_y = np.min(Y)
    max_y = np.max(Y)
    Y_train = (Y_train - min_y)/(max_y - min_y)
    Y_test = (Y_test - min_y)/(max_y - min_y)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)
    X = np.concatenate([X_train,X_test])
    Y = np.concatenate([Y_train,Y_test])

    model_CcGAN = CcGAN(Y=Y)
    model_PcDGAN = PcDGAN(Y=Y)

    model_CcGAN.estimator.load_weights('./Weights/Estimator/' + args.estimator)
    model_PcDGAN.estimator.load_weights('./Weights/Estimator/' + args.estimator)
    model_CcGAN.embedder.load_weights('./Weights/Embedder/' + args.embedder)

    CcGAN_paths = glob('./Weights/Generator/CcGAN/*.index')
    PcDGAN_paths = glob('./Weights/Generator/PcDGAN/*.index')

    print('\nFound %i CcGAN Checkpoints' % (len(CcGAN_paths)))
    print('Found %i PcDGAN Checkpoints\n' % (len(PcDGAN_paths)))

    def get_generator(generator, estimator,embedder=lambda x: x):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def generate_samples(condition, size=1000):
            c = tf.random.uniform(minval=0.0, maxval=1.0, shape=(size, 5))
            z = tf.random.normal(stddev=0.5,shape=(size, 10))
            samples = generator(c,z, embedder(condition), training=False)[0]
            y = estimator(samples,training=False)[0]
            good_samples = tf.gather(samples,tf.where(tf.math.abs(y-condition)<0.05)[:,0])
            return samples, y, good_samples 
        return generate_samples

    print("\nEvaluating CcGAN Checkpoints ...\n")
    divers = []
    KDEs = []
    MAEs = []
    CcGAN_sim_samples = []
    batch_size = 500
    if args.simonly == 0:
        progress_Cc = trange(1000*len(CcGAN_paths),position=0, leave=True)
    else:
        progress_Cc = trange(len(CcGAN_paths),position=0, leave=True)
    
    for CcGAN_path in CcGAN_paths:
        model_CcGAN.generator.load_weights(CcGAN_path[0:-6])
        sample_generator = get_generator(model_CcGAN.generator,model_CcGAN.estimator,model_CcGAN.embedder)
        if args.simonly == 0:
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
        sim_conditions = tf.cast(np.expand_dims(np.concatenate(np.tile(np.expand_dims(np.linspace(0.05,0.95,20),-1),[1,100]),0),-1),tf.float32)
        CcGAN_sim_samples.append(sample_generator(sim_conditions,2000)[0])
        if args.simonly == 1:
            progress_Cc.update(1)
    
    if args.simonly == 0:
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
    if args.simonly == 0:
        progress_PcD = trange(1000*len(PcDGAN_paths),position=0, leave=True)
    else:
        progress_PcD = trange(len(PcDGAN_paths),position=0, leave=True)

    for PcDGAN_path in PcDGAN_paths:
        model_PcDGAN.generator.load_weights(PcDGAN_path[0:-6])
        sample_generator = get_generator(model_PcDGAN.generator,model_PcDGAN.estimator)
        if args.simonly == 0:
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
        sim_conditions = tf.cast(np.expand_dims(np.concatenate(np.tile(np.expand_dims(np.linspace(0.05,0.95,20),-1),[1,100]),0),-1),tf.float32)
        PcDGAN_sim_samples.append(sample_generator(sim_conditions,2000)[0])
        if args.simonly == 1:
            progress_PcD.update(1)

    if args.simonly == 0:
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
        plt.savefig('./Evaluation/Diversity Comparison.png',dpi=300, bbox_inches='tight')

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
        plt.savefig('./Evaluation/Probability Comparison.png',dpi=300, bbox_inches='tight')

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
        plt.savefig('./Evaluation/MAE Comparison.png',dpi=300, bbox_inches='tight')

        print("\nSummary of Performance:\n")
        print(tabulate([['CcGAN', str(CcGAN_MAE_overall) + '+/-' + str(CcGAN_MAE_std_overall),str(CcGAN_KDE_overall) + '+/-' + str(CcGAN_KDE_std_overall),str(CcGAN_diver_overall) + '+/-' + str(CcGAN_diver_std_overall)], ['PcDGAN',  str(PcDGAN_MAE_overall) + '+/-' + str(PcDGAN_MAE_std_overall),str(PcDGAN_KDE_overall) + '+/-' + str(PcDGAN_KDE_std_overall),str(PcDGAN_diver_overall) + '+/-' + str(PcDGAN_diver_std_overall)]], headers=['Model', 'Label Score', 'Probability Density', 'Diversity']))
        
    print("\nSimulating CcGAN Airfoils ...\n")
    sim_conds = np.linspace(0.05,0.95,20)
    KDEs = []
    MAEs = []
    for sim_samples in CcGAN_sim_samples:
        y = batch_simulate(sim_samples,'./Samples/CcGAN/txts/')
        KDE = []
        MAE = []
        for i in range(20):
            ry = (y[i*100:(i+1)*100] - min_y)/(max_y-min_y)
            ry = ry[~np.isnan(ry)]
            kde_y = np.expand_dims(ry,-1)
            bandwidth = np.arange(0.01, 0.1, .01)
            kde = KernelDensity(kernel='gaussian')
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(kde_y)
            kde = grid.best_estimator_
            KDE.append(np.exp(kde.score_samples([[sim_conds[i]]])[0]))
            MAE.append(np.mean(np.abs(ry-sim_conds[i])))
        KDEs.append(KDE)
        MAEs.append(MAE)

    CcGAN_KDE = np.nanmean(KDEs,0)
    CcGAN_KDE_overall = np.nanmean(KDEs)
    CcGAN_MAE = np.nanmean(MAEs,0)
    CcGAN_MAE_overall = np.nanmean(MAEs)
    CcGAN_KDE_std = np.nanstd(KDEs,0)
    CcGAN_KDE_std_overall = np.nanstd(KDEs)
    CcGAN_MAE_std = np.nanstd(MAEs,0)
    CcGAN_MAE_std_overall = np.nanstd(MAE)

    print("\nSimulating PcDGAN Airfoils ...\n")
    KDEs = []
    MAEs = []
    for sim_samples in PcDGAN_sim_samples:
        y = batch_simulate(sim_samples,'./Samples/PcDGAN/txts/')
        KDE = []
        MAE = []
        for i in range(20):
            ry = (y[i*100:(i+1)*100] - min_y)/(max_y-min_y)
            ry = ry[~np.isnan(ry)]
            kde_y = np.expand_dims(ry,-1)
            bandwidth = np.arange(0.01, 0.1, .01)
            kde = KernelDensity(kernel='gaussian')
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(kde_y)
            kde = grid.best_estimator_
            KDE.append(np.exp(kde.score_samples([[sim_conds[i]]])[0]))
            MAE.append(np.mean(np.abs(ry-sim_conds[i])))
        KDEs.append(KDE)
        MAEs.append(MAE)
    
    PcDGAN_KDE = np.nanmean(KDEs,0)
    PcDGAN_KDE_overall = np.nanmean(KDEs)
    PcDGAN_MAE = np.nanmean(MAEs,0)
    PcDGAN_MAE_overall = np.nanmean(MAEs)
    PcDGAN_KDE_std = np.nanstd(KDEs,0)
    PcDGAN_KDE_std_overall = np.nanstd(KDEs)
    PcDGAN_MAE_std = np.nanstd(MAEs,0)
    PcDGAN_MAE_std_overall = np.nanstd(MAE)


    plt.figure(figsize=(18,12))
    plt.rc('font', size=45)
    plt.plot(sim_conds,PcDGAN_KDE,color='#003F5C')
    plt.fill_between(sim_conds,PcDGAN_KDE-PcDGAN_KDE_std,PcDGAN_KDE+PcDGAN_KDE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
    plt.plot(sim_conds,CcGAN_KDE,color='#FFA600')
    plt.fill_between(sim_conds,CcGAN_KDE-CcGAN_KDE_std,CcGAN_KDE+CcGAN_KDE_std,facecolor='#FFA600',edgecolor="#FFA600",alpha=0.3)
    plt.legend(['PcDGAN','CcGAN'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=4)

    plt.title('KDE vs Condition')
    plt.xlabel('Condition')
    plt.ylabel('KDE of Condition Label in Output Distribution')
    plt.savefig('./Evaluation/Probability Comparison(Physics Simulation).png',dpi=300, bbox_inches='tight')

    plt.figure(figsize=(18,12))
    plt.rc('font', size=45)
    plt.plot(sim_conds,PcDGAN_MAE,color='#003F5C')
    plt.fill_between(sim_conds,PcDGAN_MAE-PcDGAN_MAE_std,PcDGAN_MAE+PcDGAN_MAE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
    plt.plot(sim_conds,CcGAN_MAE,color='#FFA600')
    plt.fill_between(sim_conds,CcGAN_MAE-CcGAN_MAE_std,CcGAN_MAE+CcGAN_MAE_std,facecolor='#FFA600',edgecolor="#FFA600",alpha=0.3)
    plt.legend(['PcDGAN','CcGAN'],loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False, framealpha=0.0, ncol=4)

    plt.title('Label Score vs Condition')
    plt.xlabel('Condition')
    plt.ylabel('Label Score')
    plt.savefig('./Evaluation/MAE Comparison(Physics Simulation).png',dpi=300, bbox_inches='tight')

    print("\nSummary of Performance Based on Simulation:\n")
    print(tabulate([['CcGAN', str(CcGAN_MAE_overall) + '+/-' + str(CcGAN_MAE_std_overall),str(CcGAN_KDE_overall) + '+/-' + str(CcGAN_KDE_std_overall)], ['PcDGAN',  str(PcDGAN_MAE_overall) + '+/-' + str(PcDGAN_MAE_std_overall),str(PcDGAN_KDE_overall) + '+/-' + str(PcDGAN_KDE_std_overall)]], headers=['Model', 'Label Score', 'Probability Density']))
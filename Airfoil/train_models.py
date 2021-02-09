import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from glob import glob
import tensorflow as tf
import numpy as np
from tqdm.autonotebook import trange
from utils import compute_diversity_loss, diversity_score
import matplotlib.pyplot as plt
import pprint
from utils import batch_simulate
from GANs import CcGAN,PcDGAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate


parser = argparse.ArgumentParser(description='Train Parameters')
parser.add_argument('model', type=str, default='PcDGAN', help='Set which model architecture you intend to train. Default: PcDGAN, Options: CcGAN, PcDGAN')
parser.add_argument('--mode', type=str, default='train', help='Mode of operation, either train or evaluate. Default: Train')
parser.add_argument('--vicinal_type', type=str, help='The type of vicinal approach. Default: soft, Options: soft, hard', default='soft')
parser.add_argument('--kappa', type=float, help='Vicinal loss kappa. If negative automatically calculated and scaled by the absolute value of the number. Default: -1.0 for PcDGAN -2.0 for CcGAN', default=None)
parser.add_argument('--sigma', type=float, help='Vicinal loss sigma. If negative automatically calculated. Default: -1.0', default=-1.0)

parser.add_argument('--estimator', type=str, help='Name of the estimator checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--embedder', type=str, help='Name of the embedder checkpoint saved in the weights folder. Default: best_checkpoint', default='best_checkpoint')

parser.add_argument('--lambda0', type=float, default=3.0, help='PcDGAN lambda0. Default: 3.0')
parser.add_argument('--lambda1', type=float, default=0.4, help='PcDGAN lambda1. Default: 0.4')
parser.add_argument('--lambert_cutoff', type=float, default=4.7, help='PcDGAN parameter "a". Default: 4.7')

parser.add_argument('--gen_lr', type=float, default=1e-4, help='Generator learning rate. Default: 1e-4')
parser.add_argument('--disc_lr', type=float, default=1e-4, help='Discriminator learning rate. Default: 1e-4')
parser.add_argument('--train_steps', type=int, default=20000, help='Number of training steps. Default: 20000')
parser.add_argument('--batch_size', type=int, default=32, help='GAN training Batch size. Default: 32')

parser.add_argument('--id', type=str, default='', help='experiment ID or name. Default: ')
args = parser.parse_args()

if __name__ == "__main__":
    #check directories
    if not os.path.exists('./Weights'):
        os.mkdir('./Weights')

    if not os.path.exists('./Samples'):
        os.mkdir('./Samples')

    if not os.path.exists('./Evaluation'):
        os.mkdir('./Evaluation')

    if not os.path.exists('./Weights/Discriminator'):
        os.mkdir('./Weights/Discriminator')

    if not os.path.exists('./Weights/Generator'):
        os.mkdir('./Weights/Generator')

    if not os.path.exists('./Weights/Estimator'):
        os.mkdir('./Weights/Estimator')

    if not os.path.exists('./Weights/Embedder'):
        os.mkdir('./Weights/Embedder')

    if not os.path.exists('./Weights/Discriminator/CcGAN'):
        os.mkdir('./Weights/Discriminator/CcGAN')

    if not os.path.exists('./Weights/Discriminator/PcDGAN'):
        os.mkdir('./Weights/Discriminator/PcDGAN')

    if not os.path.exists('./Weights/Generator/CcGAN'):
        os.mkdir('./Weights/Generator/CcGAN')

    if not os.path.exists('./Weights/Generator/PcDGAN'):
        os.mkdir('./Weights/Generator/PcDGAN')

    if not os.path.exists('./Samples/CcGAN'):
        os.mkdir('./Samples/CcGAN')

    if not os.path.exists('./Samples/PcDGAN'):
        os.mkdir('./Samples/PcDGAN')

    if not os.path.exists('./Samples/CcGAN/txts'):
        os.mkdir('./Samples/CcGAN/txts')

    if not os.path.exists('./Samples/PcDGAN/txts'):
        os.mkdir('./Samples/PcDGAN/txts')

    if not os.path.exists('./Evaluation/CcGAN'):
        os.mkdir('./Evaluation/CcGAN')

    if not os.path.exists('./Evaluation/PcDGAN'):
        os.mkdir('./Evaluation/PcDGAN')

    if len(glob('./data/*.npy')):
        print('Loading Airfoil Data...')
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

    else:
        raise Exception('could not find data. Please download and place the data in the data folder.')

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

    if args.model == 'PcDGAN':
        if args.kappa == None:
            args.kappa = -1.0
        if args.disc_lr == None:
            args.disc_lr = 5e-5
        model = PcDGAN(lambda0=args.lambda0, lambda1=args.lambda1, kappa=args.kappa, sigma=args.sigma, lambert_cutoff=args.lambert_cutoff, strategy=args.vicinal_type,Y=Y)

        if os.path.exists('./Weights/Estimator/'+args.estimator+'.index'):
            model.estimator.load_weights('./Weights/Estimator/'+args.estimator)
        else:
            raise Exception('could not find estimator weights.')

        print('Current Settings:')
        pprint.pprint(vars(args))

        if args.mode == 'train':
            flag = True
            while flag:
                flag = False
                try:
                    model.train(X,Y,train_steps=args.train_steps,batch_size=args.batch_size, disc_lr=args.disc_lr, gen_lr=args.gen_lr)
                except:
                    print("\nCUDA error encountered. Restarting training ...\n")
                    flag = True
                    model = PcDGAN(lambda0=args.lambda0, lambda1=args.lambda1, kappa=args.kappa, sigma=args.sigma, lambert_cutoff=args.lambert_cutoff, strategy=args.vicinal_type,Y=Y)
            model.discriminator.save_weights('./Weights/Discriminator/PcDGAN/experiment'+args.id)
            model.generator.save_weights('./Weights/Generator/PcDGAN/experiment'+args.id)
        else:
            model.discriminator.load_weights('./Weights/Discriminator/PcDGAN/experiment'+args.id)
            model.generator.load_weights('./Weights/Generator/PcDGAN/experiment'+args.id)
        sample_generator = get_generator(model.generator,model.estimator)
        divers = []
        KDEs = []
        MAEs = []
        batch_size = 500
        progress = trange(1000,position=0, leave=True)
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
                progress.update(1)
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
        
        print("\nSummary of Performance:\n")
        print(tabulate([['PcDGAN',  str(PcDGAN_MAE_overall) + '+/-' + str(PcDGAN_MAE_std_overall),str(PcDGAN_KDE_overall) + '+/-' + str(PcDGAN_KDE_std_overall),str(PcDGAN_diver_overall) + '+/-' + str(PcDGAN_diver_std_overall)]], headers=['Model', 'Label Score', 'Probability Density', 'Diversity']))

        print('Generating Sample Airfoils ...')
        conds = np.linspace(0.05,0.95,20)
        batch_size = 100
        conditions = []
        samples = []
        for i in trange(20):
            samples_, _, _ = sample_generator(conds[i] * tf.ones([batch_size,1]),batch_size)
            conditions.append(conds[i] * tf.ones([batch_size,1]))
            samples.append(samples_)
        samples = np.concatenate(samples,0)
        conditions = np.concatenate(conditions,0)
        np.save('./Samples/PcDGAN/airfoils_' + str(args.id) + '.npy', samples)

        conds = np.linspace(0.05,0.95,100)

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,PcDGAN_diver,color='#003F5C')
        plt.fill_between(conds,PcDGAN_diver-PcDGAN_diver_std,PcDGAN_diver+PcDGAN_diver_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('Diversity vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Diversity')
        plt.savefig('./Evaluation/PcDGAN/Diversity_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,PcDGAN_KDE,color='#003F5C')
        plt.fill_between(conds,PcDGAN_KDE-PcDGAN_KDE_std,PcDGAN_KDE+PcDGAN_KDE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('KDE vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('KDE of Condition Label in Output Distribution')
        plt.savefig('./Evaluation/PcDGAN/KDE_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,PcDGAN_MAE,color='#003F5C')
        plt.fill_between(conds,PcDGAN_MAE-PcDGAN_MAE_std,PcDGAN_MAE+PcDGAN_MAE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('Label Score vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Label Score')
        plt.savefig('./Evaluation/PcDGAN/Label Score_' + str(args.id) + '.png')

        LDRs = batch_simulate(samples,'./Samples/PcDGAN/txts/')
        sim_conds = np.linspace(0.05,0.95,20)
        KDE = []
        MAE = []
        for i in range(20):
            ry = (LDRs[i*100:(i+1)*100] - min_y)/(max_y-min_y)
            ry = ry[~np.isnan(ry)]
            bandwidth = np.arange(0.01, 0.1, .01)
            kde = KernelDensity(kernel='gaussian')
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(y.numpy())
            kde = grid.best_estimator_
            KDE.append(np.exp(kde.score_samples([[conds[i]]])[0]))
            MAE.append(np.mean(np.abs(ry-sim_conds[i])))
        

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(sim_conds,KDE,color='#003F5C')
        plt.title('KDE vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('KDE of Condition Label in Output Distribution')
        plt.savefig('./Evaluation/PcDGAN/KDE_Simulated_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(sim_conds,MAE,color='#003F5C')
        plt.title('Label Score vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Label Score')
        plt.savefig('./Evaluation/PcDGAN/Label Score_Simulated_' + str(args.id) + '.png')

    if args.model == 'CcGAN':
        if args.kappa == None:
            args.kappa = -2.0
        if args.disc_lr == None:
            args.disc_lr = 1e-4
        model = CcGAN(kappa=args.kappa, sigma=args.sigma, strategy=args.vicinal_type,Y=Y)

        if os.path.exists('./Weights/Estimator/'+args.estimator+'.index'):
            model.estimator.load_weights('./Weights/Estimator/'+args.estimator)
        else:
            raise Exception('could not find estimator weights.')

        if os.path.exists('./Weights/Embedder/'+args.embedder+'.index'):
            model.embedder.load_weights('./Weights/Embedder/'+args.embedder)
        else:
            raise Exception('could not find Embedder weights.')

        print('Current Settings:')
        pprint.pprint(vars(args))

        if args.mode == 'train':
            model.train(X,Y,train_steps=args.train_steps,batch_size=args.batch_size, disc_lr=args.disc_lr, gen_lr=args.gen_lr)
            model.discriminator.save_weights('./Weights/Discriminator/CcGAN/experiment'+args.id)
            model.generator.save_weights('./Weights/Generator/CcGAN/experiment'+args.id)
        else:
            model.discriminator.load_weights('./Weights/Discriminator/CcGAN/experiment'+args.id)
            model.generator.load_weights('./Weights/Generator/CcGAN/experiment'+args.id)

        sample_generator = get_generator(model.generator,model.estimator,model.embedder)
        divers = []
        KDEs = []
        MAEs = []
        batch_size = 500
        progress = trange(1000,position=0, leave=True)
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
                progress.update(1)
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
        
        print("\nSummary of Performance:\n")
        print(tabulate([['CcGAN', str(CcGAN_MAE_overall) + '+/-' + str(CcGAN_MAE_std_overall),str(CcGAN_KDE_overall) + '+/-' + str(CcGAN_KDE_std_overall),str(CcGAN_diver_overall) + '+/-' + str(CcGAN_diver_std_overall)]], headers=['Model', 'Label Score', 'Probability Density', 'Diversity']))

        print('Generating Sample Airfoils ...')
        conds = np.linspace(0.05,0.95,20)
        batch_size = 100
        conditions = []
        samples = []
        for i in trange(20):
            samples_, _, _ = sample_generator(conds[i] * tf.ones([batch_size,1]),batch_size)
            conditions.append(conds[i] * tf.ones([batch_size,1]))
            samples.append(samples_)
        samples = np.concatenate(samples,0)
        conditions = np.concatenate(conditions,0)
        np.save('./Samples/CcGAN/airfoils_' + str(args.id) + '.npy', samples)

        conds = np.linspace(0.05,0.95,100)

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,CcGAN_diver,color='#003F5C')
        plt.fill_between(conds,CcGAN_diver-CcGAN_diver_std,CcGAN_diver+CcGAN_diver_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('Diversity vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Diversity')
        plt.savefig('./Evaluation/CcGAN/Diversity_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,CcGAN_KDE,color='#003F5C')
        plt.fill_between(conds,CcGAN_KDE-CcGAN_KDE_std,CcGAN_KDE+CcGAN_KDE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('KDE vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('KDE of Condition Label in Output Distribution')
        plt.savefig('./Evaluation/CcGAN/KDE_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(conds,CcGAN_MAE,color='#003F5C')
        plt.fill_between(conds,CcGAN_MAE-CcGAN_MAE_std,CcGAN_MAE+CcGAN_MAE_std,facecolor='#003F5C',edgecolor="#003F5C",alpha=0.3)
        plt.title('Label Score vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Label Score')
        plt.savefig('./Evaluation/CcGAN/Label Score_' + str(args.id) + '.png')

        LDRs = batch_simulate(samples,'./Samples/CcGAN/txts/')
        sim_conds = np.linspace(0.05,0.95,20)
        KDE = []
        MAE = []
        for i in range(20):
            ry = (LDRs[i*100:(i+1)*100] - min_y)/(max_y-min_y)
            ry = ry[~np.isnan(ry)]
            bandwidth = np.arange(0.01, 0.1, .01)
            kde = KernelDensity(kernel='gaussian')
            grid = GridSearchCV(kde, {'bandwidth': bandwidth})
            grid.fit(y.numpy())
            kde = grid.best_estimator_
            KDE.append(np.exp(kde.score_samples([[conds[i]]])[0]))
            MAE.append(np.mean(np.abs(ry-sim_conds[i])))
        

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(sim_conds,KDE,color='#003F5C')
        plt.title('KDE vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('KDE of Condition Label in Output Distribution')
        plt.savefig('./Evaluation/CcGAN/KDE_Simulated_' + str(args.id) + '.png')

        plt.figure(figsize=(18,12))
        plt.rc('font', size=45)
        plt.plot(sim_conds,MAE,color='#003F5C')
        plt.title('Label Score vs Condition')
        plt.xlabel('Condition')
        plt.ylabel('Label Score')
        plt.savefig('./Evaluation/CcGAN/Label Score_Simulated_' + str(args.id) + '.png')
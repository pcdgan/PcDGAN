import argparse
import os
import numpy as np
from glob import glob
from GANs import CcGAN
import pprint

parser = argparse.ArgumentParser(description='Estimator Embedder Training')
parser.add_argument('--data', type=str, help='The path to the data. Default: ./data', default='./data')
parser.add_argument('--estimator_save_name', type=str, help='The file name of the best checkpoint saved in the weights estimator folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--embedder_save_name', type=str, help='The file name of the best checkpoint saved in the embedder weights folder. Default: best_checkpoint', default='best_checkpoint')
parser.add_argument('--estimator_lr', type=float, default=1e-4, help='Initial estimator learning rate before decay. Default: 1e-4')
parser.add_argument('--embedder_lr', type=float, default=1e-4, help='Initial embedder learning rate before decay. Default: 1e-4')
parser.add_argument('--estimator_train_steps', type=int, default=10000, help='Number of training steps for estimator. Default: 10000')
parser.add_argument('--embedder_train_steps', type=int, default=10000, help='Number of training steps for embedder. Default: 10000')
parser.add_argument('--estimator_batch_size', type=int, default=256, help='Batch size for estimator Default: 256')
parser.add_argument('--embedder_batch_size', type=int, default=256, help='Batch size for embedder Default: 256')

args = parser.parse_args()

# Load Data
X_train = np.load(args.data + '/xs_train.npy').astype('float32')
X_test = np.load(args.data + '/xs_test.npy').astype('float32')
Y_train = np.load(args.data + '/ys_train.npy').astype('float32')
Y_test = np.load(args.data + '/ys_test.npy').astype('float32')
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

print('Training Parameters:')
pprint.pprint(vars(args))

#check directories
if not os.path.exists('./Weights'):
    os.mkdir('./Weights')

if not os.path.exists('./Samples'):
    os.mkdir('./Samples')
    
if not os.path.exists('./Evaluation'):
    os.mkdir('./Evaluations')
    
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
    
dummy_model = CcGAN()
dummy_model.train_estimator(X_train,Y_train,X_test,Y_test, lr=args.estimator_lr, train_steps=args.estimator_train_steps, batch_size=args.estimator_batch_size, early_stop_save='./Weights/Estimator/' + args.estimator_save_name)
dummy_model.train_embedder(X_train,Y_train,X_test,Y_test, lr=args.embedder_lr, train_steps=args.embedder_train_steps, batch_size=args.embedder_batch_size, early_stop_save='./Weights/Embedder/' + args.embedder_save_name)
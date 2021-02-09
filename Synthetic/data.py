import numpy as np
import tensorflow as tf
import itertools
import math

def gen_grid(d, points_per_axis, lb=0., rb=1.):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(0, d):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)


class SparseGrid2D(object):
    
    def __init__(self, N, lb=-0.75, rb=0.75, perturb=0.02):
        self.name = 'SparseGrid2D'
        data = np.random.uniform(low=lb,high=rb,size=(N,2))
        self.data = data
        
class Uniform2D(object):
    
    def __init__(self, N, lb=-0.5, rb=0.5):
        self.name = 'Uniform2D'
        data = np.random.uniform(low=lb,high=rb,size=(N,2))
        self.data = data
        
class Ring2D(object):
    
    def __init__(self, N, n_mixture=8, std=0.01, radius=0.5):
        """Gnerate 2D Ring"""
        thetas = np.linspace(0, 2 * np.pi, n_mixture, endpoint=False)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        centers = np.vstack((xs, ys)).T
        data = []
        for i in range(N):
            data.append(np.random.normal(centers[np.random.choice(n_mixture)], std))
        self.data = np.array(data)
        

class Grid2D(object):
    
    def __init__(self, N, n_mixture=9, std=0.01):
        """Generate 2D Grid"""
        centers = SparseGrid2D(n_mixture, lb=-0.4, rb=0.4, perturb=0.).data
        data = []
        for i in range(N):
            data.append(np.random.normal(centers[np.random.choice(centers.shape[0])], std))
        self.data = np.array(data)
        
    
class Donut2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D donut"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
                if norm_x <= 0.5 and norm_x >= 0.25:
                    data.append(x)
                    break
        self.data = np.array(data)
        
    
class ThinDonut2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D donut"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
                #if norm_x <= 0.425 and norm_x >= 0.375:
                if norm_x <= 0.375 and norm_x >= 0.325:
                    data.append(x)
                    break
        self.data = np.array(data)
        
    
class Arc2D(object):
    
    def __init__(self, N, lb=-1., ub=1.):
        """Gnerate 2D arc"""
        data = []
        for i in range(N):
            while True:
                x = np.random.uniform(-0.5, 0.5, size=2)
                norm_x = np.linalg.norm(x)
                theta = np.arctan2(x[1], x[0])
                if norm_x <= 0.5 and norm_x >= 0.25 and not (theta > -11./12*np.pi and theta < 0):
                    data.append(x)
                    break
        self.data = np.array(data)

        
class Function(object):
    
    def __init__(self):
        pass
    
    def equation(self, x):
        y = 0.
        for i in range(self.n_modes):
            y += tf.exp(-.5 * tf.reduce_sum(tf.square(x-self.centers[i]), axis=1)/self.sigma**2)
        return y
    
    def evaluate(self, data):
        return self.equation(data)
    
    def entropy(self, data):
        assert hasattr(self, 'sigma')
        N = data.shape[0]
        counts = []
        for center in self.centers:
            distances = np.linalg.norm(data-center, axis=1)
            count = sum(distances<self.sigma)
            counts.append(count)
        counts = np.array(counts)
        rates = counts/N
        entropy = -np.sum(rates*np.log(rates+1e-8))
        return entropy


class Linear(Function):
    
    def __init__(self):
        self.name = 'Linear'
        self.dim = 2
        
    def equation(self, x):
        y = tf.reduce_sum(x, axis=1)
        return y


class MixGrid(Function):
    
    def __init__(self, n_modes=9, lb=-0.5, rb=0.5):
        self.name = 'MixGrid'
        self.dim = 2
        self.n_modes = n_modes
        
        points_per_axis = int(n_modes**0.5)
        self.sigma = (rb-lb)/points_per_axis/4
        
        self.centers = gen_grid(2, points_per_axis, lb=lb+2*self.sigma, rb=rb-2*self.sigma)
    

class MixRing(Function):
    
    def __init__(self, n_modes=4, radius=0.4):
        self.name = 'MixRing'
        self.dim = 2
        self.n_modes = n_modes
        self.radius = radius
        
        thetas = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        self.centers = np.vstack((xs, ys)).T
        
        self.sigma = np.pi*self.radius/self.n_modes/2
        

class MixRing1(MixRing):
    
    def __init__(self):
        super(MixRing1, self).__init__(n_modes=1)
        self.name = 'MixRing1'
        

class MixRing4(MixRing):
    
    def __init__(self):
        super(MixRing4, self).__init__(n_modes=4)
        self.name = 'MixRing4'
        

class MixRing6(MixRing):
    
    def __init__(self):
        super(MixRing6, self).__init__(n_modes=6)
        self.name = 'MixRing6'
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import noise
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from random import random
import scipy.io
from tqdm import tqdm
import tensorflow as tf
import time
# In[2]:


class DataGenerator: 
    
    def __init__(self, topNm = 1000):
        self.dim = 3
        mapfile = scipy.io.loadmat('EyeRGBMap.mat')
        # print(mapfile.keys())
        self.colorMap = np.array(mapfile['ColorMap']).reshape(5001,self.dim)
        self.topNm = topNm
        
    def normalize(self, data):
        m = np.max(data)
        mi = np.min(data)
        norm = (data - mi) / (m - mi)
        return norm
    
    def addNoise(self, image, std):
        row,col,ch= image.shape
        mean = 0
        sigma = stds

        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + np.multiply(image,gauss)
        return noisy
    
    def tensorNormalize(self, data):
        m = tf.math.argmax(data)
        mi = tf.math.argmin(data)
        
        data = tf.cast(data, tf.float32)
        m = tf.cast(m, tf.float32)
        mi = tf.cast(mi, tf.float32)
        
        norm = (data - mi) / (m - mi)
        return norm
    
    def negativeNormalize(self, inp):
        minimum = 0
        maximum = 1
        average      = (minimum + maximum) / 2;
        rng        = (maximum - minimum) / 2;
        normalized_x = (inp - average) / rng;
        return normalized_x;

    def fetchImageFromDepth(self, depth):
        newArray = np.zeros((144,144,self.dim))
        for i in range(144):
            for k in range(144):
                newArray[i][k] = self.colorMap[int(depth[i][k])]
        return newArray
    
    def fetchTensorFromDepth(self, depth):
        newArray = tf.zeros((144,144,self.dim))
        for i in range(144):
            for k in range(144):
                print(tf.gather_nd(depth, [[i,k]]))
                newArray[i][k] = self.colorMap[tf.gather_nd(depth, [[i,k]])]
        return newArray
        

    def generatePerlinData(self, numData, std = 0):
        shape = (144,144)
        scale = 100
        octaves = 6
        persistence = 0.5
        lacunarity = 1.8

        depthMaps = []
        images = []



        for z in tqdm(range(numData)):
            world = np.zeros(shape)
            randZ = random()
            for i in range(shape[0]):
                for j in range(shape[1]):
                    world[i][j] = noise.pnoise3(i/scale, 
                                                j/scale, 
                                                randZ,
                                                octaves=octaves, 
                                                persistence=persistence, 
                                                lacunarity=lacunarity, 
                                                repeatx=1024, 
                                                repeaty=1024, 
                                                base=40)
            normalizedWorld = np.rint(self.normalize(np.reshape(world, (144,144,1)))*self.topNm)
            depthMaps.append(normalizedWorld)
            images.append(self.addNoise(self.fetchImageFromDepth(normalizedWorld), std))
        
        images = np.array(images)
        depthMaps = np.array(depthMaps)
            
        images = self.negativeNormalize(self.normalize(images[:,0:144,0:144,0:self.dim]))
        depthMaps = self.negativeNormalize(self.normalize(depthMaps[:,0:144,0:144,0:1]))
            
            
        return (images, depthMaps)


    def generateGaussianData(self, numData, std = 0):
        
       
        maps = np.load("20kGauss.npy")
        
                    
        maps = maps[0:numData,:,:]
        gImages = []

        for i in tqdm(range(numData)):
                        
            maps[i] = np.rint(self.normalize(maps[i])*self.topNm)
            gImages.append(self.addNoise(self.fetchImageFromDepth(maps[i]), std))
            
        gImages = np.array(gImages)
        maps = np.array(maps)
        
            
        gImages = self.negativeNormalize(self.normalize(gImages[:,0:144,0:144,0:self.dim]))
        maps = self.negativeNormalize(self.normalize(maps[:,0:144,0:144]))
        return (gImages, maps.reshape((numData,144,144,1)))
    
    
    def generateMixedNoise(self, numPerlin, numGaussian, std = 0):
        gImages, gMaps = self.generateGaussianData(numGaussian, std = std)
        pImages, pMaps = self.generatePerlinData(numPerlin, std = std)
                
        images = np.concatenate((gImages, pImages))
        maps = np.concatenate((gMaps, pMaps))
        
        
        return self.unison_shuffled_copies(images,maps)
    
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def fetchTestingSet(self):
        gImages, gMaps = self.generateGaussianData(1000)
        pImages, pMaps = self.generatePerlinData(1000)
        
        return (gImages, gMaps, pImages, pMaps)
        

        
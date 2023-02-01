#!/usr/bin/env python
# coding: utf-8

# In[5]:


import noise
import cv2
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from random import random, uniform, randint, sample
import scipy.io
from PIL import Image, ImageEnhance
from tqdm import tqdm
import tensorflow as tf
import time
from math import sqrt, ceil
import copy
# In[2]:


class VideoGenerator: 
    
    def __init__(self, topNm = 1000, frameRate = 4, blinkDuration = 3, dim = 144):
        self.channels = 3
        self.dim = dim
        self.colorMap = np.array(scipy.io.loadmat('EyeRGBMap.mat')['ColorMap']).reshape(5001,self.channels)
        self.topNm = topNm
        self.gaussMaps = np.load("20kGauss.npy")
        self.frameRate = frameRate
        self.blinkDuration = blinkDuration

        
    def normalize(self, data):
        m = np.max(data)
        mi = np.min(data)
        norm = (data - mi) / (m - mi)
        return norm
    
    def normTwoValues(self, data, dMin, dMax):
        return np.interp(data, (data.min(), data.max()), (dMin, dMax))

    
    def addNoise(self, image, std):
        row,col,ch= image.shape
        mean = 0
        sigma = std

        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + np.multiply(image,gauss)
        return noisy
    
    def fetchStep(self, time, alpha, initial_frame):
        return np.divide(initial_frame,(1 + alpha*sqrt(time)))
    
    def generateVideos(self, numVideos, variance = 0.12):
        image, dMap = self.generatePerlinData(1)
        _, nMap = self.generateGaussianData(1)
        image = image[0]
        dMap = dMap[0,:,:,0]
        nMap = nMap[0,:,:,0]
        
        alpha = uniform(0.16,0.28)
        
        
        nMap = self.normTwoValues(nMap, alpha-variance, alpha + variance)
        nFrames= self.frameRate * self.blinkDuration
                
        
        finalImages = np.zeros((numVideos, nFrames, self.dim, self.dim, 3))
        finalCleanImages = np.zeros((numVideos, nFrames, self.dim, self.dim, 3))
        finalMaps = np.zeros((numVideos, nFrames, self.dim, self.dim))
    

        
        
        for i in tqdm(range(numVideos)):
            print("status", i)
            for k in range(self.frameRate * self.blinkDuration):
                time = 1/self.frameRate * k
                
                x = np.linspace(0,2000,num=144)
                y = np.linspace(0, 2000,num=144)
                X, Y = np.meshgrid(x, y)
                Y = np.flip(Y,0)
                
                step = Y - self.fetchStep(time, nMap, dMap)
                
                
                finalImages[i,k] = self.fetchImageFromDepth(step) * self.circleTransform(diameter = 20, value = 0, jitter = 7) * self.circleTransform(diameter = 120, value = 0.6, jitter = 7)
                finalImages[i, k] = self.add_eyelashes(finalImages[i, k])
                
                finalCleanImages[i,k] = self.fetchImageFromDepth(step)
                
                
                finalMaps[i,k] = step
                
                #saturation --> 0.5 1 (normal)
                #brightness --> 0.5 1 (normal)
                #randomly apply transforms --> randomn decimal (inclusive)
                #transform should be video wide

            
            self.applySaturationTransform(finalImages[i])
            self.applyBrightnessTransform(finalImages[i])
            self.applyBlurs(finalImages[i])
            self.addEyelashes(finalImages[i])
                
                
        return (finalImages, finalCleanImages, finalMaps)
    
    def applyBrightnessTransform(self, imageSet):
        frames = sample(range(0,self.frameRate * self.blinkDuration-1), randint(3,6))


       
        for frame in frames:
            rgb = Image.fromarray(np.uint8(imageSet[frame]))
            enhancer = ImageEnhance.Brightness(rgb)
            new_image = enhancer.enhance(np.random.random() * 0.5 + 0.5)
            imageSet[frame] = new_image
    
    def applySaturationTransform(self, imageSet):
        frames = sample(range(0,self.frameRate * self.blinkDuration-1), randint(3,6))
       
        for frame in frames:
            rgb = Image.fromarray(np.uint8(imageSet[frame]))
            enhancer = ImageEnhance.Color(rgb)
            new_image = enhancer.enhance(np.random.random() * 0.5 + 0.5)
            imageSet[frame] = new_image

    
    def applyBlurs(self, imageSet):
        frames = sample(range(0,self.frameRate * self.blinkDuration-1), randint(3,6))
        
        for frame in frames:
            imageSet[frame] = self.frameBlur(imageSet[frame])
            
    
        
    
    
    def frameBlur(self, img):
        h_kernel_size = randint(3,9)
        v_kernel_size = randint(3,9)

        # Create the vertical kernel.
        kernel_v = np.zeros((v_kernel_size, v_kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.zeros((h_kernel_size,h_kernel_size))

        # Fill the middle row with ones.
        kernel_v[:, int((v_kernel_size - 1)/2)] = np.ones(v_kernel_size)
        kernel_h[int((h_kernel_size - 1)/2), :] = np.ones(h_kernel_size)

        # Normalize.
        kernel_v /= v_kernel_size
        kernel_h /= h_kernel_size

        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(img, -1, kernel_v)

        # Apply the horizontal kernel.
        horizontal_mb = cv2.filter2D(vertical_mb, -1, kernel_h)
        
        return horizontal_mb    
        
        
    
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
        newArray = np.zeros((self.dim,self.dim,self.channels))
        for i in range(self.dim):
            for k in range(self.dim):
                newArray[i][k] = self.colorMap[int(depth[i][k])]
        return newArray
    
    def fetchTensorFromDepth(self, depth):
        newArray = tf.zeros((self.dim,self.dim,self.channels))
        for i in range(self.dim):
            for k in range(self.dim):
                newArray[i][k] = self.colorMap[tf.gather_nd(depth, [[i,k]])]
        return newArray
        

    def generatePerlinData(self, numData, std = 0):
        shape = (self.dim,self.dim)
        scale = 100
        octaves = 6
        persistence = 0.5
        lacunarity = 1.8

        depthMaps = []
        images = []



        for z in range(numData):
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
            normalizedWorld = np.rint(self.normalize(np.reshape(world, (self.dim,self.dim,1)))*self.topNm)
            depthMaps.append(normalizedWorld)
            images.append(self.addNoise(self.fetchImageFromDepth(normalizedWorld), std))
        
        images = np.array(images)
        depthMaps = np.array(depthMaps)
            
        images = self.normalize(images[:,0:self.dim,0:self.dim,0:self.channels]) #needs to be fixed
        depthMaps = self.normalize(depthMaps[:,0:self.dim,0:self.dim,0:1]) * self.topNm #needs to be fixed
            
            
        return (images, depthMaps)


    def generateGaussianData(self, numData, std = 0):
        
        maps = np.copy(self.gaussMaps)
            
                    
        maps = maps[0:numData,:,:]
        gImages = []

        for i in range(numData):
                        
            maps[i] = np.rint(self.normalize(maps[i])*self.topNm)
            gImages.append(self.addNoise(self.fetchImageFromDepth(maps[i]), std))
            
        gImages = np.array(gImages)
        maps = np.array(maps)
        
            
        gImages = self.normalize(gImages[:,0:self.dim,0:self.dim,0:self.channels])
        maps = self.normalize(maps[:,0:self.dim,0:self.dim])
        return (gImages, maps.reshape((numData,self.dim,self.dim,1)))
    
    
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
        images, _, maps = self.generateVideos(100)
        return images, maps
            
    
    def circleTransform(self, diameter, value, jitter):
        
        img = np.ones((self.dim,self.dim,3))
        
        center = (72 + randint(0, jitter),72 + randint(0, jitter))


        """
            Creates a matrix of ones filling a circle.
        """

        # gets the radious of the image
        radious  = diameter//2

        # gets the row and column center of the image
        row, col = center 

        # generates theta vector to variate the angle
        theta = np.arange(0, 360)*(np.pi/180)

        # generates the indexes of the column
        y = (radious*np.sin(theta)).astype("int32") 

        # generates the indexes of the rows
        x = (radious*np.cos(theta)).astype("int32") 

        # with:
        # img[x, y] = 1
        # you can draw the border of the circle 
        # instead of the inner part and the border. 

        # centers the circle at the input center
        rows = x + (row)
        cols  = y + (col)

        # gets the number of rows and columns to make 
        # to cut by half the execution
        nrows = rows.shape[0] 
        ncols = cols.shape[0]

        # makes a copy of the image
        img_copy = copy.deepcopy(img)

        # We use the simetry in our favour
        # does reflection on the horizontal axes 
        # and in the vertical axes

        for row_down, row_up, col1, col2 in zip(rows[:nrows//4],
                                np.flip(rows[nrows//4:nrows//2]),
                                cols[:ncols//4],
                                cols[nrows//2:3*ncols//4]):

            img_copy[row_up:row_down, col2:col1] = value


        return img_copy
    
    def add_eyelashes(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        img2 = Image.open(r"sample_eyelash")
  
        im_pil.paste(img2, (0,0), mask = img2)
        
        return np.asarray(im_pil)
#         height = image.shape[0]
#         width = image.shape[1]
#         x_current = 30
#         x_end = 100
#         y_start = 10
#         img = image
#         while(x_current < x_end):
#             x_shift = ceil(np.random.random() * 10)
#             x_offset = ceil(((np.random.random() * 2) - 1) * 20)
#             length = ceil(40 + ((np.random.random() * 2) - 1) * 20)
#             img = cv2.line(image, (x_current, y_start), (x_current - x_offset, y_start + length), (0,0,0), 2)
#             x_current += x_shift
#         return img
        
        
  
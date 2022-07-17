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
        self.stdGlobal = uniform(0.01, 0.1)
        
        
    def cleanNegativeNormalize(self, data, top):
        return (data / (top/2)) - 1

        
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
    
    
    def randomRangeNormalize(self,image):
        start = randint(0,self.topNm-1000)
        rng = randint(600,1000)
        
        return self.normTwoValues(image, start, rng + start)
    
    def fr(self, data):
        print(np.min(data), np.max(data))
    
    def generateVideos(self, numVideos, variance = 0.12):
        
        nFrames= self.frameRate * self.blinkDuration
                
        
        finalImages = np.zeros((numVideos, nFrames, self.dim, self.dim, 3))
        finalCleanImages = np.zeros((numVideos, nFrames, self.dim, self.dim, 3))
        finalMaps = np.zeros((numVideos, nFrames, self.dim, self.dim))
    

        
        
        for i in tqdm(range(numVideos)):
            
            image, dMap = self.generatePerlinData(1) #0, 255 & 0,1
            
        
            _, nMap = self.generateGaussianData(1)
            image = image[0]
            nMap = nMap[0,:,:,0]
            dMap = dMap[0,:,:,0]

            alpha = uniform(0.16,0.28)
            


            nMap = self.normTwoValues(nMap, alpha-variance, alpha + variance)
            
            
            
            for k in range(self.frameRate * self.blinkDuration):
                time = 1/self.frameRate * k
                
                
                
                step = self.fetchStep(time, nMap, dMap)
          
                finalImages[i,k] = self.fetchImageFromDepth(step)
                imageLength = finalImages[i,k].shape[0]
                innerDiameter = round(self.dim * 0.139)
                outerDiameter = round(self.dim * 0.833)
                finalImages[i,k] = finalImages[i,k] * self.circleTransform(diameter = innerDiameter, value = 0, jitter = 1)
                finalImages[i,k] = finalImages[i,k] * self.circleTransform(diameter = outerDiameter, value = 0.6, jitter = 1)
                
                finalCleanImages[i,k] = self.fetchImageFromDepth(step)
                
                
                finalMaps[i,k] = step

            
            self.applySaturationTransform(finalImages[i])
            self.applyBrightnessTransform(finalImages[i])
            self.addGaussianNoise(finalImages[i])
            self.applyBlurs(finalImages[i])
            self.addEyelashes(finalImages[i])
            
                
                
        return (self.cleanNegativeNormalize(finalImages, 255), self.cleanNegativeNormalize(finalCleanImages, 255), self.cleanNegativeNormalize(finalMaps, self.topNm))
    
    
    
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
        for i, frame in enumerate(imageSet):
            if(i in frames):
                imageSet[i] = self.frameBlur(frame)
            else:
                imageSet[i] = self.smallBlur(frame)
    
    def smallBlur(self, image):
        return cv2.blur(image, (10,10))
            
    
        
    
    
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

    def fetchImageFromDepth(self, depth, std = 0.01):
        newArray = np.zeros((self.dim,self.dim,self.channels))
        for i in range(self.dim):
            for k in range(self.dim):
                newArray[i][k] = self.colorMap[int(depth[i][k])]
        return self.addNoise(newArray, self.stdGlobal)
    
    def fetchTensorFromDepth(self, depth):
        newArray = tf.zeros((self.dim,self.dim,self.channels))
        for i in range(self.dim):
            for k in range(self.dim):
                newArray[i][k] = self.colorMap[tf.gather_nd(depth, [[i,k]])]
        return newArray
        

    def generatePerlinData(self, numData, std = 0.01):
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
            normalizedWorld = self.randomRangeNormalize(np.rint(self.normalize(np.reshape(world, (self.dim,self.dim,1)))*self.topNm))
            depthMaps.append(normalizedWorld)
            images.append(self.addNoise(self.fetchImageFromDepth(normalizedWorld), self.stdGlobal))
        
        images = np.array(images)
        depthMaps = np.array(depthMaps)
            
        images = images[:,0:self.dim,0:self.dim,0:self.channels] #needs to be fixed
        depthMaps = depthMaps[:,0:self.dim,0:self.dim,0:1] #needs to be fixed
            
            
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
        
        center = (round(self.dim/2) + randint(0, jitter),round(self.dim/2) + randint(0, jitter))


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
    
    def addEyelashes(self, imageSet):
        # frames = sample(range(0,self.frameRate * self.blinkDuration-1), randint(3,6))
        frames = np.arange(len(imageSet))

       
        for frame in frames:
            im_pil = Image.fromarray(np.uint8(imageSet[frame]))
            
            eyelash = Image.open(r"small_eyelash.png")
            eyelash = eyelash.resize((round(self.dim * 0.111), round(self.dim * 0.257)))
            size = eyelash.size

            num_eyelashes = randint(8, 12)
            for i in range(0, num_eyelashes):
                rotated = eyelash.rotate(randint(-40, 40))
                scaleW = uniform(0.5, 0.6)
                scaleH = uniform(0.6, 1.2)
                new_size = (ceil(size[0] * scaleW), ceil(size[1] * scaleH))
                resized = rotated.resize(new_size)
                x_coord = randint(round(0.174 * self.dim), round(0.868 * self.dim))
                y_coord = round(0.0694 * self.dim) + randint(-5, 5)

                im_pil.paste(resized, (x_coord, y_coord), mask = resized)

            imageSet[frame] = im_pil
            
    def addGaussianNoise(self, imageSet):
        frames = np.arange(len(imageSet))

       
        for frame in frames:
            image = imageSet[frame]
            # if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 0.1
            sigma = uniform(0.01, 0.1)
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
        return noisy
#         elif noise_typ == "s&p":
#             row,col,ch = image.shape
#             s_vs_p = 0.5
#             amount = 0.004
#             out = np.copy(image)
#             # Salt mode
#             num_salt = np.ceil(amount * image.size * s_vs_p)
#             coords = [np.random.randint(0, i - 1, int(num_salt))
#                 for i in image.shape]
#                     out[coords] = 1

#           # Pepper mode
#           num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
#           coords = [np.random.randint(0, i - 1, int(num_pepper))
#                   for i in image.shape]
#           out[coords] = 0
#           return out
#       elif noise_typ == "poisson":
#           vals = len(np.unique(image))
#           vals = 2 ** np.ceil(np.log2(vals))
#           noisy = np.random.poisson(image * vals) / float(vals)
#           return noisy
#       elif noise_typ =="speckle":
#           row,col,ch = image.shape
#           gauss = np.random.randn(row,col,ch)
#           gauss = gauss.reshape(row,col,ch)        
#           noisy = image + image * gauss
#           return noisy

        
        
  
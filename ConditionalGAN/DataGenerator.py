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
from PIL import Image, ImageEnhance
from random import random, uniform, randint, sample
import cv2
import copy
from multiprocessing import Pool
from tqdm import tqdm# In[2]:
import multiprocessing as mp



class DataGenerator: 
    
    def __init__(self, topNm = 1000):
        self.dim = 3
        self.width = 144
        mapfile = scipy.io.loadmat('EyeRGBMap.mat')
        # print(mapfile.keys())
        self.colorMap = np.array(mapfile['ColorMap']).reshape(5001,self.dim)
        self.topNm = topNm
        
    def normalize(self, data):
        m = np.max(data)
        mi = np.min(data)
        norm = (data - mi) / (m - mi)
        return norm
    
    
    def normalize_array(self, array):
        
        lower_bound = randint(500, 1800)
        upper_bound = lower_bound + randint(200, 700)
        
        minimum, maximum = np.min(array), np.max(array)
        array = (array - minimum) * (upper_bound - lower_bound) / (maximum - minimum) + lower_bound
        return np.rint(np.clip(array, lower_bound, upper_bound))
    
    def addNoise(self, image, std):
        row,col,ch= image.shape
        mean = 0
        sigma = std

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
        newArray = np.zeros((self.width,self.width,self.dim))
        for i in range(self.width):
            for k in range(self.width):
                newArray[i][k] = self.colorMap[int(depth[i][k])]
        return newArray
    
    def fetchTensorFromDepth(self, depth):
        newArray = tf.zeros((self.width,self.width,self.dim))
        for i in range(self.width):
            for k in range(self.width):
                print(tf.gather_nd(depth, [[i,k]]))
                newArray[i][k] = self.colorMap[tf.gather_nd(depth, [[i,k]])]
        return newArray
    
    
    def normalize_maps(self, arr):
        arr = np.array(arr, dtype=np.float32)
        arr = 2 * (arr - 0) / (2500 - 0) - 1
        return arr

    def normalize_images(self, arr):
        arr = np.array(arr, dtype=np.float32)
        arr = 2 * (arr - 0) / (255 - 0) - 1
        return arr
    



    def generate_data(self, z, width, topNm):
        shape = (width, width)
        persistence = 0.5
        lacunarity = 1.8

        world = np.zeros(shape)
        randZ = random()
        octaves = randint(4, 6) #more octaves = more "noise" in depth map. less octaves = smoother, simpler
        scale = randint(200, 400) #more scale = less patterning. 
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
        normalizedWorld = self.normalize_array(np.reshape(world, (width, width, 1)) * topNm)

        noisy = self.add_gaussian_noise(self.fetchImageFromDepth(normalizedWorld), 26) * self.circleTransform(diameter=35, value=0, jitter=7) * self.circleTransform(diameter=120, value=0.6, jitter=7)
        noisy = self.applySaturationTransform(noisy)
        noisy = self.applyBrightnessTransform(noisy)
        noisy = self.frameBlur(noisy)

        return (noisy, normalizedWorld)
    
    
    def generatePerlinData(self, numData, std=0):
        depthMaps = np.empty((numData, self.width, self.width, 1))
        images = np.empty((numData, self.width, self.width, self.dim))

        with Pool(processes=mp.cpu_count()) as pool:
            results = [pool.apply_async(self.generate_data, args=(i, self.width, self.topNm)) for i in range(numData)]
            results = [result.get() for result in tqdm(results, total=numData)]

        for i, (noisy, normalizedWorld) in enumerate(results):
            depthMaps[i] = normalizedWorld
            images[i] = noisy

        images = self.normalize_images(images[:,0:self.width,0:self.width,0:self.dim])
        depthMaps = self.normalize_maps(depthMaps[:,0:self.width,0:self.width,0:1])

        return (images, depthMaps)



        

#     def generatePerlinData(self, numData, std = 0):
#         shape = (self.width,self.width)
#         persistence = 0.5
#         lacunarity = 1.8

#         depthMaps = []
#         images = []



#         for z in tqdm(range(numData)):
#             world = np.zeros(shape)
#             randZ = random()
#             octaves = randint(4, 6) #more octaves = more "noise" in depth map. less octaves = smoother, simpler
#             scale = randint(200, 400) #more scale = less patterning. 
#             for i in range(shape[0]):
#                 for j in range(shape[1]):
#                     world[i][j] = noise.pnoise3(i/scale, 
#                                                 j/scale, 
#                                                 randZ,
#                                                 octaves=octaves, 
#                                                 persistence=persistence, 
#                                                 lacunarity=lacunarity, 
#                                                 repeatx=1024, 
#                                                 repeaty=1024, 
#                                                 base=40)
#             normalizedWorld = self.normalize_array(np.reshape(world, (self.width,self.width,1)) * self.topNm) #yields our SCALED array
#             depthMaps.append(normalizedWorld)
            
#             noisy = self.add_gaussian_noise(self.fetchImageFromDepth(normalizedWorld)) * self.circleTransform(diameter = 35, value = 0, jitter = 7) * self.circleTransform(diameter = 120, value = 0.6, jitter = 7)
#             noisy = self.applySaturationTransform(noisy)
#             noisy = self.applyBrightnessTransform(noisy)
#             noisy = self.frameBlur(noisy)
                        
            
#             images.append(noisy)
        
#         images = np.array(images)
#         depthMaps = np.array(depthMaps)
            
#         images = self.normalize_images(images[:,0:self.width,0:self.width,0:self.dim])
#         depthMaps = self.normalize_maps(depthMaps[:,0:self.width,0:self.width,0:1])
            
            
#         return (images, depthMaps)


    def generateGaussianData(self, numData, std = 0):
        
       
        maps = np.load("20kGauss.npy")
        
                    
        maps = maps[0:numData,:,:]
        gImages = []

        for i in tqdm(range(numData)):
                        
            maps[i] = np.rint(self.normalize(maps[i])*self.topNm)
            gImages.append(self.add_gaussian_noise(self.fetchImageFromDepth(maps[i])))
            
        gImages = np.array(gImages)
        maps = np.array(maps)
        
            
        gImages = self.negativeNormalize(self.normalize(gImages[:,0:self.width,0:self.width,0:self.dim]))
        maps = self.negativeNormalize(self.normalize(maps[:,0:self.width,0:self.width]))
        return (gImages, maps.reshape((numData,self.width,self.width,1)))
    
    
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
        
    
    def applyBrightnessTransform(self, image):
        rgb = Image.fromarray(np.uint8(image))
        enhancer = ImageEnhance.Brightness(rgb)
        new_image = enhancer.enhance(np.random.random() * 0.5 + 0.5)
        return np.asarray(new_image)
    
    def applySaturationTransform(self, image):
        rgb = Image.fromarray(np.uint8(image))
        enhancer = ImageEnhance.Color(rgb)
        new_image = enhancer.enhance(np.random.random() * 0.5 + 0.5)
        return np.asarray(new_image)

    
    
    def frameBlur(self, img):
        h_kernel_size = randint(2,6)
        v_kernel_size = randint(2,6)

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
        
        
    def circleTransform(self, diameter, value, jitter):
        
        img = np.ones((self.width,self.width,3))
        
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
    
    def add_gaussian_noise(self, image, stdev):
        # Get the dimensions of the image
        rows, cols, channels = image.shape

        # Generate Gaussian noise with mean 0 and standard deviation stdev
        noise = np.random.normal(0, stdev, (rows, cols, channels))

        # Convert the image to float type
        image = image.astype(np.float)

        # Add the noise to the image
        noisy_image = cv2.add(image, noise)

        # Clip the image to the range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)

        # Convert the image back to uint8 type
        noisy_image = noisy_image.astype(np.uint8)

        return noisy_image
#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import heapq
import csv
from matplotlib import cm
import time
from math import sqrt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from multiprocessing import Pool


class ModelEvaluation:
    
    
    def __init__(self, topNm = 1000):
        self.topNm = topNm

    def addNoise(self, image, std):
        row,col,ch= image.shape
        mean = 0
        sigma = std

        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + np.multiply(image,gauss)
        return noisy


    def calc_mae(self, predictions, targets):
            m = tf.keras.metrics.MeanAbsoluteError()
            m.update_state(predictions,targets)
            return m.result().numpy()

    def calc_rmse(self, predictions, targets):
            m = tf.keras.metrics.RootMeanSquaredError()
            m.update_state(predictions,targets)
            return m.result().numpy()

    def evaluateTrio(self, model, images, maps, index, download = True, noise = False, std = 0.1, fileName = "tmp.png", label = "general"):

        
        
            #PREDICTION
            image = images[index]
            
            
            
            if noise:
                image = self.addNoise(image, std)
            tmpMap = maps[index]
            
            t0 = time.time()
            predMap = model(image[None,:,:,:], training=True).numpy()
            t1 = time.time()
            
            predMap = self.normalize(predMap[0]) * self.topNm
            tmpMap = self.normalize(tmpMap) * self.topNm
            
            
            #SLICING AND DICING TO FIRST INDEX, OVERFITTING TO 144
            predMap = predMap[0:144,:,:]
            tmpMap = tmpMap[0:144,:,:]
            image = image[0:144,:,:]
            
            

            #DISPLAY/DOWNLOAD
            x = np.arange(0, 144)
            y = np.arange(0, 144)

            fig = plt.figure(figsize=plt.figaspect(0.33))
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax2= fig.add_subplot(1, 3, 2, projection='3d')
            
            ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax1.set_title('Ground Truth')
            
            ax2.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax2.set_title('Prediction')
            
            ax3= fig.add_subplot(1, 3, 3)
            x = np.arange(0, 144)
            y = np.arange(0, 144)
            X, Y = np.meshgrid(x, y)
            pData = np.reshape(tmpMap[None,:,:,:], (144,144))
            pData2 = np.reshape(predMap, (144,144))
            surf = ax1.plot_surface(X, Y, pData)
            surf2 = ax2.plot_surface(X, Y, pData2)
                        
            
            ax3.imshow(image)
            if not download:
                plt.show()
            else:
                plt.savefig(label + "/" + fileName)
                

            return predMap.reshape(144,144), tmpMap.reshape(144,144)


    def saveMap(self,depthMap, filepath, dim = 51):

            fig = plt.figure(figsize = (5,5))
            ax1 = fig.add_subplot(projection='3d')

            x = np.arange(0, dim)
            y = np.arange(0, dim)
            X, Y = np.meshgrid(x, y)
            pData = np.reshape(depthMap[None,:,:,:], (dim,dim))
            surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)

            plt.savefig(filepath)
    
    def showMap(self,depthMap, dim = 48):

            fig = plt.figure(figsize = (5,5))
            ax1 = fig.add_subplot(projection='3d')

            x = np.arange(0, dim)
            y = np.arange(0, dim)
            X, Y = np.meshgrid(x, y)
            pData = np.reshape(depthMap[None,:,:,:], (dim,dim))
            surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)
            
            ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax1.set_title('Patient Tear Film Thickness Profile')
            
            

            plt.show()

        
    

    def normalize(self,data):
            m = np.max(data)
            mi = np.min(data)
            norm = (data - mi) / (m - mi)
            return norm
                
        
    
        
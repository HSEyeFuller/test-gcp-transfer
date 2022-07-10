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
import shutil
import os
import cv2
import re


class ModelEvaluation:
    
    
    def __init__(self, d1, d2, topNm = 1000):
        self.topNm = topNm
        self.dim = d2

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
        
        
        
    def saveImage(self, image, fn):
        data = np.reshape(image, (self.dim,self.dim,3))
        plt.imshow(self.imRegulate(data), interpolation='nearest')
        plt.savefig(f'tmp/{fn}.png')
        
        
    def saveMap(self, image, fn):
        pData = np.reshape(image, (self.dim,self.dim))

        fig = plt.figure(figsize = (5,5))
        ax1 = fig.add_subplot(projection='3d')
        # ax1.set_zlim(0, self.topNm)
        x = np.arange(0, self.dim)
        y = np.arange(0, self.dim)
        X, Y = np.meshgrid(x, y)
        surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)

        ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
        ax1.set_title('Patient Tear Film Thickness Profile')
        
        plt.savefig(f'tmp/{fn}.png')

        
    def imRegulate(self, data):
        m = np.max(data)
        mi = np.min(data)
        norm = ((data - mi) / (m - mi))*255
        return norm.astype(np.uint8)







    def saveImages(self, imgSet):


        try:
            shutil.rmtree('tmp/')
        except:
            print("tmp/ does not exist")


        try:
            os.mkdir('tmp/')
        except:
            print("tmp/ exists")

        try:
            os.remove('tmp.zip')
        except:
            print("tmp.zip does not exist")
        for frame in range(imgSet.shape[0]):
            self.saveImage(imgSet[frame], frame)

        shutil.make_archive('tmp', 'zip', 'tmp/')

    def sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def saveMaps(self, imgSet):
        
        print("SHAPE", imgSet.shape)


        try:
            shutil.rmtree('tmp/')
        except:
            print("tmp/ does not exist")


        try:
            os. mkdir('tmp/')
        except:
            print("tmp/ exists")

        try:
            os.remove('tmp.zip')
        except:
            print("tmp.zip does not exist")
        for frame in range(imgSet.shape[0]):
            self.saveMap(imgSet[frame], frame)
            

    def saveEyeVideo(self,imgSet, savePath):
        self.saveImages(imgSet)
        image_folder = 'tmp'

        images = [img for img in self.sorted_alphanumeric(os.listdir(image_folder)) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(f'{savePath}.avi', 0, 4, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        
        
        # self.convert_avi_to_mp4(f'{savePath}.avi', f'{savePath}.mp4')

        
    def convert_avi_to_mp4(self, avi_file_path, output_name):
        os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}'".format(input = avi_file_path, output = output_name))
        os.popen("rm '{input}'".format(input = avi_file_path))
        return True
        
        
    def saveMapVideo(self, imgSet, savePath):
        self.saveMaps(imgSet)
        image_folder = 'tmp'

        images = [img for img in self.sorted_alphanumeric(os.listdir(image_folder)) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(f'{savePath}.avi', 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        
        # self.convert_avi_to_mp4(f'{savePath}.avi', f'{savePath}.mp4')
        
        
        
    def unpack(self, tiles):
        depth = round(tiles.shape[0]/self.dim)
        channels = tiles.shape[2]
        output = np.zeros((depth, self.dim, self.dim, channels))
        print(output.shape, self.dim)
        for i in range(depth):
            output[i] = tiles[i*self.dim:(i+1)*self.dim,:]
            
            
        return output 
    
    
    def runTestingInference(self, model, image, dMap): #assumes pre-determined input format

            
   
            
            t0 = time()
            predMap = model(image[None,:,:,:], training=True).numpy()
            t1 = time.time()
            
            
            
            self.saveEyeVideo(self.unpack(image), "eye")
            self.saveMapVideo(self.unpack(dMap), "gt")
            self.saveMapVideo(self.unpack(predMap), "output")  
            
            
    def reNormalize(self, data):
        return data * (self.topNm/2) + self.topNm
        
            
            
            
            




    def evaluateTrio(self, model, images, maps, index, download = True, noise = False, std = 0.1, fileName = "tmp.png", label = "general"):

        
        
            #PREDICTION
            image = images[index]
            
            
            
            
            if noise:
                image = self.addNoise(image, std)
            tmpMap = self.reNormalize(maps[index])
            
            t0 = time.time()
            predMap = self.reNormalize(model(image[None,:,:,:], training=True).numpy()[0])
            t1 = time.time()
            
            
            
            self.saveEyeVideo(self.unpack(image), label + "/" + fileName + "eye")
            self.saveMapVideo(self.unpack(tmpMap), label + "/" + fileName + "input")
            self.saveMapVideo(self.unpack(predMap), label + "/" + fileName + "output")
            
            
            
            
            #SLICING AND DICING TO FIRST INDEX, OVERFITTING TO 144
            predMap = predMap[0:self.dim,:,:]
            tmpMap = tmpMap[0:self.dim,:,:]
            image = image[0:self.dim,:,:]
            
            
     
            

            #DISPLAY/DOWNLOAD
            x = np.arange(0, self.dim)
            y = np.arange(0, self.dim)

            fig = plt.figure(figsize=plt.figaspect(0.33))
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            ax2= fig.add_subplot(1, 3, 2, projection='3d')
            
            ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax1.set_title('Ground Truth')
            
            ax2.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax2.set_title('Prediction')
            
            ax3= fig.add_subplot(1, 3, 3)
            x = np.arange(0, self.dim)
            y = np.arange(0, self.dim)
            X, Y = np.meshgrid(x, y)
            pData = np.reshape(tmpMap[None,:,:,:], (self.dim,self.dim))
            pData2 = np.reshape(predMap, (self.dim,self.dim))
            surf = ax1.plot_surface(X, Y, pData)
            surf2 = ax2.plot_surface(X, Y, pData2)
                        
            
            ax3.imshow(image)
            if not download:
                plt.show()
            else:
                plt.savefig(label + "/" + fileName)
                

            return predMap.reshape(self.dim,self.dim), tmpMap.reshape(self.dim,self.dim)



    def findOutliers(self, model, images, maps, dim, numSave = 5, std = 0, filePath = "outliers_hsi"):
                numImages = images.shape[0]
                rmses = []
                for i in tqdm(range(numImages)):
                    image = np.copy(images[i])
                    image = self.addNoise(image, std)
                    tmpMap = maps[i]

                    predMap = model(image[None,:,:,:], training=True).numpy()

                    targetOne = self.normalize(tmpMap.reshape(48,48))*self.topNm
                    targetTwo = self.normalize(predMap.reshape(48,48))*self.topNm
                    rmses.append((sqrt(self.mean_squared_error(targetOne, targetTwo)), i))

    #             os.mkdir(os.path.join(filePath,"best"))
    #             os.mkdir(os.path.join(filePath,"worst"))

                best = heapq.nlargest(numSave, rmses)
                worst = heapq.nsmallest(numSave, rmses)
                for i in range(len(best)):
                    rmse, index = best[i]
                    print("BAD", rmse)
                    self.evaluateTrio(model, images, depthMaps, index, self.dim, download = True, noise = True, std = std, filePath = f'{filePath}/worst/{i}')
                for i in range(len(worst)):
                    rmse, index = worst[i]
                    print("GOOD", rmse)
                    self.evaluateTrio(model, images, depthMaps, index, self.dim, download = True, noise = True, std = std, filePath = f'{filePath}/best/{i}')

                return "Completed"



    def showMap(self,depthMap):

            fig = plt.figure(figsize = (5,5))
            ax1 = fig.add_subplot(projection='3d')

            x = np.arange(0, self.dim)
            y = np.arange(0, self.dim)
            X, Y = np.meshgrid(x, y)
            pData = np.reshape(depthMap[None,:,:,:], (self.dim,self.dim))
            surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)
            
            ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
            ax1.set_title('Patient Tear Film Thickness Profile')
            
            

            plt.show()


    def batchEvaluate(self,model, images, maps, noise = False, stds = [0.1], fileName = "cGan.csv"):
            rmse = 0
            numImages = images.shape[0]
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Noise Value", "RMSE"])
                for std in stds:
                    for i in range(numImages):
                        image = np.copy(images[i])
                        if noise:
                            image = self.addNoise(image, std)
                        tmpMap = maps[i]

                        predMap = model(image[None,:,:,:], training=True).numpy()

                        targetOne = self.normalize(tmpMap.reshape(48,48))*self.topNm
                        targetTwo = self.normalize(predMap.reshape(48,48))*self.topNm
                        rmse = rmse + sqrt(mean_squared_error(targetOne, targetTwo))
                    print(rmse/numImages)
                    writer.writerow([std, rmse/numImages])
                    rmse = 0
                    mae = 0
            return "Completed"
        
    def tester(self, resource):
        return resource
    
    def tester2(self):
        pool = Pool()                         # Create a multiprocessing Pool
        print(pool.map(self.tester, ["one", "two", "three"]))  # process data_inputs iterable with pool
            
        
        
    def fetchAverageRMSE(self, model, images,maps, std, overrideNm = True):

            
            rmse = 0
            numImages = images.shape[0]
            scale = 0
                
            if overrideNm:
                scale = 1000
            else:
                scale = self.topNm
                    
            for i in tqdm(range(numImages)):
                image = np.copy(images[i])
                image = self.addNoise(image, std)
                tmpMap = maps[i]

                predMap = model(image[None,:,:,:], training=True).numpy()
                
                
                
                targetOne = self.normalize(tmpMap.reshape(48,48))*scale
                targetTwo = self.normalize(predMap.reshape(48,48))*scale
                rmse = rmse + self.calc_rmse(targetOne, targetTwo)
            return rmse/numImages

    def normalize(self,data):
            m = np.max(data)
            mi = np.min(data)
            norm = (data - mi) / (m - mi)
            return norm
                
        
    
        
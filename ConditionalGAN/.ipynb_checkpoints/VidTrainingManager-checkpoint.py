#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from Model import Model
from DeepModel import DeepModel
from VideoGenerator import VideoGenerator
from Losses import Losses
from ModelEvaluation import ModelEvaluation
import time
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from Database import Database
import multiprocessing
import concurrent.futures
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing
from matplotlib import pyplot as plt
from matplotlib import cm
import shutil
import cv2

class VidTrainingManager:
    
    def __init__(self, trainingDict):
        self.label = trainingDict["name"]
        self.optimalCheckpoint = trainingDict["optimalCheckpoint"]

        models = DeepModel(trainingDict)
        self.generator, self.discriminator = self.extractModels(models)
             
        
        self.topNm = trainingDict["topNm"]
        self.numVid = trainingDict["numVid"]
        
        self.database = Database()
        self.database.uploadSession(trainingDict)
        self.vidGenerator = VideoGenerator(self.topNm)
        self.losses = Losses()
        self.evaluator = ModelEvaluation(self.topNm)
        
        self.generator_optimizer = tf.keras.optimizers.Adam(trainingDict["gen_lr"], beta_1=trainingDict["gen_b1"])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(trainingDict["disc_lr"], beta_1=trainingDict["disc_b1"])
        
        
        self.tIMages, self.tMaps = self.vidGenerator.fetchTestingSet()
        
        self.checkpoint_dir = self.label + "/checkpoints/"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,               discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        
        
        self.EPOCHS = trainingDict["epochs"]
        self.summary_writer = tf.summary.create_file_writer(
  self.label + "/graph/")
        

    @tf.function()
    def train_step(self,input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print("INPUT", input_image)
        print("MAP", target)
        gen_output = self.generator(input_image, training=True)
        print("GEN OUTPUT", gen_output)
        disc_real_output = self.discriminator([input_image, target], training=True)
        print("DISC OUTPUT 1", disc_real_output)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)
        print("DISC OUTPUT 2", disc_generated_output)
        gen_total_loss, gen_gan_loss, gen_l1_loss = self.losses.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = self.losses.discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   self.discriminator.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                              self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  self.discriminator.trainable_variables))

      with self.summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        

    def fit(self, startingPoint = 0):
    
      images, _, maps = self.vidGenerator.generateVideos(self.numVid)
      
      
      self.makeImageDirectory()
    

      for epoch in range(startingPoint, self.EPOCHS):
        start = time.time()
        
        # Train
        for n in range(images.shape[0]):
          image = images[n]
          tmpMap = maps[n].reshape((12, 144, 144, 1))

        
          self.train_step(image[None, :, :, :, :],tmpMap[None, :, :, :, :] , epoch)


        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
        self.saveModel()
        
      self.checkpoint.save(file_prefix = self.checkpoint_prefix)
      self.saveModel()
    
    

    

    
    def extractModels(self, models):
        return (models.Generator(), models.Discriminator())
    
    def fetchTestingRMSE(self, std):
        gRMSE = self.evaluator.fetchAverageRMSE(self.generator, self.gImages, self.gMaps, std)
        pRMSE = self.evaluator.fetchAverageRMSE(self.generator, self.pImages, self.pMaps, std)
        
        return (gRMSE, pRMSE)
    
    def makeImageDirectory(self):
        try:
            os.makedirs(os.path.join(self.label, "example_1"))
            os.makedirs(os.path.join(self.label, "example_2"))

        except OSError as error:  
            print(error)
            
            
    def runInference(self):
        images, _, maps = self.generateVideos(1)
        
        predMap = self.generator(images, training=True).numpy()
        

        
        
            
            
    def saveModel(self):
        self.generator.save(self.label + "/model")

        
    def previewGenerator(self):
        return tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)
    
    def previewDiscriminator(self):
        return tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)
    
    def restoreLatestCheckpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint("./" + self.checkpoint_dir))
        print("Restored Latest Checkpoint")
    
    def restoreCheckpoint(self, number):
        self.checkpoint.restore("./" + self.checkpoint_dir + "./ckpt-" + str(number))
        
    def restoreOptimalCheckpoint(self):
        self.checkpoint.restore("./" + self.checkpoint_dir + "./ckpt-" + str(self.optimalCheckpoint))



    def saveImages(imgSet):


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
            saveImage(imgSet[frame], frame)

        shutil.make_archive('tmp', 'zip', 'tmp/')

    def saveMaps(imgSet):


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
            saveMap(imgSet[frame], frame)

        shutil.make_archive('tmp', 'zip', 'tmp/')

    def saveEyeVideo(imgSet):
        saveImages(imgSet)
        image_folder = 'tmp'
        video_name = 'video1.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 4, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    def saveMapVideo(imgSet):
        saveMaps(imgSet)
        image_folder = 'tmp'
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        
    def showMap(depthMap, dim = 48):

        pData = np.reshape(depthMap, (dim,dim))

        fig = plt.figure(figsize = (5,5))
        ax1 = fig.add_subplot(projection='3d')

        x = np.arange(0, dim)
        y = np.arange(0, dim)
        X, Y = np.meshgrid(x, y)
        surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)

        ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
        ax1.set_title('Patient Tear Film Thickness Profile')



        return plt.show()

    def showImage(image, dim = 144):
        data = np.reshape(image, (dim,dim,3))
        plt.imshow(imRegulate(data), interpolation='nearest')
        return plt.show()

    def saveImage(image, fn, dim = 144):
        data = np.reshape(image, (dim,dim,3))
        plt.imshow(imRegulate(data), interpolation='nearest')
        plt.savefig(f'tmp/{fn}.png')

    def saveMap(image, fn, dim = 144):
        pData = np.reshape(image, (dim,dim))

        fig = plt.figure(figsize = (5,5))
        ax1 = fig.add_subplot(projection='3d')
        ax1.set_zlim(0, 3000)
        x = np.arange(0, dim)
        y = np.arange(0, dim)
        X, Y = np.meshgrid(x, y)
        surf = ax1.plot_surface(X, Y, pData, cmap = cm.coolwarm)

        ax1.set_zlabel("Thickness" + " (" + "nm" + ")")
        ax1.set_title('Patient Tear Film Thickness Profile')


        plt.savefig(f'tmp/{fn}.png')

    def imRegulate(data):
        m = np.max(data)
        mi = np.min(data)
        norm = ((data - mi) / (m - mi))*255
        return norm.astype(np.uint8)




        
        
        
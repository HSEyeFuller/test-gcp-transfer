#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from Model import Model
from DataGenerator import DataGenerator
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
from random import randint


class TrainingManager:
    
    def __init__(self, trainingDict, images, maps, tImages, tMaps):
        self.label = trainingDict["name"]
        self.hsi = trainingDict["hsi"]
        self.optimalCheckpoint = trainingDict["optimalCheckpoint"]
        self.std = trainingDict["std"]
        
        self.d1 = images.shape[1]
        self.d2 = images.shape[2]
        
        self.nTraining = images.shape[0]

        models = Model(trainingDict, self.d1, self.d2)
        self.generator, self.discriminator = self.extractModels(models)
             
        self.numPerlin = trainingDict["numPerlin"]
        self.numGaussian = trainingDict["numGaussian"]
        
        
        self.topNm = trainingDict["topNm"]


        
        self.database = Database()
        self.dataGenerator = DataGenerator(self.hsi, self.topNm)
        self.losses = Losses(topNm = self.topNm)
        self.evaluator = ModelEvaluation(self.d1, self.d2, topNm = self.topNm)
        
        self.generator_optimizer = tf.keras.optimizers.Adam(trainingDict["gen_lr"], beta_1=trainingDict["gen_b1"])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(trainingDict["disc_lr"], beta_1=trainingDict["disc_b1"])
        
        
#         self.gImages, self.gMaps, self.pImages, self.pMaps = self.dataGenerator.fetchTestingSet()

        self.images = images
        self.maps = maps
        
        self.tImages = tImages
        self.tMaps = tMaps
        self.nTesting = tMaps.shape[0]
        
        self.checkpoint_dir = self.label + "/checkpoints/"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,               discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)
        
        
        self.EPOCHS = trainingDict["epochs"]
        self.summary_writer = tf.summary.create_file_writer(
  self.label + "/graph/")
        
        self.training_losses = {'gen_total_loss': [], 'gen_gan_loss': [], 'gen_l1_loss': [], 'disc_loss': [], 'gen_true_l1_loss': []}
        

    @tf.function()
    def train_step(self,input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = self.generator(input_image, training=True)
        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss, gen_true_l1_loss = self.losses.generator_loss(disc_generated_output, gen_output, target)
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
        tf.summary.scalar('gen_true_l1_loss', gen_true_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        
        self.training_losses['gen_total_loss'].append(gen_total_loss)
        self.training_losses['gen_gan_loss'].append(gen_gan_loss)
        self.training_losses['gen_l1_loss'].append(gen_l1_loss)
        self.training_losses['gen_true_l1_loss'].append(gen_true_l1_loss)
        self.training_losses['disc_loss'].append(disc_loss)
        

    def fit(self, startingPoint = 0):
    
#       images, maps = self.dataGenerator.generateMixedNoise(self.numPerlin, self.numGaussian, self.std)
      
     
      
#       print("NUM TRAINING", numTraining)

      self.makeImageDirectory()
    
      imagesSubset = self.images
      mapsSubset = self.maps
      for epoch in range(startingPoint, self.EPOCHS):
        start = time.time()

        self.evaluator.evaluateTrio(self.generator, self.tImages, self.tMaps, randint(0,self.nTesting-1), fileName = "example_1/" + str(epoch), label = self.label)
        # Train
        for n in range(self.nTraining):
          image = imagesSubset[n]
          image = image[None,:,:,:]
          tmpMap = mapsSubset[n]
          tmpMap = tmpMap[None,:,:,:]
          self.train_step(image,tmpMap , epoch)
        print()


        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
        self.saveModel()
        
      self.checkpoint.save(file_prefix = self.checkpoint_prefix)
      self.saveModel()
    
    
    def runInference(self, index, std = 0.1):
        return self.evaluator.evaluateTrio(self.generator, self.tImages, self.tMaps, index)
    
        
    def fetchConsistencyPair(self, image):
        image = image.eval(session=tf.compat.v1.Session()) 
        print("IMAGE", type(image))
        prediction = self.dataGenerator.normalize(self.generator(image)*5000)
        print("PREDICITON", type(prediction))
        recoveredImage = self.dataGenerator.fetchImageFromDepth(prediction.reshape(48,48))
        print("RECOVERED", recoveredImage.shape)
        image = self.dataGenerator.normalize(image) * 5000
        recoveredImage = self.dataGenerator.normalize(recoveredImage) * 5000
        
        return (image, recoveredImage)

    
    def extractModels(self, models):
        return (models.Generator(), models.Discriminator())
    
    def fetchTestingRMSE(self):
        tRMSE = self.evaluator.fetchAverageRMSE(self.generator, self.tImages, self.tMaps)
        
        return (tRMSE)
    
    def makeImageDirectory(self):
        try:
            os.makedirs(os.path.join(self.label, "example_1"))
            # os.makedirs(os.path.join(self.label, "example_2"))

        except OSError as error:  
            print(error)
        
            
    def reNormalize(self, data):
        return data * (self.topNm/2) + self.topNm
            
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


    def runImageInference(self, filename):
        image = Image.open(filename)
        data = np.asarray(image)
        t0 = time.time()
        predMap = self.generator(data[None,:,:,:], training=True).numpy()
        t1 = time.time()
        
        print("Inference Time", str(t1-t0))

        self.evaluator.showMap(self.evaluator.normalize(predMap)*1000)
        return self.evaluator.normalize(predMap)*1000
    
    def findOptimalCheckpoint(self, upperLimit, noise = 0.1):
        
        prevRMSE = 0
        
        for checkpoint in range(1,upperLimit):
            self.restoreCheckpoint(checkpoint)
            tRMSE = self.fetchTestingRMSE()
               
            if checkpoint == 1:
                prevRMSE = tRMSE
            
            if tRMSE > prevRMSE:
                upperLimit = checkpoint - 1
                break
            
        
        
#         self.database.setOptimumCheckpoint(upperLimit, self.label)

        return upperLimit
    
    def evaluateCheckpoints(self, upperLimit, noise = 0.1):
        
        prevRMSE = 0
        
        for checkpoint in range(1,upperLimit):
            self.restoreCheckpoint(checkpoint)
            tRMSE = self.fetchTestingRMSE()
               
            print("CHECKPOINT", tRMSE)
        
        print("FINISHED ANALYSIS")



        
        
        
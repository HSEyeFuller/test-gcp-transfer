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
import multiprocessing
import concurrent.futures
from pathos.multiprocessing import ProcessingPool as Pool
import pathos.multiprocessing

class TrainingManager:
    
    def __init__(self, trainingDict, images, maps, tImages, tMaps, dataset = None):
        models = Model(trainingDict)
        topNm = trainingDict["topNm"]

        self.dataset = dataset
        
        self.label = trainingDict["name"] #name of training job
        self.generator, self.discriminator = self.extractModels(models) #models
             
        self.dataGenerator = DataGenerator(topNm) #define data generator
        self.losses = Losses() #define losses
        self.evaluator = ModelEvaluation(topNm) #define evaluator
        
        self.generator_optimizer = tf.keras.optimizers.Adam(trainingDict["gen_lr"], beta_1=trainingDict["gen_b1"]) #define generator optimizer
        self.discriminator_optimizer = tf.keras.optimizers.Adam(trainingDict["disc_lr"], beta_1=trainingDict["disc_b1"]) #define disciminator optimizer
        
        
        self.images = images #training images
        self.maps = maps #training maps
        
        self.tImages = tImages #testing imates
        self.tMaps = tMaps #testing maps
        
        self.checkpoint_dir = self.label + "/checkpoints/" #directory of checkpoints
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt") #checkpoint prefix
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,               discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator) #checkpoint storage
        self.EPOCHS = trainingDict["epochs"] #number epochs
        self.summary_writer = tf.summary.create_file_writer(
              self.label + "/graph/") #write summary to TensorBoard. 
        

        



    @tf.function()
    def train_step(self,input_image, target, epoch):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = self.generator(input_image, training=True)
        disc_real_output = self.discriminator([input_image, target], training=True)
        disc_generated_output = self.discriminator([input_image, gen_output], training=True)

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
        
        
    def tfFit(self):
      start = time.time()

      for step, (input_image, target) in self.dataset.repeat().take(self.EPOCHS).enumerate():
        if (step) % 1000 == 0:

          if step != 0:
            print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

          start = time.time()

          print(f"Step: {step//1000}k")

        self.train_step(input_image[tf.newaxis,...], target[tf.newaxis,...], step)

        # Training step
        if (step+1) % 10 == 0:
          print('.', end='', flush=True)


      
        

    def fit(self, startingPoint = 0):
      self.makeImageDirectory()
    
      for epoch in range(startingPoint, self.EPOCHS):
        start = time.time()

        
        
        #TODO --> change index evaluations. Need quick way to see performance (mosaic preferred)
        self.evaluator.evaluateTrio(self.generator, self.images, self.maps, 0, 48, fileName = "example_1/" + str(epoch), label = self.label) 
        self.evaluator.evaluateTrio(self.generator, self.images, self.maps, 1, 48, fileName = "example_2/" +str(epoch), label = self.label)
        
        # Training segment
        for n in range(self.images.shape[0]):
          image = self.images[n]
          image = image[None,:,:,:]
          tmpMap = self.maps[n]
          tmpMap = tmpMap[None,:,:,:]
          self.train_step(image, tmpMap, epoch)
        print()

        #save checkpoint
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
        
                                                            time.time()-start))
        #save model
        # self.saveModel()
      
    
    def fetchRange(self, array, message):
        print(message, np.min(array), np.max(array))
    
    #save a specific image from the provided testing set
    def runInference(self, index, std = 0.1):
        return self.evaluator.evaluateTrio(self.generator, self.tImages, self.tMaps, index)
    
    
    
    def extractModels(self, models):
        return (models.Generator(), models.Discriminator())
    
    
    def makeImageDirectory(self): #create directories
        try:
            os.makedirs(os.path.join(self.label, "example_1"))
            os.makedirs(os.path.join(self.label, "example_2"))

        except OSError as error:  
            print(error)
        
                
    def saveModel(self):
        self.generator.save(self.label + "/model")

        
    def previewGenerator(self):
        return tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)
    
    def previewDiscriminator(self):
        return tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)

    def runImageInference(self, filename):
        image = Image.open(filename)
        data = np.asarray(image)
        t0 = time.time()
        predMap = self.generator(data[None,:,:,:], training=True).numpy()
        t1 = time.time()
        
        print("Inference Time", str(t1-t0))

        self.evaluator.showMap(self.evaluator.normalize(predMap)*1000)
        return self.evaluator.normalize(predMap)*1000
    


        
        
        
#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class Losses:
    
    def __init__(self):
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.LAMBDA = 100
    

    def generator_loss(self, disc_generated_output, gen_output, target):
      disc_generated_output = tf.cast(disc_generated_output, tf.float32)
      gen_output = tf.cast(gen_output, tf.float32)
      target = tf.cast(target, tf.float32)


      gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

      # mean absolute error
      l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

      total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

      return total_gen_loss, gan_loss, l1_loss
    
    def cycle_loss(self, disc_generated_output, gen_output, target, image, recoverImage):
      disc_generated_output = tf.cast(disc_generated_output, tf.float32)
      gen_output = tf.cast(gen_output, tf.float32)
      target = tf.cast(target, tf.float32)
      image = tf.cast(image, tf.float32)
      recoverImage = tf.cast(recoverImage, tf.float32)



      gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

      # mean absolute error
      l1_loss_maps = tf.reduce_mean(tf.abs(target - gen_output))
      l1_loss_images = tf.reduce_mean(tf.abs(image - recoverImage))

      total_gen_loss = gan_loss + (self.LAMBDA * l1_loss_maps) + (self.LAMBDA * l1_loss_images)

      return total_gen_loss, gan_loss, l1_loss
    

    
    

    def discriminator_loss(self, disc_real_output, disc_generated_output):
      disc_real_output = tf.cast(disc_real_output, tf.float32)
      disc_generated_output = tf.cast(disc_generated_output, tf.float32)

      real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

      generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

      total_disc_loss = real_loss + generated_loss

      return total_disc_loss
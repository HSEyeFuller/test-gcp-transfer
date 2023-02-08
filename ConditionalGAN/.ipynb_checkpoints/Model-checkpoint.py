#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    
    def __init__(self, trainingDict):
        self.OUTPUT_CHANNELS = 1
        self.gen_down = trainingDict["gen_down"]
        self.gen_up = trainingDict["gen_up"]
        self.gen_dropout = trainingDict["gen_dropout"]
        self.disc_down = trainingDict["disc_down"]
        self.d1 = trainingDict["d1"]
        self.d2 = trainingDict["d2"]
            
        


    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
            ))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False,
            ))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[self.d1, self.d2, 3])
        
        down_stack = [self.downsample(128, 4, apply_batchnorm=False),
                      self.downsample(256, 4), self.downsample(512, 4)]
        
        down_stack = []
        up_stack = []
        
        for layer in self.gen_down:
            down_stack.append(self.downsample(layer, 4))
            
        for i in range(len(self.gen_up)):
            up_stack.append(self.upsample(self.gen_up[i],4,apply_dropout = self.gen_dropout[i]))


        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(
            self.OUTPUT_CHANNELS,
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh',
            )

        x = inputs

      # Downsampling through the model

        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections

        for (up, skip) in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.d1, self.d2, 1],
                                    name='input_image')
        tar = tf.keras.layers.Input(shape=[self.d1, self.d2, 3],
                                    name='target_image')

        x = tf.keras.layers.concatenate([inp, tar]) 
        
        for layer in self.disc_down:
            x = self.downsample(layer,4,False)(x)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(x)  
        conv = tf.keras.layers.Conv2D(64, 4, strides=1,
                kernel_initializer=initializer,
                use_bias=False)(zero_pad1) 

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                kernel_initializer=initializer)(zero_pad2)  

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    

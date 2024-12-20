"""
Borrowed from Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
import tensorflow.signal as tf_signal
import numpy as np
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self, fourier, fourier_only, random_fourier, combined, combined_random):
        super(YourModel, self).__init__()
        self.fourier = fourier
        self.random_fourier = random_fourier
        self.fourier_only = fourier_only
        self.combined = combined
        self.combined_random = combined_random

        print("Fourier:", self.fourier)
        print("Random fourier:", self.random_fourier)
        print("Fourier only:", self.fourier_only)
        print("Combined:", self.combined)
        print("Combined Random:", self.combined_random)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate)
        
        #fourier: conv_blocks, head
        #random-fourier: conv_blocks, head
        #fourier-only: fourier_head
        #combined: fourier_head, conv_blocks, head, combined

        if not fourier_only:
            self.conv_blocks = [
                # block 1
                Conv2D(64, 3, padding="same", activation=None, name="block1_conv1"),
                BatchNormalization(name="block1_bn1"),
                ReLU(name="block1_relu1"),
                Conv2D(64, 3, padding="same", activation=None, name="block1_conv2"),
                BatchNormalization(name="block1_bn2"),
                ReLU(name="block1_relu2"),
                MaxPool2D(2, name="block1_pool"),
                Dropout(0.3, name="block1_dropout"),
                
                # block 2
                Conv2D(128, 3, padding="same", activation=None, name="block2_conv1"),
                BatchNormalization(name="block2_bn1"),
                ReLU(name="block2_relu1"),
                Conv2D(128, 3, padding="same", activation=None, name="block2_conv2"),
                BatchNormalization(name="block2_bn2"),
                ReLU(name="block2_relu2"),
                MaxPool2D(2, name="block2_pool"),
                Dropout(0.3, name="block2_dropout"),
                
                # block 3
                Conv2D(256, 3, padding="same", activation=None, name="block3_conv1"),
                BatchNormalization(name="block3_bn1"),
                ReLU(name="block3_relu1"),
                Conv2D(256, 3, padding="same", activation=None, name="block3_conv2"),
                BatchNormalization(name="block3_bn2"),
                ReLU(name="block3_relu2"),
                MaxPool2D(2, name="block3_pool"),
                Dropout(0.3, name="block3_dropout"),
                
                # block 4
                Conv2D(512, 3, padding="same", activation=None, name="block4_conv1"),
                BatchNormalization(name="block4_bn1"),
                ReLU(name="block4_relu1"),
                Conv2D(512, 3, padding="same", activation=None, name="block4_conv2"),
                BatchNormalization(name="block4_bn2"),
                ReLU(name="block4_relu2"),
                MaxPool2D(2, name="block4_pool"),
                Dropout(0.3, name="block4_dropout"),
            ]
            self.conv_blocks = tf.keras.Sequential(self.conv_blocks, name="conv_base")

        
        #if fourier only or combined:
        if fourier_only:
            self.fourier_head = [
                Dense(1024, activation="relu", name="fc1"),
                Dropout(0.3, name="dropout1"),
                Dense(512, activation="relu", name="fc2"),
                Dropout(0.3, name="dropout2"),
                Dense(256, activation="relu", name="fc3"),
                Dropout(0.3, name="dropout3"),
                Dense(128, activation="relu", name="fc4"),
                Dropout(0.3, name="dropout4"),
                Dense(1, activation="sigmoid", name="output"),
            ]
            self.fourier_head = tf.keras.Sequential(self.fourier_head, name="fourier_head")
        
        elif combined or combined_random:
            self.head = [
                Dense(512, activation="relu", name="fc1"),
                Dropout(0.3, name="dropout1"),
                Dense(512, activation="relu", name="fc3"),  
            ]
            self.fourier_head = [
                Dense(1024, activation="relu", name="fc1"),
                Dropout(0.3, name="dropout1"),
                Dense(512, activation="relu", name="fc2"),
                Dropout(0.3, name="dropout2"),
                Dense(256, activation="relu", name="fc3"),  
            ]
            self.combined_head = [
                Dense(256, activation="relu", name="fc1"),
                Dropout(0.3, name="dropout1"),
                Dense(1, activation="sigmoid", name="output")
            ]
            self.head = tf.keras.Sequential(self.head, name="head")
            self.fourier_head = tf.keras.Sequential(self.fourier_head, name="fourier_head")
            self.combined_head = tf.keras.Sequential(self.combined_head, name="combined_head")

        else:
            self.head = [
                Dense(512, activation="relu", name="fc1"),
                Dropout(0.3, name="dropout1"),
                Dense(512, activation="relu", name="fc2"),
                Dropout(0.3, name="dropout2"),
                Dense(1, activation="sigmoid", name="output")
            ]
            self.head = tf.keras.Sequential(self.head, name="head")

    def apply_fourier_transform(self, x):
        """ Applies Fourier Transform to the input tensor. """
        x = tf.cast(x, tf.float32) 
        x = tf_signal.rfft2d(x)
        x_mag = tf.abs(x) 
        x_phase = tf.math.angle(x)
        x_mag_pooled = tf.reduce_mean(x_mag, axis=-1)
        x_phase_pooled = tf.reduce_mean(x_phase, axis=-1)
        x_mag_flattened = tf.keras.layers.Flatten()(x_mag_pooled)
        x_phase_flattened = tf.keras.layers.Flatten()(x_phase_pooled)
        return x_mag_flattened, x_phase_flattened

    def call(self, x):
        conv_output_func = tf.keras.layers.GlobalAveragePooling2D(name="gap_conv_output")
        # conv_output_func = tf.keras.layers.Flatten()

        # concatenated Fourier
        if self.fourier:
            x_mag_flattened, x_phase_flattened  = self.apply_fourier_transform(x)

            conv_output = self.conv_blocks(x)
            conv_output_gapped = conv_output_func(conv_output)

            combined_features = tf.keras.layers.Concatenate()([conv_output_gapped, x_mag_flattened, x_phase_flattened])
            x = self.head(combined_features)

        # concatenated random (instead of Fourier)
        elif self.random_fourier:
            uniform_noise = tf.random.uniform(shape=tf.shape(x), minval=0, maxval=255, dtype=tf.float32)
            x_mag_flattened, x_phase_flattened  = self.apply_fourier_transform(uniform_noise)

            conv_output = self.conv_blocks(x)
            conv_output_gapped = conv_output_func(conv_output)

            combined_features = tf.keras.layers.Concatenate()([conv_output_gapped, x_mag_flattened, x_phase_flattened])
            x = self.head(combined_features)

        elif self.fourier_only: 
            x_mag_flattened, x_phase_flattened = self.apply_fourier_transform(x)

            combined_features = tf.keras.layers.Concatenate()([x_mag_flattened, x_phase_flattened])
            x = self.fourier_head(combined_features)

        elif self.combined:
            x_mag_flattened, x_phase_flattened = self.apply_fourier_transform(x)

            combined_features = tf.keras.layers.Concatenate()([x_mag_flattened, x_phase_flattened])
            x_fourier = self.fourier_head(combined_features)

            conv_output = self.conv_blocks(x)
            conv_output_gapped = conv_output_func(conv_output)
            x_cnn = self.head(conv_output_gapped)

            combined_arch = tf.keras.layers.Concatenate()([x_fourier, x_cnn])
            x = self.combined_head(combined_arch)

        elif self.combined_random:
            uniform_noise = tf.random.uniform(shape=tf.shape(x), minval=0, maxval=255, dtype=tf.float32)
            x_mag_flattened, x_phase_flattened  = self.apply_fourier_transform(uniform_noise)

            combined_features = tf.keras.layers.Concatenate()([x_mag_flattened, x_phase_flattened])
            x_fourier = self.fourier_head(combined_features)

            conv_output = self.conv_blocks(x)
            conv_output_gapped = conv_output_func(conv_output)
            x_cnn = self.head(conv_output_gapped)

            combined_arch = tf.keras.layers.Concatenate()([x_fourier, x_cnn])
            x = self.combined_head(combined_arch)
            
        else: 
            x = self.conv_blocks(x)
            x = conv_output_func(x)
            x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for binary classification. """
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)

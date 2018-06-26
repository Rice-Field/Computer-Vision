# Jonathon Rice, CAP 5415
# python 3.6
# Tensorflow and TFLearn
# This file loads previous weights and produces images
# Uncomment bottom section to continue training

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import *
from tflearn.data_utils import *

# Load CIFAR-10 Dataset
# If no dataset, program should download it

# X, Y --> Training Data
# X_test, Y_test --> Testing Data
# Load CIFAR-10 Dataset
from tflearn.datasets import cifar10
(X, Y), (testX, testY) = cifar10.load_data()

# Scale range of input to [-1,1]
X += -(np.min(X))
X /= np.max(X) / (1 - -1)
X += -1

# print(X)

# shuffle the data in unison
X, Y = tflearn.data_utils.shuffle(X, Y)

image_dim = 3072 # 32*32 pixels
z_dim = 100 # Noise data points
total_samples = len(X)


# Generator begins with noise and creates a 4x4 image at 128 filters
# each deconvolution (transpose) has a stride of 2 that upsamples the image
# Outputs 32x32, 3 channel image
def generator(z, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        gnet = tflearn.fully_connected(z, n_units= 4 * 4 * 128, activation=None)
        gnet = tflearn.batch_normalization(gnet)
        gnet = tflearn.activations.relu(gnet)
        gnet = tf.reshape(gnet, shape=[-1, 4, 4, 128])

        gnet = tflearn.conv_2d_transpose(gnet, 64, 5, [8,8], strides=2, activation='linear')
        gnet = tflearn.batch_normalization(gnet)
        gnet = tflearn.activations.relu(gnet)

        gnet = tflearn.conv_2d_transpose(gnet, 32, 5, [16,16], strides=2, activation='linear')
        gnet = tflearn.batch_normalization(gnet)
        gnet = tflearn.activations.relu(gnet)

        gnet = tflearn.conv_2d_transpose(gnet, 3, 5, [32,32], strides=2, activation='tanh')
        return gnet


# Discriminator has no fully connected networks besides the single neuron output
# output predicts if image is fake or real
# uses strides in convolution to pool
def discriminator(img, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        dnet = tflearn.conv_2d(img, 32, 5, strides=2, activation=None)
        dnet = tflearn.batch_normalization(dnet)
        dnet = tflearn.activations.leaky_relu(dnet, alpha=0.2)

        dnet = tflearn.conv_2d(dnet, 64, 5, strides=2, activation=None)
        dnet = tflearn.batch_normalization(dnet)
        dnet = tflearn.activations.leaky_relu(dnet, alpha=0.2)

        dnet = tflearn.conv_2d(dnet, 128, 5, strides=2, activation=None)
        dnet = tflearn.batch_normalization(dnet)
        dnet = tflearn.activations.leaky_relu(dnet, alpha=0.2)

        dnet = fully_connected(dnet, 1, activation='sigmoid')
        return dnet

# Set input sizes for both networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, 32, 32, 3], name='disc_input')

# Generate fake images
gen_sample = generator(gen_input)
# Input real and fake images into discriminator
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# Lower the learning rate and Beta1 for adam optimizer
adamgan = tflearn.optimizers.Adam (learning_rate=0.0002, beta1=0.5)

# The training operations for the two networks
# We define two variables to keep each network separate in training
# Both input the same learning rate
gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gen_model = tflearn.regression(gen_sample, placeholder=None, optimizer=adamgan,
            loss=gen_loss, trainable_vars=gen_vars,
            batch_size=64, name='target_gen', op_name='GEN')

disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(disc_real, placeholder=None, optimizer=adamgan,
            loss=disc_loss, trainable_vars=disc_vars,
            batch_size=64, name='target_disc', op_name='DISC')

# Define GAN model, that outputs the generated images.
gan = tflearn.DNN(gen_model, checkpoint_path='gan-cifar.tfl.ckpt')

# Load the model
gan.load("gan-cifar.tfl.ckpt-23460")

# Uncomment below to allow further training of saved networks

# # Training
# # Generate noise to feed to the generator
# z = np.random.uniform(-1., 1., size=[total_samples, z_dim])

# # Start training, feed both noise and real images.
# gan.fit(X_inputs={gen_input: z, disc_input: X},
#         Y_targets=None, n_epoch=1, run_id='gan-cifar')

# # Save the model
# gan.save("gan-cifar.tfl.ckpt")

# Generate images from noise, using the generator network.
f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    for j in range(4):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        # Generate image from noise
        temp = gan.predict([z])
        # Scale range to 0-1
        temp += -(np.min(temp))
        temp /= np.max(temp)
        a[j][i].imshow(np.reshape(temp, (32, 32, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
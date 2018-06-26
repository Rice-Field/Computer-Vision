from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import *
from tflearn.data_utils import *

# Load data from pkl file, the data used is a combination of cifar10
# X, Y --> Training Data
# X_test, Y_test --> Testing Data
# Load CIFAR-10 Dataset
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data()

# X = np.repeat(X[:, :, np.newaxis], 3, axis=2)
# X = np.reshape(X, newshape=[-1, 28, 28, 3])

X += -(np.min(X))
X /= np.max(X) / (1 - -1)
X += -1

# print(X)

# shuffle the data in unison
X, Y = tflearn.data_utils.shuffle(X, Y)

image_dim = 784 # 28*28 pixels
z_dim = 100 # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):

        x = tflearn.fully_connected(x, n_units= 512, activation='relu')
        x = dropout(x, 0.4)
        x = tflearn.fully_connected(x, n_units= image_dim, activation='tanh')

        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):

        x = fully_connected(x, 512, activation='leaky_relu')
        x = dropout(x, 0.4)
        x = fully_connected(x, 1, activation='sigmoid')

        return x

# Build Networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, image_dim], name='disc_input')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# Build Training Ops for both Generator and Discriminator.

adamgan = tflearn.optimizers.Adam (learning_rate=0.0002, beta1=0.5)
# momentumd = tflearn.optimizers.Momentum(learning_rate=0.0002, momentum=0.5, lr_decay=.00001)

gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gen_model = tflearn.regression(gen_sample, placeholder=None, optimizer='adam',
            loss=gen_loss, trainable_vars=gen_vars,
            batch_size=64, name='target_gen', op_name='GEN')

disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(disc_real, placeholder=None, optimizer='adam',
            loss=disc_loss, trainable_vars=disc_vars,
            batch_size=64, name='target_disc', op_name='DISC')

# Define GAN model, that output the generated images.
gan = tflearn.DNN(gen_model, checkpoint_path='gan-mnist.tfl.ckpt')

# Load the model
gan.load("gan-mnist.tfl")

# # Training
# # Generate noise to feed to the generator
# z = np.random.uniform(-1., 1., size=[total_samples, z_dim])

# # Start training, feed both noise and real images.
# gan.fit(X_inputs={gen_input: z, disc_input: X},
#         Y_targets=None, n_epoch=100, run_id='gan-mnist')

# # Save the model
# gan.save("gan-mnist.tfl")

# Generate images from noise, using the generator network.
f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    for j in range(4):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        # Generate image from noise. Extend to 3 channels for matplot figure.
        temp = gan.predict([z])
        temp += -(np.min(temp))
        temp /= np.max(temp)
        temp = np.reshape(np.repeat(temp[:, :, np.newaxis], 3, axis=2),
                        newshape=(28, 28, 3))
        a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
from __future__ import division, print_function, absolute_import

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow import losses
import os
import math

from detgenhist import detGen

tf.reset_default_graph()

# select number of matrices to generate
matrices = 100000
# uncomment all matrices = xmax and ymax if on 12
maxorno = 4 # 12 to include max vects, 10 to not
dim = 4 
matrix_dim = 16 # 120 if 12, otherwise 100

A = np.load('affinity.npy')
B = np.load('solutions.npy')

# Scale range of input to [-1,1]
B = B.astype(float)
B += -(np.min(B.astype(float)))
B /= np.max(B) / (1 - -1)
B += -1

# Network Params
image_dim = 16
noise_dim = 100 # Noise data points

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    print('dataset shuffled')
    return a, b

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def sample_Z(m, n):
    return np.random.normal(0., .25, size=[m, n])

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def plot(samples):
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(4, 4), cmap='Greys_r')

    return fig

def cnn(x, reuse=False):
    with tf.variable_scope('cnn', reuse=reuse):

        cnet = tf.reshape(x, [-1, dim, dim, 1])

        cnet = tf.layers.conv2d(cnet, 128, 4, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(False))
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        cnet = tf.layers.conv2d(cnet, 64, 3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(False))
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        cnet = tf.layers.conv2d(cnet, 32, 2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(False))
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        cnet = tf.layers.conv2d(cnet, 16, 2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(False))
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        # cnet = tf.layers.dense(cnet, 16)
        # cnet = tf.nn.relu(cnet)
        # cnet = tf.reshape(cnet, [-1, 4, 4, 1])

        cnet = tf.layers.conv2d_transpose(cnet, 32, 4, strides=1, padding='same')
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        cnet = tf.layers.conv2d_transpose(cnet, 16, 3, strides=1, padding='same')
        cnet = tf.contrib.layers.batch_norm(inputs=cnet, center=True, scale=True, is_training=True)
        cnet = lrelu(cnet)

        cnet = tf.layers.conv2d_transpose(cnet, 1, 2, strides=1, padding='same')
        cnet = tf.nn.tanh(cnet)

        return cnet


# Build Networks
# Network Inputs
graph = tf.Graph()
with graph.as_default():
    affinity_matrix_input = tf.placeholder(tf.float32, shape=[None, 16])
    binary_matrix_input = tf.placeholder(tf.float32, shape=[None, 16])

    # Build Generator Network
    gen_sample = cnn(affinity_matrix_input)


    # gen_label = tf.ones_like(disc_fake)

    gen_loss2 = tf.losses.huber_loss(binary_matrix_input, tf.contrib.layers.flatten(gen_sample))
    # gen_loss2 = tf.losses.mean_squared_error(binary_matrix_input, tf.contrib.layers.flatten(gen_sample))
    # gen_loss2 = tf.losses.huber_loss(binary_matrix_input, gen_sample)
    gen_loss = gen_loss2

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)

    # Generator Network Variables
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    init = tf.global_variables_initializer()

if not os.path.exists('autoencodertest/'):
    os.makedirs('autoencodertest/')

if not os.path.exists('saves/'):
    os.makedirs('saves/')

# Initialize the variables (i.e. assign their default value)

# Add ops to save and restore all the variables.
with tf.Session(graph=graph) as sess:
    sess.run(init)
    saver = tf.train.Saver()

    tf.get_default_graph().finalize()

    num_steps = 3124
    # num_steps = 100

    batch_size = 32
    correct = 0

    saver.restore(sess, "./saves/autoencoder.ckpt")
    print("Model restored.")

    F = np.loadtxt('pickframe.txt')

    A = np.zeros([4,4], dtype=float)
    imgtable = np.zeros([8, 100, 50, 3], dtype=np.uint8)
    # for i in range():
    startframe = F[0]
    endframe = F[1]
    j = 0
    for cframe, cBB, cXY, chist, frame in detGen(startframe, endframe):
        for k in range(len(cframe[0])):
            imgtable[(j*4)+k] = cframe[0,k]
        j += 1
        

    # for i in range(matrices):
    # for i in range(2):
        affinity = np.reshape(A[i], [1,16])
        samples = sess.run(gen_sample, feed_dict={affinity_matrix_input: affinity})
        answer = np.reshape(np.around(samples), (4,4))
        if np.array_equal(answer,B[i]):
            correct += 1
        # print(B[i])
        # print(answer)
        # print()

        # if i % 500 == 0:
        #     fig = plot(affinity.astype(np.float32))
        #     plt.savefig('autoencodertest/affinity{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        #     plt.close(fig)

        #     sol = np.reshape(B[i], [1,16])
        #     fig = plot(sol.astype(np.float32))
        #     plt.savefig('autoencodertest/GT{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        #     plt.close(fig)

        #     answer = np.reshape(answer, [1,16])
        #     fig = plot(answer.astype(np.float32))
        #     plt.savefig('autoencodertest/out{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        #     plt.close(fig)

    percent = correct/100000.0
    print("number correct %d" % correct)
    print("percent %g" % percent)

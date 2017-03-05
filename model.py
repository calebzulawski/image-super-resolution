#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def gaussian_2d(size, sigma):
    """
    Returns a Gaussian kernel of shape `size` x `size` with standard deviation `sigma`.
    """

    w = (size - 1.)/2.
    y,x = np.ogrid[-w:w+1, -w:w+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussian_kernel(channels, size, sigma):
    """
    Returns a Gaussian kernel of size [size, size, channels, 1] with standard deviation `sigma`.
    """

    kernel = gaussian_2d(size, sigma)
    kernel = np.repeat(kernel[:, :, np.newaxis], channels, axis=2)
    return np.expand_dims(kernel, 3)

def produce_low_resolution(input, k=3, blur_size=3, blur_sigma=0.5):
    """
    Produces a batch of low resolution images from the high resolution images `input`.
    The images are produced by applying a Gaussian blur with kernel size `blur_size` x `blur_size`
    and standard deviation `blur_sigma`, downsampling by `k`, and applying bicubic interpolation
    up to the size of the input batch.
    """

    n_channels = input.get_shape().as_list()[3];

    # Apply Gaussian blur
    kernel = gaussian_kernel(n_channels, blur_size, blur_sigma)
    lr = tf.nn.depthwise_conv2d_native(input, kernel, [1, 1, 1, 1], 'VALID')

    # Downsample the image
    lr = tf.nn.depthwise_conv2d_native(lr, tf.ones([1, 1, n_channels, 1]), [1, k, k, 1], 'VALID')

    # Apply bicubic interpolation
    lr = tf.image.resize_bicubic(lr, input.get_shape().as_list()[1:3])

    # Apply clamping/quantization
    lr = tf.fake_quant_with_min_max_args(lr, min=0, max=1)

    return lr

class Model():
    def __init__(self, batch_size=10, fsub=33, n_channels=3, f1=9, f2=1, f3=5, n1=64, n2=32, is_training=True):
        self.batch_size = batch_size
        self.fsub = fsub
        self.n_channels = n_channels
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.n1 = n1
        self.n2 = n2
        self.is_training = is_training

    def build_model(self):
        self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.fsub, self.fsub, self.n_channels])

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.001),
                            weights_regularizer=slim.l2_regularizer(0.002)):
            net = self.input
            net = slim.conv2d(net, self.n1, [self.f1, self.f1], padding='VALID', scope='conv1')
            net = slim.conv2d(net, self.n2, [self.f2, self.f2], padding='VALID', scope='conv2')
            net = slim.conv2d(net, self.input.get_shape()[3], [self.f3, self.f3], padding='VALID', scope='conv3')
        self.output = net

        low_res = produce_low_resolution(self.input)
        self.loss = tf.nn.l2_loss(self.input - low_res)

if __name__ == "__main__":
    print("Testing model...")
    m = Model()
    m.build_model()
    check = tf.add_check_numerics_ops()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    images = np.random.uniform(size=[10, 33, 33, 3])
    output, loss, check = sess.run([m.output, m.loss, check], feed_dict={m.input: images})
    print("Input has shape:  {}".format(images.shape))
    print("Output has shape: {}".format(output.shape))
    print("Loss: {}".format(loss))

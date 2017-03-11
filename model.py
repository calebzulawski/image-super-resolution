#!/usr/bin/env python

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensor_ops import center_crop_to_size, produce_low_resolution

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
        if self.is_training:
            self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.fsub, self.fsub, self.n_channels])
        else:
            self.input = tf.placeholder(tf.float32, shape=[None, None, None, self.n_channels])

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.001),
                            weights_regularizer=slim.l2_regularizer(0.002)):
            net = self.input
            net = slim.conv2d(net, self.n1, [self.f1, self.f1], padding='VALID', scope='conv1')
            net = slim.conv2d(net, self.n2, [self.f2, self.f2], padding='VALID', scope='conv2')
            net = slim.conv2d(net, self.input.get_shape()[3], [self.f3, self.f3], padding='VALID', scope='conv3')
        self.output = net

        if self.is_training:
            input_cropped = center_crop_to_size(self.input, self.output.get_shape().as_list())
            self.loss = tf.nn.l2_loss(self.output - input_cropped)
            for reg_loss in tf.losses.get_regularization_losses():
                self.loss += reg_loss

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

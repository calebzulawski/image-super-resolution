#!/usr/bin/env python

import time
import tensorflow as tf
import validation
from tensor_ops import center_crop_to_size

class Trainer():
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model

        self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

    def train_iter(self, batcher, validate=False):
        if validate:
            # There's probably a better way to do this than getting a batch, then feeding it back into the model
            batch, batch_cropped = self.sess.run([batcher, center_crop_to_size(batcher, self.model.output.get_shape().as_list())])
            _, loss, psnr = self.sess.run([self.optimizer, self.model.loss, validation.psnr(batch_cropped, self.model.output)], feed_dict={self.model.input: batch})
            print([loss, psnr])
        else:
            batch = self.sess.run(batcher)
            self.sess.run(self.optimizer, feed_dict={self.model.input: batch})

    def train(self, batcher):
        for i in range(100000):
            self.train_iter(batcher, i % 100 == 0)

if __name__ == '__main__':
    import model
    from files import FileReader
    m = model.Model()
    m.build_model()
    with tf.Session() as sess:
        t = Trainer(sess, m)
        f = FileReader('./images/crop_256/*/*.JPEG', (33, 33), batch_size=10)
        tf.global_variables_initializer().run()
        f.start_queue_runners()
        t.train(f.get_batch())
        f.stop_queue_runners()

#!/usr/bin/env python

import time
import tensorflow as tf
import validation
from tensor_ops import center_crop_to_size

save_path = "./saved_model/model.ckpt"

class Trainer():
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(model.loss)

    def train_iter(self, batcher, validate=False, saver=None, path=None, val=None):
        if validate:
            # There's probably a better way to do this than getting a batch, then feeding it back into the model
            batch = self.sess.run(val)
            _, loss, psnr = self.sess.run([self.optimizer, self.model.loss, validation.gain(self.model.input, self.model.output)], feed_dict={self.model.input: batch})
            print('Loss: {:.2f}\tGain compared to bicubic interpolation: {:.2f} dB'.format(loss, psnr))
            if saver and path:
                sp = saver.save(self.sess, path)
        else:
            batch = self.sess.run(batcher)
            self.sess.run(self.optimizer, feed_dict={self.model.input: batch})

    def train(self, batcher, saver=None, path=None, val=None):
        if val is None:
            val = batcher
        for i in range(100000):
            self.train_iter(batcher, i % 2 == 0, saver=saver, path=path, val=val)

if __name__ == '__main__':
    batch_size = 512
    import model
    from files import FileReader
    m = model.Model(batch_size=batch_size)
    m.build_model()
    with tf.Session() as sess:
        t = Trainer(sess, m)
        f = FileReader('./images/sets/train/*.JPEG', (33, 33), batch_size=batch_size)
        v = FileReader('./images/sets/validation/*.JPEG', (33, 33), batch_size=batch_size)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        try:
            saver.restore(sess,save_path)
        except:
            print('Error while restoring');
        f.start_queue_runners()
        v.start_queue_runners()
        t.train(f.get_batch(), val=v.get_batch())
        v.stop_queue_runners()
        f.stop_queue_runners()

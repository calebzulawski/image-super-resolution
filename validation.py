#!/usr/bin/env python

import tensorflow as tf

def psnr(original, modified):
    """
    Returns the peak signal-to-noise ratio between a tensor and its original, normalized
    against a maximum value of 1.
    """
    mse = tf.reduce_mean(tf.squared_difference(original, modified))
    numerator = -10 * tf.log(mse)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator # equivalent to log10(mse)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    with tf.Session() as sess:
        import files
        import model

        batch_size = 20
        crop_shape = (33, 33)

        # Read a batch and create the low resolution versions
        f = files.FileReader('./images/crop_256/*/*.JPEG', crop_shape, batch_size=batch_size)
        batch = f.get_batch()
        lowres = model.produce_low_resolution(batch)
        tf.global_variables_initializer().run()

        f.start_queue_runners()
        orig, lr, calc_psnr = sess.run([batch, lowres, psnr(batch, lowres)])
        f.stop_queue_runners()

        # Plot the batch of images
        fig = plt.figure()
        for index in range(batch_size):
            axes = fig.add_subplot(4, batch_size/2, index * 2 + 1)
            axes.set_axis_off()
            axes.imshow(orig[index], interpolation='nearest')
            axes = fig.add_subplot(4, batch_size/2, index * 2 + 2)
            axes.set_axis_off()
            axes.imshow(lr[index], interpolation='nearest')
        fig.suptitle('Batch of {} sub-images and their low resolution bicubic interpolated counterparts\nAverage PSNR = {:.2f}'.format(batch_size, calc_psnr), fontsize=20)
        plt.show()

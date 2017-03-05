#!/usr/bin/env python

import tensorflow as tf

class FileReader():
    """
    FileReader manages all the queueing involved in loading images from files, taking random crops,
    and forming batches.
    """

    def __init__(self, glob, crop_shape, batch_size=10):
        """
        Creates a FileReader that matches image filenames with `glob`,
        crops them to `crop_shape`, and produces batches of size `batch_size`.
        """
        # Create a queue of all the filenames
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(glob), shuffle=True)

        # Create a reader and decode a jpeg image
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file, channels=3)

        # Crop the image to the desired size
        crop_shape = tf.concat([crop_shape, [3]], 0)
        cropped = tf.random_crop(image, crop_shape)

        # Create a batch
        self.batch = tf.train.batch([cropped], batch_size=batch_size)

    def get_batch(self):
        """
        Returns a batch.
        """
        return self.batch

    def start_queue_runners(self):
        """
        Starts up the queue runner threads.  Must be called before executing.
        """
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def stop_queue_runners(self):
        """
        Stops the queue runner threads.  Should be called when finished executing.
        """
        self.coord.request_stop()
        self.coord.join(self.threads)

if __name__ == "__main__":
    with tf.Session() as sess:
        from matplotlib import pyplot as plt
        import numpy as np

        # Display M x N images, for a total batch size of M*N
        M = 5
        N = 10
        batch_size = M*N
        crop_shape = (128, 128)

        # Open the file reader and generate 1 batch
        f = FileReader('./images/crop_256/*/*.JPEG', crop_shape, batch_size=batch_size)
        tf.global_variables_initializer().run()
        f.start_queue_runners()
        batch = sess.run(f.get_batch())
        f.stop_queue_runners()

        # Plot the batch of images
        fig = plt.figure()
        for m in range(M):
            for n in range(N):
                index = m*N + n
                axes = fig.add_subplot(M, N, index + 1)
                axes.set_axis_off()
                axes.imshow(batch[index], interpolation='nearest')
        fig.suptitle('Batch of {} sub-images of shape {}'.format(batch_size, crop_shape), fontsize=20)
        plt.show()

import tensorflow as tf
import numpy as np

def center_crop_to_size(input, size):
    input_shape = input.get_shape().as_list()
    begin = []
    for i, s in zip(input_shape, size):
        begin.append(int((i-s)/2))
    return tf.slice(input, begin, size)

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

    # Apply clipping and quantization
    lr = tf.clip_by_value(lr, 0, 1)
    lr = tf.fake_quant_with_min_max_args(lr, min=0, max=1)

    return lr

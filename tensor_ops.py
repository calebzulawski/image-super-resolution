import tensorflow as tf

def center_crop_to_size(input, size):
    input_shape = input.get_shape().as_list()
    begin = []
    for i, s in zip(input_shape, size):
        begin.append(int((i-s)/2))
    return tf.slice(input, begin, size)

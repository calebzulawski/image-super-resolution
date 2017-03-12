#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    import sys
    import tensorflow as tf
    from model import Model

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, help='operating mode (train or generate)')
    parser.add_argument('--model', type=str, default='./saved_model/model.ckpt')
    parser.add_argument('--input', type=str, nargs='+', default=['./input/input.JPEG'])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--subimage-size', type=int, default=33)

    args = parser.parse_args()

    def reshape_for_output(image):
        shape = image.shape
        if len(shape) > 3:
            shape = shape[1:]
        return image.reshape(shape)*255

    def write_image_to_file(image, filename):
        image = reshape_for_output(image)
        images_out = tf.image.encode_jpeg(image)
        fh = open(filename, "wb+")
        fh.write(images_out.eval())
        fh.close()


    if args.mode == 'train':
        from train import Trainer
        from files import FileReader
        m = Model(batch_size=args.batch_size)
        m.build_model()
        with tf.Session() as sess:
            t = Trainer(sess, m)
            f = FileReader('./images/sets/train/*.JPEG', (args.subimage_size, args.subimage_size), batch_size=args.batch_size)
            v = FileReader('./images/sets/validation/*.JPEG', (args.subimage_size, args.subimage_size), batch_size=args.batch_size)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            try:
                saver.restore(sess, args.model)
            except:
                print('No save file found.  Creating new file at {}'.format(args.model));
            f.start_queue_runners()
            v.start_queue_runners()
            t.train(f.get_batch(), saver=saver, path=args.model, val=v.get_batch())
            v.stop_queue_runners()
            f.stop_queue_runners()
    elif args.mode == 'generate':
        if args.input is None:
            print("must provide an input file in generate mode")
            sys.exit(1)
        from files import FileReader
        from validation import produce_low_resolution as plr
        m = Model(is_training=False)
        m.build_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            try:
                saver.restore(sess, args.model)
            except:
                print('Could not load model file: {}!'.format(args.model))
                sys.exit(1)
            # Generate new images here
            filename_queue = tf.train.string_input_producer(args.input)
            image_reader = tf.WholeFileReader()
            _, image_file = image_reader.read(filename_queue)
            image = tf.image.decode_jpeg(image_file, channels=3)
            # Crop the image to the desired size
            crop_shape = tf.concat([(256,256), [3]], 0)
            cropped = tf.random_crop(image, crop_shape)
            cropped = tf.cast(cropped, tf.float32)/255.

            batch = tf.train.batch([cropped], batch_size=1)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            output, bicubic = sess.run([m.image_output, m.bicubic],  feed_dict={m.input: batch.eval()})

            coord.request_stop()
            coord.join(threads)
            # Calculate PSNR gain for each one
            write_image_to_file(output, './outputs/output.jpg')

            write_image_to_file(bicubic, './outputs/bicubic.jpg')
    else:
        print('Invalid "mode": {}!'.format(args.mode))
    sys.exit(0)

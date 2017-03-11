#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    import sys
    import tensorflow as tf
    from model import Model

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, help='operating mode (train or generate)')
    parser.add_argument('--model', type=str, default='./saved_model/model.ckpt')
    parser.add_argument('--input', type=str, nargs='+')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--subimage-size', type=int, default=33)

    args = parser.parse_args()

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
            filename_queue = tf.train.string_input_producer(['./input/input.JPEG'])
            image_reader = tf.WholeFileReader()
            _, image_file = image_reader.read(filename_queue)
            image = tf.image.decode_jpeg(image_file, channels=3)
            # Crop the image to the desired size
            crop_shape = tf.concat([(128,128), [3]], 0)
            cropped = tf.random_crop(image, crop_shape)
            cropped = tf.cast(cropped, tf.float32)/255.

            batch = tf.train.batch([cropped], batch_size=1)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            output = sess.run(m.output, feed_dict={m.input: batch.eval()})
            
            coord.request_stop()
            coord.join(threads)
            # Calculate PSNR gain for each one
            print('encoding jpegs')
            out_shape = output.shape
            if len(out_shape) > 3:
                out_shape = out_shape[1:]
            print(out_shape)

            output = output.reshape(out_shape)
            
            images_out = tf.image.encode_jpeg(output,name="output")
            print('writing to file')
            fh = open("./outputs/output.jpeg", "wb+")
            fh.write(images_out.eval())
            fh.close()
            print('file closed')
    else:
        print('Invalid "mode": {}!'.format(args.mode))
    sys.exit(0)

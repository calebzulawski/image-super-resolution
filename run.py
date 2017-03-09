#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    import sys
    import tensorflow as tf
    from model import Model

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, help='operating mode (train or generate)')
    parser.add_argument('--model', type=str, default='saved_model/model.ckpt')
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
            f = FileReader('./images/crop_256/*/*.JPEG', (args.subimage_size, args.subimage_size), batch_size=args.batch_size)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            try:
                saver.restore(sess, args.model)
            except:
                print('No save file found.  Creating new file at {}'.format(args.model));
            f.start_queue_runners()
            t.train(f.get_batch(), saver=saver, path=args.model)
            f.stop_queue_runners()
    elif args.mode == 'generate':
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
            # Calculate PSNR gain for each one
    else:
        print('Invalid "mode": {}!'.format(args.mode))
    sys.exit(0)

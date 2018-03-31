import numpy as np
import tensorflow as tf

import argparse
import glob
import time
from datetime import datetime
import os
import cPickle

from model_vrnn import VRNN
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict function
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
    - implement separate MDCT training and sampling version
'''

def next_batch(args):
    t0 = np.random.randn(args.batch_size, 1, (2 * args.chunk_samples))
    mixed_noise = np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #x = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    #y = t0 + mixed_noise + np.random.randn(
    #    args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1
    x = np.sin(2 * np.pi * (np.arange(args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1
    y = np.sin(2 * np.pi * (np.arange(1, args.seq_length + 1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(
        args.batch_size, args.seq_length, (2 * args.chunk_samples)) * 0.1 + mixed_noise*0.1

    y[:, :, args.chunk_samples:] = 0.
    x[:, :, args.chunk_samples:] = 0.
    return x, y


def train(args, model):
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname)
    n_batches = 100
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loaded model"
        start = time.time()
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            state = model.initial_state_c, model.initial_state_h
            for b in xrange(n_batches):
                x, y = next_batch(args)
                feed = {model.input_data: x, model.target_data: y}
                train_loss, _, cr, summary, sigma, mu, input, target= sess.run(
                        [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
                                                             feed)
                summary_writer.add_summary(summary, e * n_batches + b)
                if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                    print "model saved to {}".format(checkpoint_path)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
                    .format(e * n_batches + b,
                            args.num_epochs * n_batches,
                            e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
                start = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=3,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    args = parser.parse_args()

    model = VRNN(args)

    train(args, model)

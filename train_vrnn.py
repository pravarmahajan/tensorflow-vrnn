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
def pad_w_zeros(trip_seg, seq_length, num_features):
    l = len(trip_seg)
    if l <= seq_length:
        return  [[0]*num_features]*(seq_length+1-l) + trip_seg
    else:
        return trip_seg[:seq_length+1]

def load_data(args):
    trip_segments = np.load('{}.npy'.format(args.traj_data))
    print("Number of samples: {}".format(trip_segments.shape[0]))
    np.random.shuffle(trip_segments)
    split_idx = int((1-args.val_frac) * trip_segments.shape[0])
    return trip_segments[:split_idx], trip_segments[split_idx:]

def batch_generator(data, args):
    num_batches = 1+len(data)//args.batch_size
    for i in range(num_batches):
        yield data[i:i+args.batch_size, :, :]

def train(args, model):
    dirname = 'save-vrnn-1'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = False
    train_data, val_data = load_data(args)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loaded model"
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            train_batches = batch_generator(train_data, args)
            train_loss = 0.0
            for (b, x) in enumerate(train_batches):
                feed = {model.input_data: x[:, :args.seq_length, :], model.target_data: x[:, 1:, :]}
                loss, _, cr, summary, sigma, mu, input, target= sess.run(
                        [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
                                                             feed)
                train_loss += loss
            summary_writer.add_summary(summary, e)

            val_batches = batch_generator(val_data, args)
            val_loss = 0.0
            for (b, x) in enumerate(val_batches):
                feed = {model.input_data: x[:, :args.seq_length, :], model.target_data: x[:, 1:, :]}
                val_loss += sess.run(model.cost, feed)
            if e % args.save_every == 0 and e > 0:
                checkpoint_path = os.path.join(dirname, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=e)
                print "model saved to {}".format(checkpoint_path)
            print "{}/{}, train_loss = {:.6f}, std = {:.3f}" \
                    .format(e, args.num_epochs, train_loss/train_data.shape[0],
                            sigma.mean(axis=0).mean(axis=0))
            print("Validation Loss: {:.2f}".format(val_loss/val_data.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=256,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=128,
                        help='RNN sequence length')
    parser.add_argument('--num_features', type=int, default=35,
                        help='Number of features')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--traj_data', type=str, default='data/smallSample_50_200',
                        help='path to trajectory data')
    parser.add_argument('--val_frac', type=float, default='0.2',
                        help='fraction to use for validation')
    args = parser.parse_args()

    model = VRNN(args)

    train(args, model)

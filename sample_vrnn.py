import tensorflow as tf

import os
import cPickle
from model_vrnn import VRNN
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn
from train_vrnn import load_data

def plot_sample(sample, name):
    plt.clf()
    plt.imshow(sample.T, cmap='hot', interpolation='nearest')
    plt.savefig('{}.png'.format(name))

with open(os.path.join('save-vrnn', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save-vrnn')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)
sample_data,mus,sigmas = model.sample(sess,saved_args)
import ipdb; ipdb.set_trace()
plot_sample(sample_data, 'sample')
print(sample_data)

train, val = load_data(saved_args)
for i in range(5):
    random_data = train[np.random.randint(0, train.shape[0])]
    plot_sample(random_data, 'random_{}'.format(i))
    print(random_data)

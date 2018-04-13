import os
import cPickle

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn

from train_vrnn import load_data
from model_vrnn import VRNN
from model_config import config

dirname = config['model_path']

def plot_sample(sample, name):
    plt.clf()
    plt.imshow(sample.T, cmap='hot', interpolation='nearest')
    plt.savefig('{}.png'.format(name))

with open(os.path.join(dirname, 'config.pkl')) as f:
    saved_args = cPickle.load(f)

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

saver.restore(sess, os.path.join(dirname, 'model.ckpt-30'))
sample_data,mus,sigmas = model.sample(sess,saved_args)
print(mus)
print(sigmas)
sample_data[sample_data<0] = 0
plot_sample(sample_data, 'sample')
print(sample_data)

train, val = load_data(saved_args)
for i in range(5):
    random_data = train[np.random.randint(0, train.shape[0])]
    plot_sample(random_data, 'random_{}'.format(i))
    print(random_data)

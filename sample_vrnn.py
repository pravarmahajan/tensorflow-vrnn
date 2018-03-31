import tensorflow as tf

import os
import cPickle
from model_vrnn import VRNN
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from train_vrnn import next_batch
#num = 100
with open(os.path.join('save-vrnn', 'config.pkl')) as f:
    saved_args = cPickle.load(f)
num = saved_args.seq_length

model = VRNN(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

ckpt = tf.train.get_checkpoint_state('save-vrnn')
print "loading model: ",ckpt.model_checkpoint_path

saver.restore(sess, ckpt.model_checkpoint_path)
prev_x,sample_data,mus,sigmas= model.sample(sess,saved_args)

#plt.scatter(range(num), sample_data[:, 0], c='b', s=1)
#plt.scatter(range(num), sample_data[:, 1], c='g', s=1)
fig, axes = plt.subplots(figsize = (50, 10))
axes.plot(range(num), sample_data[:, 0], c='b')
axes.plot(range(num), prev_x[:, :, 0].squeeze(), c='g')
import ipdb; ipdb.set_trace()
#plt.plot(range(num), sample_data[:, 1], c='g')
plt.savefig('test.png')

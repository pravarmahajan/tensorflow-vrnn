import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import homogeneity_score
from sklearn.preprocessing import LabelEncoder

import cPickle
import os

from model_vrnn import VRNN
from model_config import config

with open(os.path.join(config['model_path'], 'config.pkl')) as f:
    saved_args = cPickle.load(f)

trip_segments = np.load('{}.npy'.format(saved_args.traj_data))/config['scale']

with open(saved_args.traj_data+"_keys.pkl", 'rb') as f:
    labels = np.array(cPickle.load(f))[:, 0]

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
model = VRNN(saved_args, True)

rng_state = np.random.get_state()
np.random.shuffle(trip_segments)
np.random.set_state(rng_state)
np.random.shuffle(labels)

split_idx = int((1-saved_args.val_frac) * trip_segments.shape[0])

train_data, val_data = trip_segments[:split_idx], trip_segments[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]
seq_length = saved_args.seq_length

with tf.Session() as sess:
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, os.path.join(config['model_path'], 'model.ckpt-30'))
    feed = {model.input_data: train_data[:, :seq_length, :]}
    [train_c, train_h] = \
            sess.run([model.final_state_c, model.final_state_h],feed)
    feed = {model.input_data: val_data[:, :seq_length, :]}
    [val_c, val_h] = \
            sess.run([model.final_state_c, model.final_state_h],feed)


print("train data generated")
clustering_model = MiniBatchKMeans(n_clusters=3)
train_pred_clusters = clustering_model.fit_predict(train_h)
print("Homogenity score (train) = %0.2f" % homogeneity_score(train_labels, train_pred_clusters).item())
val_pred_clusters = clustering_model.predict(val_h)
print("Homogenity score (val) = %0.2f" % homogeneity_score(val_labels, val_pred_clusters).item())

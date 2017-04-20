import numpy as np
np.random.seed = 123

import pandas as pd
from itertools import combinations
from math import ceil, floor, log, sin, cos, sqrt, pi
import io

import h5py

from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph

import data_utils
import net_utils

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()
color = {'training': 'b', 'validation': 'g', 'test': 'r'}


path = '../pdbbind/v2016/'
logdir = './logdir/relu/'
fname = './pdbbind_20a_19f_relu_kp%s_lmb%s_f%s_c%s_lr%s'
vsize = 1000
grid_spacing = 1.0

print('\n---- FEATURES ----\n')
print('atomic properties:', data_utils.FEATURE_NAMES)

columns = {name: i for i, name in enumerate(data_utils.FEATURE_NAMES)}


# # uncomment this part to create files with the training and validation sets
# with h5py.File('%s/training_set.hdf' % path, 'w') as g, \
#      h5py.File('%s/validation_set.hdf' % path, 'w') as h:
#     with h5py.File('%s/refined.hdf' % path, 'r') as f:
#         print('refined')
#         refined_shuffled = shuffle(list(f.keys()), random_state=123)
#         for pdb_id in refined_shuffled[:vsize]:
#             ds = h.create_dataset(pdb_id, data=f[pdb_id])
#             ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
#         for pdb_id in refined_shuffled[vsize:]:
#             ds = g.create_dataset(pdb_id, data=f[pdb_id])
#             ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
#     with h5py.File('%s/general.hdf' % path, 'r') as f:
#         print('general')
#         for pdb_id in f:
#             ds = g.create_dataset(pdb_id, data=f[pdb_id])
#             ds.attrs['affinity'] = f[pdb_id].attrs['affinity']

# # uncomment this part to create a symlink for the test set
# print('core')
# import os
# os.symlink(os.path.abspath('%s/core.hdf' % path), '%s/test_set.hdf' % path)


datasets = ['training', 'validation', 'test']

ids = {}
affinity = {}
coords = {}
features = {}

for dictionary in [ids, affinity, coords, features]:
    for dataset_name in datasets:
        dictionary[dataset_name] = []

for dataset_name in datasets:
    with h5py.File('%s/%s_set.hdf' % (path, dataset_name), 'r') as f:
        for pdb_id in f:
            dataset = f[pdb_id]

            coords[dataset_name].append(dataset[:, :3])
            features[dataset_name].append(dataset[:, 3:])
            affinity[dataset_name].append(dataset.attrs['affinity'])
            ids[dataset_name].append(pdb_id)

    ids[dataset_name] = np.array(ids[dataset_name])
    affinity[dataset_name] = np.reshape(affinity[dataset_name], (-1, 1))


# normalize charges
charges = []
for feature_data in features['training']:
    charges.append(feature_data[..., columns['partialcharge']])

charges = np.concatenate([c.flatten() for c in charges])

m = charges.mean()
std = charges.std()
print('charges: mean=%s, sd=%s' % (m, std))
print('use sd as scaling factor')


# Read and rotate structures
# Create matrices for all possible $90^{\circ}$ rotations of a box

def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""
    axis = np.asarray(axis)
    axis = axis/sqrt(np.dot(axis, axis))
    a = cos(theta/2.0)
    b, c, d = -axis*sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


all_rotations = [rotation_matrix([1, 1, 1], 0)]

# about X, Y and Z - 9 rotations
for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t*pi / 2.0
        all_rotations.append(rotation_matrix(axis, theta))

# about each face diagonal - 6 rotations
for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    all_rotations.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    all_rotations.append(rotation_matrix(axis, theta))

# about each space diagonal - 8 rotations
for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    all_rotations.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        all_rotations.append(rotation_matrix(axis, theta))


def get_batch(dataset_name, indices, rotation=0):
    global coords, features, all_rotations, std
    x = []
    for i, idx in enumerate(indices):
        coords_idx = np.dot(coords[dataset_name][idx], all_rotations[rotation])
        features_idx = features[dataset_name][idx]
        x.append(data_utils.make_grid(coords_idx, features_idx,
                 grid_resolution=grid_spacing))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std
    return x


print('\n---- DATA ----\n')

tmp = get_batch('training', range(50))

assert ((tmp[:, :, :, :, columns['moltype']] == 0)
        | (tmp[:, :, :, :, columns['moltype']] == 1)
        | (tmp[:, :, :, :, columns['moltype']] == -1)).all()

assert ((tmp[:, :, :, :, columns['moltype']] == 0).any()
        and (tmp[:, :, :, :, columns['moltype']] == 1).any()
        and (tmp[:, :, :, :, columns['moltype']] == -1).any()).all()

idx1 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['moltype']] == 1.0)]
idx2 = [[i[0]] for i in np.where(tmp[:, :, :, :, columns['moltype']] == -1.0)]

print('\nexamples:')
for mtype, mol in [['ligand', tmp[idx1]], ['protein', tmp[idx2]]]:
    print(' ', mtype)
    for name, num in columns.items():
        print('  ', name, mol[0, num])
    print('')


# Best error we can get without any training (MSE from training set mean):
t_baseline = ((affinity['training'] - affinity['training'].mean())**2).mean()
v_baseline = ((affinity['validation'] - affinity['training'].mean())**2).mean()
print('baseline mse: training=%s, validation=%s' % (t_baseline, v_baseline))


# NET PARAMS

ds_sizes = {dataset: len(affinity[dataset]) for dataset in datasets}
_, isize, *_, in_chnls = get_batch('training', [0]).shape
osize = 1

for set_name, set_size in ds_sizes.items():
    print('%s %s samples' % (set_size, set_name))

# convolutional layers
conv_patch = 5
pool_patch = 2
conv_channels = [64, 128, 256]

# fully connected layers
hsizes = [1000, 500, 200]

# regularization
kp = 0.5          # dropout
lmbda = 0.001       # weight decay

# training
learning_rate = 1e-5
batch_size = 20
num_batches = {dataset: size // batch_size
               for dataset, size in ds_sizes.items()}
num_epochs = 20
to_keep = 10

print('\n---- MODEL ----\n')
print(isize-1, 'A box')
print(in_chnls, 'features')
print('')
print('convolutional layers: %s channels, %sA patch + max pooling with %sA patch'
      % (', '.join((str(i) for i in conv_channels)), conv_patch, pool_patch))
print('fully connected layers:', ', '.join((str(i) for i in hsizes)), 'neurons')
print('regularization: dropout (keep %s) and L2 (lambda %s)' % (kp, lmbda))
print('')
print('learning rate', learning_rate)
print(num_batches['training'], 'batches,', batch_size, 'examples each')
print(num_batches['validation'], 'validation batches')
print(num_batches['test'], 'test batches')
print('')
print(num_epochs, 'epochs, best', to_keep, 'saved')

net_utils.make_network(isize=isize, in_chnls=in_chnls, osize=osize,
                       conv_patch=conv_patch, pool_patch=pool_patch,
                       conv_channels=conv_channels,
                       hsizes=hsizes,
                       kp=kp, lmbda=lmbda, learning_rate=learning_rate)


graph = tf.get_default_graph()

train_writer = tf.summary.FileWriter('%s/training_set' % logdir, graph)
val_writer = tf.summary.FileWriter('%s/validation_set' % logdir)

net_summaries, training_summaries = net_utils.make_summaries()

# import all tensors created with make_network and make_summaries
from net_utils import *


convs = '_'.join((str(i) for i in conv_channels))
fcs = '_'.join((str(i) for i in hsizes))

saver = tf.train.Saver(max_to_keep=to_keep)
prefix = fname % (kp, lmbda, fcs, convs, learning_rate)

err = float('inf')

print('\n---- TRAINING ----\n')
with tf.Session() as session:
    tf.set_random_seed(123)
    session.run(tf.global_variables_initializer())

    summary_imp = tf.Summary()
    feature_imp = session.run(feature_importance)
    image = feature_importance_plot(feature_imp)
    summary_imp.value.add(tag='feature_importance_%s' % 0, image=image)
    train_writer.add_summary(summary_imp, 0)

    stats_net = session.run(
        net_summaries,
        feed_dict={x: get_batch('training', range(batch_size)),
                   t: affinity['training'][:batch_size],
                   keep_prob: 1.0}
    )

    train_writer.add_summary(stats_net, 0)

    for epoch in range(num_epochs):
        for rotation in range(24):
            print('rotation', rotation)
            # TRAIN #
            x_t, y_t = shuffle(range(ds_sizes['training']), affinity['training'])

            for b in range(num_batches['training']):
                bi = b*batch_size
                bj = (b+1)*batch_size
                if b == num_batches['training'] - 1:
                    bj = ds_sizes['training']

                session.run(train, feed_dict={x: get_batch('training',
                                                           x_t[bi:bj],
                                                           rotation),
                                              t: y_t[bi:bj], keep_prob: kp})

            # SAVE STATS - per rotation #
            stats_t, stats_net = session.run(
                [training_summaries, net_summaries],
                feed_dict={x: get_batch('training', x_t[:batch_size]),
                           t: y_t[:batch_size],
                           keep_prob: 1.0}
            )

            train_writer.add_summary(stats_t, global_step.eval())
            train_writer.add_summary(stats_net, global_step.eval())

            stats_v = session.run(
                training_summaries,
                feed_dict={x: get_batch('validation', range(batch_size)),
                           t: affinity['validation'][:batch_size],
                           keep_prob: 1.0}
            )

            val_writer.add_summary(stats_v, global_step.eval())

        # SAVE STATS - per epoch #
        # training set error
        pred_t = np.zeros((ds_sizes['training'], 1))
        mse_t = np.zeros(num_batches['training'])

        for b in range(num_batches['training']):
            bi = b*batch_size
            bj = (b+1)*batch_size
            if b == num_batches['training'] - 1:
                bj = ds_sizes['training']
            weight = (bj-bi) / ds_sizes['training']

            pred_t[bi:bj], mse_t[b] = session.run(
                [y, mse],
                feed_dict={x: get_batch('training', x_t[bi:bj]),
                           t: y_t[bi:bj],
                           keep_prob: 1.0}
            )

            mse_t[b] *= weight

        mse_t = mse_t.sum()

        summary_mse = tf.Summary()
        summary_mse.value.add(tag='mse_all', simple_value=mse_t)
        train_writer.add_summary(summary_mse, global_step.eval())

        # predictions distribution
        summary_pred = tf.Summary()
        summary_pred.value.add(tag='predictions_all',
                               histo=custom_summary_histogram(pred_t))
        train_writer.add_summary(summary_pred, global_step.eval())

        # feature importance
        summary_imp = tf.Summary()
        feature_imp = session.run(feature_importance)
        image = feature_importance_plot(feature_imp)
        summary_imp.value.add(tag='feature_importance', image=image)
        train_writer.add_summary(summary_imp, global_step.eval())

        # validation set error
        mse_v = 0
        for b in range(num_batches['validation']):
            bi = b*batch_size
            bj = (b+1)*batch_size
            if b == num_batches['validation'] - 1:
                bj = ds_sizes['validation']

            weight = (bj-bi) / ds_sizes['validation']
            mse_v += weight*session.run(
                mse,
                feed_dict={x: get_batch('validation', range(bi, bj)),
                           t: affinity['validation'][bi:bj],
                           keep_prob: 1.0}
            )

        summary_mse = tf.Summary()
        summary_mse.value.add(tag='mse_all', simple_value=mse_v)
        val_writer.add_summary(summary_mse, global_step.eval())

        # SAVE MODEL #
        print('epoch: %s train error: %s, test error: %s'
              % (epoch, mse_t, mse_v))

        if mse_v <= err:
            err = mse_v
            checkpoint = saver.save(session, prefix, global_step=global_step)


# FINAL PREDICTIONS


predictions = []
rmse = {}

with tf.Session() as session:
    tf.set_random_seed(123)

    saver.restore(session, './'+checkpoint)
    saver.save(session, './%s-best' % prefix)

    for dataset in datasets:
        pred = np.zeros((ds_sizes[dataset], 1))
        mse_dataset = 0.0

        for b in range(num_batches[dataset]):
            bi = b*batch_size
            bj = (b+1)*batch_size
            if b == num_batches[dataset] - 1:
                bj = ds_sizes[dataset]

            weight = (bj-bi) / ds_sizes[dataset]
            pred[bi:bj], mse_batch = session.run(
                [y, mse],
                feed_dict={x: get_batch(dataset, range(bi, bj)),
                           t: affinity[dataset][bi:bj],
                           keep_prob: 1.0}
            )
            mse_dataset += weight * mse_batch

        predictions.append(pd.DataFrame(data={'real': affinity[dataset][:, 0],
                                              'predicted': pred[:, 0],
                                              'set': dataset}))
        rmse[dataset] = sqrt(mse_dataset)


predictions = pd.concat(predictions, ignore_index=True)

for set_name, tab in predictions.groupby('set'):
    grid = sns.jointplot('real', 'predicted', data=tab, color=color[set_name],
                         space=0.0, xlim=(0, 16), ylim=(0, 16),
                         annot_kws={'title': '%s set (rmse=%.3f)'
                                             % (set_name, rmse[dataset])})

    image = custom_summary_image(grid.fig)
    summary_pred = tf.Summary()
    summary_pred.value.add(tag='predictions_%s' % (set_name),
                           image=image)
    train_writer.add_summary(summary_pred)


train_writer.close()
val_writer.close()

import numpy as np
np.random.seed(123)

import pandas as pd
from math import sqrt

import h5py

from sklearn.utils import shuffle
import tensorflow as tf

import utils

import matplotlib as mpl
mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()
color = {'training': 'b', 'validation': 'g', 'test': 'r'}

import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')

import argparse
parser = argparse.ArgumentParser(
    description='Train 3D colnvolutional neural network on affinity data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

io_group = parser.add_argument_group('I/O')
io_group.add_argument('--input_dir', '-i', default='../pdbbind/v2016/',
                      help='directory with training, validation and test sets')
io_group.add_argument('--log_dir', '-l', default='./logdir/',
                      help='directory to store tensorboard summaries')
io_group.add_argument('--output_prefix', '-o', default='./output',
                      help='prefix for checkpoints, predictions and plots')
io_group.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                      help='distance between grid points')

arc_group = parser.add_argument_group('Netwrok architecture')
arc_group.add_argument('--conv_patch', default=5, type=int,
                       help='patch size for convolutional layers')
arc_group.add_argument('--pool_patch', default=2, type=int,
                       help='patch size for pooling layers')
arc_group.add_argument('--conv_channels', metavar='C', default=[64, 128, 256],
                       type=int, nargs='+',
                       help='number of fileters in convolutional layers')
arc_group.add_argument('--dense_sizes', metavar='D', default=[1000, 500, 200],
                       type=int, nargs='+',
                       help='number of neurons in dense layers')

reg_group = parser.add_argument_group('Regularization')
reg_group.add_argument('--keep_prob', dest='kp', default=0.5, type=float,
                       help='keep probability for dropout')
reg_group.add_argument('--l2', dest='lmbda', default=0.001, type=float,
                       help='lambda for weight decay')
reg_group.add_argument('--rotations', metavar='R', default=list(range(24)),
                       type=int, nargs='+',
                       help='rotations to perform')

tr_group = parser.add_argument_group('Training')
tr_group.add_argument('--learning_rate', default=1e-5, type=float,
                      help='learning rate')
tr_group.add_argument('--batch_size', default=20, type=int,
                      help='batch size')
tr_group.add_argument('--num_epochs', default=20, type=int,
                      help='number of epochs')
tr_group.add_argument('--num_checkpoints', dest='to_keep', default=10, type=int,
                      help='number of checkpoints to keep')

args = parser.parse_args()

logdir = args.log_dir + '/' + timestamp
prefix = args.output_prefix + '-' + timestamp

print('\n---- FEATURES ----\n')
print('atomic properties:', utils.data.FEATURE_NAMES)

columns = {name: i for i, name in enumerate(utils.data.FEATURE_NAMES)}

datasets = ['training', 'validation', 'test']

ids = {}
affinity = {}
coords = {}
features = {}

for dictionary in [ids, affinity, coords, features]:
    for dataset_name in datasets:
        dictionary[dataset_name] = []

for dataset_name in datasets:
    with h5py.File('%s/%s_set.hdf' % (args.input_dir, dataset_name), 'r') as f:
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


def get_batch(dataset_name, indices, rotation=0):
    global coords, features, std
    x = []
    for i, idx in enumerate(indices):
        coords_idx = utils.data.rotate(coords[dataset_name][idx], rotation)
        features_idx = features[dataset_name][idx]
        x.append(utils.data.make_grid(coords_idx, features_idx,
                 grid_resolution=args.grid_spacing))
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

num_batches = {dataset: size // args.batch_size
               for dataset, size in ds_sizes.items()}

print('\n---- MODEL ----\n')
print((isize-1) * args.grid_spacing, 'A box')
print(in_chnls, 'features')
print('')
print('convolutional layers: %s channels, %sA patch + max pooling with %sA patch'
      % (', '.join((str(i) for i in args.conv_channels)), args.conv_patch,
         args.pool_patch))
print('fully connected layers:', ', '.join((str(i) for i in args.dense_sizes)),
      'neurons')
print('regularization: dropout (keep %s) and L2 (lambda %s)'
      % (args.kp, args.lmbda))
print('')
print('learning rate', args.learning_rate)
print(num_batches['training'], 'batches,', args.batch_size, 'examples each')
print(num_batches['validation'], 'validation batches')
print(num_batches['test'], 'test batches')
print('')
print(args.num_epochs, 'epochs, best', args.to_keep, 'saved')

graph = utils.net.make_network(isize=isize, in_chnls=in_chnls, osize=osize,
                       conv_patch=args.conv_patch, pool_patch=args.pool_patch,
                       conv_channels=args.conv_channels,
                       dense_sizes=args.dense_sizes,
                       kp=args.kp, lmbda=args.lmbda,
                       learning_rate=args.learning_rate)


train_writer = tf.summary.FileWriter('%s/training_set' % logdir, graph,
                                     flush_secs=1)
val_writer = tf.summary.FileWriter('%s/validation_set' % logdir, flush_secs=1)

net_summaries, training_summaries = utils.net.make_summaries(graph)

x = graph.get_tensor_by_name('input/structure:0')
y = graph.get_tensor_by_name('output/prediction:0')
t = graph.get_tensor_by_name('input/affinity:0')
keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')
train = graph.get_tensor_by_name('training/train:0')
mse = graph.get_tensor_by_name('training/mse:0')
feature_importance = graph.get_tensor_by_name('net_properties/feature_importance:0')
global_step = graph.get_tensor_by_name('training/global_step:0')

convs = '_'.join((str(i) for i in args.conv_channels))
fcs = '_'.join((str(i) for i in args.dense_sizes))

with graph.as_default():
    saver = tf.train.Saver(max_to_keep=args.to_keep)


err = float('inf')

print('\n---- TRAINING ----\n')
with tf.Session(graph=graph) as session:
    tf.set_random_seed(123)
    session.run(tf.global_variables_initializer())

    summary_imp = tf.Summary()
    feature_imp = session.run(feature_importance)
    image = utils.net.feature_importance_plot(feature_imp)
    summary_imp.value.add(tag='feature_importance_%s' % 0, image=image)
    train_writer.add_summary(summary_imp, 0)

    stats_net = session.run(
        net_summaries,
        feed_dict={x: get_batch('training', range(args.batch_size)),
                   t: affinity['training'][:args.batch_size],
                   keep_prob: 1.0}
    )

    train_writer.add_summary(stats_net, 0)

    for epoch in range(args.num_epochs):
        for rotation in args.rotations:
            print('rotation', rotation)
            # TRAIN #
            x_t, y_t = shuffle(range(ds_sizes['training']), affinity['training'])

            for b in range(num_batches['training']):
                bi = b*args.batch_size
                bj = (b+1)*args.batch_size
                if b == num_batches['training'] - 1:
                    bj = ds_sizes['training']

                session.run(train, feed_dict={x: get_batch('training',
                                                           x_t[bi:bj],
                                                           rotation),
                                              t: y_t[bi:bj], keep_prob: args.kp})

            # SAVE STATS - per rotation #
            stats_t, stats_net = session.run(
                [training_summaries, net_summaries],
                feed_dict={x: get_batch('training', x_t[:args.batch_size]),
                           t: y_t[:args.batch_size],
                           keep_prob: 1.0}
            )

            train_writer.add_summary(stats_t, global_step.eval())
            train_writer.add_summary(stats_net, global_step.eval())

            stats_v = session.run(
                training_summaries,
                feed_dict={x: get_batch('validation', range(args.batch_size)),
                           t: affinity['validation'][:args.batch_size],
                           keep_prob: 1.0}
            )

            val_writer.add_summary(stats_v, global_step.eval())

        # SAVE STATS - per epoch #
        # training set error
        pred_t = np.zeros((ds_sizes['training'], 1))
        mse_t = np.zeros(num_batches['training'])

        for b in range(num_batches['training']):
            bi = b*args.batch_size
            bj = (b+1)*args.batch_size
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
                               histo=utils.net.custom_summary_histogram(pred_t))
        train_writer.add_summary(summary_pred, global_step.eval())

        # validation set error
        mse_v = 0
        for b in range(num_batches['validation']):
            bi = b*args.batch_size
            bj = (b+1)*args.batch_size
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
        print('epoch: %s train error: %s, validation error: %s'
              % (epoch, mse_t, mse_v))

        if mse_v <= err:
            err = mse_v
            checkpoint = saver.save(session, prefix, global_step=global_step)

            # feature importance
            summary_imp = tf.Summary()
            feature_imp = session.run(feature_importance)
            image = utils.net.feature_importance_plot(feature_imp)
            summary_imp.value.add(tag='feature_importance', image=image)
            train_writer.add_summary(summary_imp, global_step.eval())


# FINAL PREDICTIONS


predictions = []
rmse = {}

with tf.Session(graph=graph) as session:
    tf.set_random_seed(123)

    saver.restore(session, './'+checkpoint)
    saver.save(session, './%s-best' % prefix)

    for dataset in datasets:
        pred = np.zeros((ds_sizes[dataset], 1))
        mse_dataset = 0.0

        for b in range(num_batches[dataset]):
            bi = b*args.batch_size
            bj = (b+1)*args.batch_size
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

        predictions.append(pd.DataFrame(data={'pdbid': ids[dataset],
                                              'real': affinity[dataset][:, 0],
                                              'predicted': pred[:, 0],
                                              'set': dataset}))
        rmse[dataset] = sqrt(mse_dataset)


predictions = pd.concat(predictions, ignore_index=True)
predictions.to_csv(prefix+'-predictions.csv', index=False)

for set_name, tab in predictions.groupby('set'):
    grid = sns.jointplot('real', 'predicted', data=tab, color=color[set_name],
                         space=0.0, xlim=(0, 16), ylim=(0, 16),
                         annot_kws={'title': '%s set (rmse=%.3f)'
                                             % (set_name, rmse[dataset])})

    image = utils.net.custom_summary_image(grid.fig)
    grid.fig.savefig('%s-%s.pdf' % (prefix, set_name))
    summary_pred = tf.Summary()
    summary_pred.value.add(tag='predictions_%s' % (set_name),
                           image=image)
    train_writer.add_summary(summary_pred)


train_writer.close()
val_writer.close()

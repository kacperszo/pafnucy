import numpy as np
np.random.seed(123)

from math import ceil
import io

import tensorflow as tf

from utils.data import FEATURE_NAMES

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()


def hidden_conv(inp, out_chnls, conv_patch=5, pool_patch=2, name='conv'):

    in_chnls = inp.get_shape()[-1].value
    with tf.name_scope(name):
        w_shape = (conv_patch, conv_patch, conv_patch, in_chnls, out_chnls)

        w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.001),
                        name='w')
        b = tf.Variable(0.1*np.ones((out_chnls,), dtype=np.float32), name='b')

        conv = tf.nn.conv3d(inp, w, strides=(1, 1, 1, 1, 1), padding='SAME',
                            name='conv')
        h = tf.nn.relu(conv + b, name='h')

        pool_shape = (1, pool_patch, pool_patch, pool_patch, 1)
        h_pool = tf.nn.max_pool3d(h, ksize=pool_shape, strides=pool_shape,
                                  padding='SAME', name='h_pool')
    return h_pool, tf.reduce_sum(tf.pow(w, 2))


def hidden_fcl(inp, out_size, keep_prob, name='hidden'):
    assert len(inp.get_shape()) == 2

    in_size = inp.get_shape()[1].value

    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=(in_size, out_size),
                                            stddev=(1 / (in_size**0.5))),
                        name='w')
        b = tf.Variable(np.ones((out_size,), dtype=np.float32), name='b')

        h = tf.nn.relu(tf.matmul(inp, w) + b, name='h')
        h_drop = tf.nn.dropout(h, keep_prob, name='h_dropout')

    return h_drop, tf.reduce_sum(tf.pow(w, 2))


def convolve(inp, channels, conv_patch=5, pool_patch=2):
    prev = inp
    weights = []
    i = 0
    for num_channels in channels:
        output, w_sum = hidden_conv(prev, num_channels, conv_patch, pool_patch,
                                    name='conv%s' % i)
        i += 1
        weights.append(w_sum)
        prev = output
    return output, tf.reduce_sum(weights)


def feedforward(inp, dense_sizes, keep_prob=1.0):
    prev = inp
    weights = []
    i = 0
    for hsize in dense_sizes:
        output, w_sum = hidden_fcl(prev, hsize, keep_prob, name='fc%s' % i)
        i += 1
        weights.append(w_sum)
        prev = output
    return output, tf.reduce_sum(weights)


def make_network(isize=20, in_chnls=len(FEATURE_NAMES), osize=1,
                 conv_patch=5, pool_patch=2, conv_channels=[64, 128, 256],
                 dense_sizes=[1000, 500, 200],
                 kp=0.5, lmbda=0.001, learning_rate=1e-5):

    graph = tf.Graph()

    with graph.as_default():
        np.random.seed(123)
        tf.set_random_seed(123)
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32,
                               shape=(None, isize, isize, isize, in_chnls),
                               name='structure')
            t = tf.placeholder(tf.float32, shape=(None, osize), name='affinity')

        with tf.name_scope('convolution'):
            h_convs, w_sum_conv = convolve(x, conv_channels,
                                           conv_patch=conv_patch,
                                           pool_patch=pool_patch)
        hfsize = isize
        for _ in range(len(conv_channels)):
            hfsize = ceil(hfsize / pool_patch)
        hfsize = conv_channels[-1] * hfsize**3

        with tf.name_scope('fully_connected'):
            h_flat = tf.reshape(h_convs, shape=(-1, hfsize), name='h_flat')

            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            h_fcl, w_sum_fcl = feedforward(h_flat, dense_sizes, keep_prob=keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal(shape=(dense_sizes[-1], osize),
                            stddev=(1 / (dense_sizes[-1]**0.5))), name='w')
            b = tf.Variable(np.ones((osize,), dtype=np.float32), name='b')
            y = tf.nn.relu(tf.matmul(h_fcl, w) + b, name='prediction')

        with tf.name_scope('training'):
            global_step = tf.Variable(0, trainable=False, name='global_step')

            mse = tf.reduce_mean(tf.pow((y - t), 2), name='mse')

            with tf.name_scope('L2_cost'):
                l2 = lmbda * (w_sum_conv + w_sum_fcl + tf.reduce_sum(tf.pow(w, 2)))

            cost = tf.add(mse, l2, name='cost')

            optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
            train = optimizer.minimize(cost, global_step=global_step, name='train')

    graph.add_to_collection('output', y)
    graph.add_to_collection('input', x)
    graph.add_to_collection('target', t)
    graph.add_to_collection('kp', keep_prob)

    return graph


def custom_summary_histogram(values):

    flat = values.flatten()
    hist, bins = np.histogram(flat, bins=200)

    bins_middle = (bins[:-1] + bins[1:]) / 2

    histogram = tf.HistogramProto(min=flat.min(), max=flat.max(),
                                  num=len(flat), sum=flat.sum(),
                                  sum_squares=(flat ** 2).sum(),
                                  bucket_limit=bins_middle, bucket=hist)

    return histogram


def custom_summary_image(mpl_figure):
    imgdata = io.BytesIO()
    mpl_figure.savefig(imgdata, format='png', dpi=300)
    imgdata.seek(0)

    width, height = mpl_figure.canvas.get_width_height()

    image = tf.Summary.Image(height=height, width=width, colorspace=3,
                             encoded_image_string=imgdata.getvalue())
    imgdata.close()

    return image


def feature_importance_plot(values):
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.barplot(y=FEATURE_NAMES, x=values, ax=ax)
    fig.tight_layout()

    image = custom_summary_image(fig)
    plt.close(fig)

    return image


def make_summaries(graph):
    global FEATURE_NAMES

    in_chnls = len(FEATURE_NAMES)

    with graph.as_default():
        with tf.name_scope('net_properties'):
            # weights between input and the first layer
            wconv0 = graph.get_tensor_by_name('convolution/conv0/w:0')
            feature_weights = tf.split(wconv0, in_chnls, axis=3)
            feature_importance = tf.reduce_sum(tf.abs(wconv0),
                                               reduction_indices=[0, 1, 2, 4],
                                               name='feature_importance')

        net_summaries = tf.summary.merge((
            tf.summary.histogram('weights', wconv0),
            *(tf.summary.histogram('weights_%s' % name, value)
              for name, value in zip(FEATURE_NAMES, feature_weights)),
            tf.summary.histogram('predictions', graph.get_tensor_by_name('output/prediction:0'))
        ))

        training_summaries = tf.summary.merge((
            tf.summary.scalar('mse', graph.get_tensor_by_name('training/mse:0')),
            tf.summary.scalar('cost', graph.get_tensor_by_name('training/cost:0'))
        ))

    return net_summaries, training_summaries

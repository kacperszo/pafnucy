import unittest

import pybel
import numpy as np

import tensorflow as tf
import math

import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))


class DataUtilsTests(unittest.TestCase):

    def setUp(self):
        from glob import glob

        self.complexes = []
        for directory in glob('data/complexes/*'):
            pdb_id = directory.split('/')[-1]
            ligand = pybel.readfile('mol2', '%s/%s_ligand.mol2'
                                    % (directory, pdb_id))
            pocket = pybel.readfile('mol2', '%s/%s_pocket.mol2'
                                    % (directory, pdb_id))
            self.complexes.append((pocket, ligand))

        np.random.seed(123)

    def tearDown(self):
        self.complexes = None

    def test_encode_num(self):
        from utils.data import encode_num

        for mols in self.complexes:
            for mol in mols:
                for atom in mol:
                    encoding = encode_num(atom.atomicnum)
                    self.assertIn(encoding.sum(), [0.0, 1.0])
                    if atom.atomicnum in [6, 7, 8]:
                        self.assertEqual(encoding.sum(), 1.0)

    def test_find_smarts(self):
        from utils.data import find_smarts

        for mols in self.complexes:
            for mol in mols:
                smarts = find_smarts(mol)
                self.assertTrue(smarts.any())

    def test_get_features(self):
        from utils.data import get_features

        for mols in self.complexes:
            for mol in mols:
                coords, features = get_features(mol)
                self.assertEqual(len(coords), len(features))
                self.assertTrue((features != 0).any(axis=1).all())
                self.assertTrue((features != 0).any(axis=0).all())

    def test_rotate(self):
        from utils.data import rotate

        coords = np.random.rand(1, 3)
        length = np.linalg.norm(coords)

        for rotation in range(24):
            coords_rot = rotate(coords, rotation)
            self.assertAlmostEqual(np.linalg.norm(coords_rot), length)

    def test_make_grid(self):
        from utils.data import get_features, make_grid

        for mols in self.complexes:
            for mol in mols:
                coords, features = get_features(mol)
                coords -= coords.mean(axis=0)

                max_volume = coords.abs().max() ** 3

                for dist in [5.0, 10.0, 15.0]:
                    box_volume = (2*dist) ** 3

                    for resolution in [0.5, 1.0, 2.0]:

                        grid = make_grid(coords, features,
                                         grid_resolution=resolution,
                                         max_dist=dist)

                        if max_volume > box_volume:
                            self.assertTrue((np.sum(grid, axis=list(range(4)))
                                            <= np.sum(features, axis=0)).all())
                        else:
                            self.assertTrue((np.sum(grid, axis=list(range(4)))
                                            == np.sum(features, axis=0)).all())


class NetUtilsTests(unittest.TestCase):

    def setUp(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
        self.flat = tf.placeholder(tf.float32, shape=(None, 100))
        self.t = tf.placeholder(tf.float32, shape=(None, 1))

    def tearDown(self):
        tf.reset_default_graph()

    def test_hidden_conv(self):
        from utils.net import hidden_conv

        for out_chnls in [8, 16]:
            for pool_patch in [2, 3]:
                for conv_patch in [2, 5, 10]:
                    h, _ = hidden_conv(self.x, out_chnls, conv_patch=conv_patch,
                                       pool_patch=pool_patch)

                    input_tensor = h
                    # there are 4 operations between x and h so we need 4 steps
                    # to get back to x:
                    # MaxPooling -> Conv -> Add -> x
                    for _ in range(4):
                        input_tensor = input_tensor.op.inputs[0]
                    self.assertEqual(input_tensor, self.x)
                    shape = h.get_shape().as_list()
                    s = math.ceil(21 / pool_patch)
                    self.assertListEqual(shape, [None, s, s, s, out_chnls])

    def test_hidden_fcl(self):
        from utils.net import hidden_fcl

        keep_prob = tf.placeholder(tf.float32)

        for out_size in [8, 16]:
            h, _ = hidden_fcl(self.flat, out_size, keep_prob)
            input_tensor = h
            # there are 5 operations between x and h so we need 5 steps
            # to get back to x:
            # Dropout -> ReLU -> Add -> MatMul -> x
            for _ in range(5):
                input_tensor = input_tensor.op.inputs[0]
            self.assertEqual(input_tensor, self.flat)

            shape = h.get_shape().as_list()
            self.assertListEqual(shape, [None, out_size])

    def test_convolve(self):
        from utils.net import hidden_conv, convolve
        from tensorflow.python import pywrap_tensorflow

        for pool_patch in [2, 3]:
            for conv_patch in [2, 5, 10]:
                out_chnls = [8, 16]
                g1 = tf.Graph()
                with g1.as_default():
                    x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
                    h11, w1 = hidden_conv(x, out_chnls[0], conv_patch=conv_patch,
                                          pool_patch=pool_patch, name='conv0')
                    h12, w2 = hidden_conv(h11, out_chnls[1], conv_patch=conv_patch,
                                          pool_patch=pool_patch, name='conv1')
                    tf.reduce_sum([w1, w2])
                def1 = g1.as_graph_def().SerializeToString()

                g2 = tf.Graph()
                with g2.as_default():
                    x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
                    h2, _ = convolve(x, out_chnls, conv_patch=conv_patch,
                                     pool_patch=pool_patch)
                def2 = g2.as_graph_def().SerializeToString()

                self.assertFalse(pywrap_tensorflow.EqualGraphDefWrapper(def1, def2))

    def test_feedforward(self):
        from utils.net import hidden_fcl, feedforward
        from tensorflow.python import pywrap_tensorflow

        out_sizes = [8, 16]
        g1 = tf.Graph()
        with g1.as_default():
            x = tf.placeholder(tf.float32, shape=(None, 100))
            kp = tf.constant(1.0)
            h11, w1 = hidden_fcl(x, out_sizes[0], keep_prob=kp, name='fc0')
            h12, w2 = hidden_fcl(h11, out_sizes[1], keep_prob=kp, name='fc1')
            tf.reduce_sum([w1, w2])
        def1 = g1.as_graph_def().SerializeToString()

        g2 = tf.Graph()
        with g2.as_default():
            x = tf.placeholder(tf.float32, shape=(None, 100))
            kp = tf.constant(1.0)
            h2, _ = feedforward(x, out_sizes, keep_prob=kp)
        def2 = g2.as_graph_def().SerializeToString()

        self.assertFalse(pywrap_tensorflow.EqualGraphDefWrapper(def1, def2))

if __name__ == '__main__':
    unittest.main(verbosity=2)

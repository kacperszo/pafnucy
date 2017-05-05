import unittest

import pybel
import numpy as np
np.random.seed(123)

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

    def tearDown(self):
        self.complexes = None

    def test_encode_num(self):
        from data_utils import encode_num

        for mols in self.complexes:
            for mol in mols:
                for atom in mol:
                    encoding = encode_num(atom.atomicnum)
                    self.assertIn(encoding.sum(), [0.0, 1.0])
                    if atom.atomicnum in [6, 7, 8]:
                        self.assertEqual(encoding.sum(), 1.0)

    def test_find_smarts(self):
        from data_utils import find_smarts

        for mols in self.complexes:
            for mol in mols:
                smarts = find_smarts(mol)
                self.assertTrue(smarts.any())

    def test_get_features(self):
        from data_utils import get_features

        for mols in self.complexes:
            for mol in mols:
                coords, features = get_features(mol)
                self.assertEqual(len(coords), len(features))
                self.assertTrue((features != 0).any(axis=1).all())
                self.assertTrue((features != 0).any(axis=0).all())

    def test_rotate(self):
        from data_utils import rotate

        coords = np.random.rand(1, 3)
        length = np.linalg.norm(coords)

        for rotation in range(24):
            coords_rot = rotate(coords, rotation)
            self.assertAlmostEqual(np.linalg.norm(coords_rot), length)

    def test_make_grid(self):
        from data_utils import get_features, make_grid

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

if __name__ == '__main__':
    unittest.main(verbosity=2)

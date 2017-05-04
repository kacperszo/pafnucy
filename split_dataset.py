from sklearn.utils import shuffle
import h5py

import os

path = '../pdbbind/v2016/'
vsize = 1000

# create files with the training and validation sets
with h5py.File('%s/training_set.hdf' % path, 'w') as g, \
     h5py.File('%s/validation_set.hdf' % path, 'w') as h:
    with h5py.File('%s/refined.hdf' % path, 'r') as f:
        refined_shuffled = shuffle(list(f.keys()), random_state=123)
        for pdb_id in refined_shuffled[:vsize]:
            ds = h.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
        for pdb_id in refined_shuffled[vsize:]:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
    with h5py.File('%s/general.hdf' % path, 'r') as f:
        for pdb_id in f:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']

# create a symlink for the test set
os.symlink(os.path.abspath('%s/core.hdf' % path), '%s/test_set.hdf' % path)

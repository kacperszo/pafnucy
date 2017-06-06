[![build status](https://gitlab.com/marta-sd/affinity-net/badges/master/build.svg)](https://gitlab.com/marta-sd/affinity-net/commits/master)


[CASF benchmark](http://pubs.acs.org/doi/pdf/10.1021/ci500081m)


# Hyperparameters

## Default:
### Input:
* 1A grid
* 20A box (both ends included)
* 19 features:
    *  atom type: *B*, *C*, *N*, *O*, *P*, *S*, *Se*, *halogen*, and *metal* (one-hot or null, 9 columns)
    * hybridization (*hyb*, 1, 2, or 3)
    * connections with other heavy- and heteroatoms (*heavyvalence* and *heterovalence*)
    * additional properties defined with SMARTS patterns: *hydrophobic*, *aromatic*, *acceptor*, *donor*, and *ring* (binary)
    * partial charge (*partialcharge*, scaled by training set std)
    * ligand / protein (*moltype*, 1 for ligand, -1 for protein)

### Model
* architecture:
    * 3 convolutional layers with 64, 128, and 256 filters, each with 5A filter size and followed by max pooling with 2A patch size
    * 3 dense layers with 1000, 500, and 200 neurons
* initialization:
    * convolutional filters - weights draw from truncated normal with 0 mean and 0.001 std, all biases set to 0.1
    * dense layers - weights draw from truncated normal with 0 mean and 1/sqrt(fan_in) std, all biases set to 1.0
* regularization:
    * 0.5 dropout
    * 0.001 weight decay (L2)
* training:
    * 20 epochs
    * 20 samples per batch
    * Adam optimizer with 1e-5 learning rate
    * data augmentation (24 different orientations for each training case)
    * model evaluated after each epoch, saved only if validation error improved

## Other tested setups:
* 10 samples per batch - same error, slightly better correlation
* **5 samples per batch** - best performing model
* 0.2 dropout (keep_prob=0.8) / no dropout (keep_prob=1.0) - results for all sets are worse
* no weight decay - results are almost the same, but weights are much higher, especially for boron (B), which is present in only 166 compounds in the training set
* 0.01 weight decay - weights look much better, but results are worse (also for training set)

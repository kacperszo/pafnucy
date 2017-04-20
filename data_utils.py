import numpy as np
import pybel

# Remember namse of all features in the correct order
FEATURE_NAMES = []

ATOM_CODES = {}

# List of tuples (atomic_num, class_name) with atom types to encode.
atom_classes = [
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (15, 'P'),
    (16, 'S'),
    (34, 'Se'),
    ([9, 17, 35, 53], 'halogen'),
    ([11, 12, 20, 25, 26, 27, 28, 29, 30], 'metal')
]

for code, (atom, name) in enumerate(atom_classes):
    if type(atom) is list:
        for a in atom:
            ATOM_CODES[a] = code
    else:
        ATOM_CODES[atom] = code
    FEATURE_NAMES.append(name)

NUM_ATOM_CLASSES = len(atom_classes)


def encode_num(atomic_num):
    """Encode atom type with a binary vector. If atom type is not included in
    the `atom_classes`, its encoding is an all-zeros vector.

    Parameters
    ----------
    atomic_num: int
        Atomic number

    Returns
    -------
    encoding: np.ndarray
        Binary vector encoding atom type (one-hot or null).
    """

    global NUM_ATOM_CLASSES, ATOM_CODES

    encoding = [0]*(NUM_ATOM_CLASSES)
    try:
        encoding[ATOM_CODES[atomic_num]] = 1.0
    except:
        pass
    return encoding

# pybel.Atom properties to save
PYBEL_PROPS = ['hyb', 'heavyvalence', 'heterovalence', 'partialcharge']
FEATURE_NAMES += PYBEL_PROPS

# Remember if an atom belongs to the ligand or to the protein
FEATURE_NAMES.append('moltype')

# SMARTS definition for other properties
SMARTS = [
    ('[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]', 'hydrophobic'),
    ('[a]', 'aromatic'),
    ('[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]', 'acceptor'),
    ('[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]', 'donor'),
    ('[r]', 'ring')
]

# Compile patterns
PATTERNS = []
for smarts, name in SMARTS:
    PATTERNS.append(pybel.Smarts(smarts))
    FEATURE_NAMES.append(name)


def find_smarts(molecule, patterns=PATTERNS):
    """Find atoms that match SMARTS patterns.

    Parameters
    ----------
    molecule: pybel.Molecule
    patterns: list of pybel.Smarts, optional

    Returns
    -------
    features: np.ndarray
        NxM binary array, where N is the number of atoms in the `molecule` and
        M is the number of patterns. `features[i, j]` == 1.0 if i'th atom has
        j'th property
    """

    features = np.zeros((len(molecule.atoms), len(patterns)))

    for (pattern_id, pattern) in enumerate(patterns):
        atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                   dtype=int) - 1
        features[atoms_with_prop, pattern_id] = 1.0
    return features


def get_features(molecule, moltype=1.0):
    """Get coordinates and features for all heavy atoms in the molecule.

    Parameters
    ----------
    molecule: pybel.Molecule
    moltype: float, optional
        Molecule type. You can use it to encode whether an atom belongs to
        the ligand (1.0) or to the protein (-1.0) etc.

    Returns
    -------
    coords: np.ndarray, shape = (N, 3)
        Coordinates of all heavy atoms in the `molecule`.
    features: np.ndarray, shape = (N, F)
        Features of all heavy atoms in the `molecule`: atom type
        (one-hot encoding), pybel.Atom attributes, type of a molecule
        (e.g protein/ligand distinction), and other properties defined with
        SMARTS patterns
    """

    global PYBEL_PROPS

    coords = []
    features = []
    nonH = []
    i = 0
    for atom in molecule:
        if atom.atomicnum != 1:
            nonH.append(i)
            coords.append(atom.coords)

            features.append(
                encode_num(atom.atomicnum)
                + [atom.__getattribute__(prop) for prop in PYBEL_PROPS]
                + [moltype]
            )
        i += 1

    coords = np.array(coords, dtype=np.float32)
    features = np.array(features, dtype=np.float32)
    features = np.hstack([features, find_smarts(molecule)[nonH]])

    assert ~np.isnan(features).any(), 'got NaN when calculating features'

    return coords, features


def make_grid(coords, features, grid_resolution=1.0, max_dist=10):
    """Covert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Anstroms and atoms that are too far away are not included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    num_features = features.shape[1]

    box_size = int(2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    x, y, z = grid_coords[in_box].T
    grid = np.zeros((1, box_size, box_size, box_size, num_features),
                    dtype=np.float32)
    grid[0, x, y, z] += features[in_box]

    return grid

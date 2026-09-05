"""Pafnucy's atom featuriser and grid, without tfbio.

`tfbio` is the authors' own package. It was published on a conda channel that no longer
resolves, and nothing in this repo or in MD_DL_BA — which uses the same 19 features — carries
a reimplementation. So this is written from scratch and settled by measurement: the repo ships
`tests/data/dataset/*.hdf`, the authors' own featurised output, next to the `tests/data/
complexes/*.mol2` it was made from. `compare_featurizer.py` requires this to reproduce those
arrays column for column.

The 19 features, in the order the grid stores them:

    0-8   one-hot atom type: B C N O P S Se halogen metal
    9     hybridisation
    10    heavy valence      — heavy-atom neighbours
    11    hetero valence     — neighbours that are neither C nor H
    12    partial charge
    13    molcode            — which molecule the atom came from
    14-18 hydrophobic, aromatic, acceptor, donor, ring

The SMARTS below are the patterns tfbio used, which it in turn took from ODDT. They are not
guessed: any that is wrong shows up as a named column in the comparison, which is the point of
running it before trusting this file.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["B", "C", "N", "O", "P", "S", "Se", "halogen", "metal", "hyb",
                 "heavyvalence", "heterovalence", "partialcharge", "molcode",
                 "hydrophobic", "aromatic", "acceptor", "donor", "ring"]

# atomic numbers grouped exactly as the one-hot is laid out
ATOM_CODES: dict[int, int] = {}
for code, atomic in enumerate([5, 6, 7, 8, 15, 16, 34]):        # B C N O P S Se
    ATOM_CODES[atomic] = code
for atomic in (9, 17, 35, 53):                                   # F Cl Br I
    ATOM_CODES[atomic] = 7                                       # halogen
for atomic in ([3, 4, 11, 12, 13] + list(range(19, 32))
               + list(range(37, 51)) + list(range(55, 84))):
    ATOM_CODES[atomic] = 8                                       # metal

SMARTS = {
    "hydrophobic": "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",
    "aromatic": "[a]",
    "acceptor": "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    "donor": "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]",
    "ring": "[r]",
}


def _smarts_indices(mol, pattern: str) -> set[int]:
    """Atom indices matching the pattern. Open Babel reports these 1-based."""
    from openbabel import pybel

    return {idx for match in pybel.Smarts(pattern).findall(mol) for idx in match}


def featurize(mol, molcode: float) -> tuple[np.ndarray, np.ndarray]:
    """One molecule -> (coords, features). Hydrogens are dropped, as tfbio does.

    `molcode` marks which side of the complex an atom came from; Pafnucy passes 1 for the
    pocket and -1 for the ligand, so the network can tell them apart inside a shared grid.
    """
    # One pass over mol.atoms, keeping positions rather than object identities: pybel builds
    # a fresh Atom wrapper on every access, so `id(atom)` differs between two reads of the
    # same underlying atom and any identity-keyed map silently matches nothing.
    heavy, heavy_index_of = [], {}
    for j, atom in enumerate(mol.atoms, start=1):   # openbabel indices are 1-based
        if atom.atomicnum > 1:
            heavy_index_of[j] = len(heavy)
            heavy.append(atom)
    n_all = j
    n = len(heavy)
    coords = np.array([a.coords for a in heavy], dtype=np.float32)

    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    for i, atom in enumerate(heavy):
        code = ATOM_CODES.get(atom.atomicnum)
        if code is not None:
            features[i, code] = 1.0
        features[i, 9] = atom.hyb
        features[i, 10] = atom.heavydegree
        features[i, 11] = atom.heterodegree
        features[i, 12] = atom.partialcharge
        features[i, 13] = molcode

    # SMARTS match against the whole molecule, hydrogens included, so hits come back as
    # 1-based indices into all atoms and have to be mapped onto the heavy-atom subset.
    for offset, name in enumerate(["hydrophobic", "aromatic", "acceptor", "donor", "ring"]):
        for j in _smarts_indices(mol, SMARTS[name]):
            row = heavy_index_of.get(j)
            if row is not None:
                features[row, 14 + offset] = 1.0
    return coords, features


def make_grid(coords: np.ndarray, features: np.ndarray, max_dist: float = 10.0,
              grid_resolution: float = 1.0) -> np.ndarray:
    """Voxelise onto a cube centred on the origin, summing features that share a point.

    Identical in shape and rule to the grid Timenucy/Videonucy build from the same features:
    atoms are snapped to the nearest grid point and anything outside the box is dropped.
    """
    box = int(np.ceil(2 * max_dist / grid_resolution + 1))
    grid = np.zeros((1, box, box, box, features.shape[1]), dtype=np.float32)

    indices = np.round((coords + max_dist) / grid_resolution).astype(int)
    inside = ((indices >= 0) & (indices < box)).all(axis=1)
    for (x, y, z), f in zip(indices[inside], features[inside]):
        grid[0, x, y, z] += f
    return grid

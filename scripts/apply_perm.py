import h5py
import numpy as np

def load_permutation(fname):
    return np.array([int(line.strip()) for line in open(fname)])

def apply_permutation(in_dset, out_dset, permutation):
    out_dset[permutation] = in_dset[:]

def apply_inverse_permutation(in_dset, out_dset, permutation):
    out_dset[:] = in_dset[permutation]

def renumber(in_dset, out_dset, permutation):
    assert in_dset.shape == out_dset.shape
    out_dset[:] = permutation[in_dset.flatten()].reshape(out_dset.shape)

def load_matrix(fname, dtype):
    return np.array([[dtype(a) for a in line.split()]
                     for line in open(fname)])

def test_data_x(point_permutations):
    print('Test data_x: ', end = '')
    # Load data_x
    data_x = load_matrix('orig/data_x', float)
    new_data_x = np.zeros_like(data_x)
    apply_permutation(data_x, new_data_x, point_permutations[1][0])
    new_data_x2 = np.zeros_like(data_x)
    apply_permutation(new_data_x, new_data_x2, point_permutations[1][1])
    # Load reference
    ref = load_matrix('part/data_x', float)
    print('PASS' if np.allclose(ref, new_data_x2) else 'FAIL')

def test_mesh_x(point_permutations, cell_inv_permutations):
    print('Test mesh_x: ', end = '')
    # Load mesh_x, discard header
    mesh_x = load_matrix('orig/mesh_x', int)[1:, :]
    new_mesh_x = np.zeros_like(mesh_x)
    new_mesh_x2 = np.zeros_like(mesh_x)
    apply_inverse_permutation(mesh_x, new_mesh_x, cell_inv_permutations[0])
    apply_inverse_permutation(new_mesh_x, new_mesh_x2, cell_inv_permutations[1])
    renumber(new_mesh_x2, new_mesh_x, point_permutations[1][0])
    renumber(new_mesh_x, new_mesh_x2, point_permutations[1][1])
    # Load reference, discard header
    ref = load_matrix('part/mesh_x', int)[1:, :]
    print('PASS' if np.allclose(ref, new_mesh_x2) else 'FAIL')


def test():
    """
    Test on res_calc
    """
    # pp[data_ind][sequence_ind]
    point_permutations = [[
        load_permutation('point_permutation_{}_{}'.format(mi, i))
        for i in (1, 2)] for mi in range(4)]
    cell_inv_permutations = [
            load_permutation('cell_inv_permutation_all_1'),
            load_permutation('cell_inv_permutation_all_2'),]
    test_data_x(point_permutations)
    test_mesh_x(point_permutations, cell_inv_permutations)


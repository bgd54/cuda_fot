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

def get_cell_permutation(cell_inv_permutations, cell_permutations):
  apply_permutation(np.arange(len(cell_inv_permutations)), cell_permutations,
      cell_inv_permutations)

def apply_permutation_to_dataset(dataset, point_permutations):
  # used on datasets on the 'to set's of the reordered mesh
  tmparr = np.zeros_like(dataset)
  tmpres = np.zeros_like(dataset)
  apply_permutation(dataset, tmparr, point_permutations[0])
  apply_permutation(tmparr, tmpres, point_permutations[1])
  dataset[:] = tmpres

def apply_inv_permutation_to_mesh(mesh, point_permutations, cell_inv_permutations):
  # used on meshes from the reordered set
  new_mesh = np.zeros_like(mesh)
  new_mesh2 = np.zeros_like(mesh)
  apply_inverse_permutation(mesh[()], new_mesh, cell_inv_permutations[0])
  apply_inverse_permutation(new_mesh, new_mesh2, cell_inv_permutations[1])
  renumber(new_mesh2, new_mesh, point_permutations[0])
  renumber(new_mesh, mesh, point_permutations[1])
  #  mesh[:] = new_mesh2

def apply_permutation_to_mesh(mesh, point_permutations_from, point_permutations):
  # used on meshes between 2 set which is a 'to set' of the original mesh
  new_mesh = np.zeros_like(mesh)
  new_mesh2 = np.zeros_like(mesh)
  apply_permutation(mesh[()], new_mesh, point_permutations_from[0])
  apply_permutation(new_mesh, new_mesh2, point_permutations_from[1])
  renumber(new_mesh2, new_mesh, point_permutations[0])
  renumber(new_mesh, mesh, point_permutations[1])

def apply_renumber_to_mesh(mesh, point_permutations):
  # used when from set is not renumbered but to set has permutations
  new_mesh = np.zeros_like(mesh)
  renumber(mesh[()], new_mesh, point_permutations[0])
  renumber(new_mesh, mesh, point_permutations[1])

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


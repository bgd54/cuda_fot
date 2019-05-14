#!/usr/bin/python3
import h5py
import os
import numpy as np
import apply_perm as perm

def apply_permutation_to_dataset(dataset, point_permutations):
  # used on datasets on the 'to set's of the reordered mesh
  tmparr = np.zeros_like(dataset)
  tmpres = np.zeros_like(dataset)
  perm.apply_permutation(dataset, tmparr, point_permutations[0])
  perm.apply_permutation(tmparr, tmpres, point_permutations[1])
  dataset[:] = tmpres

def apply_inv_permutation_to_mesh(mesh, point_permutations, cell_inv_permutations):
  # used on meshes from the reordered set
  new_mesh = np.zeros_like(mesh)
  new_mesh2 = np.zeros_like(mesh)
  perm.apply_inverse_permutation(mesh[()], new_mesh, cell_inv_permutations[0])
  perm.apply_inverse_permutation(new_mesh, new_mesh2, cell_inv_permutations[1])
  perm.renumber(new_mesh2, new_mesh, point_permutations[0])
  perm.renumber(new_mesh, mesh, point_permutations[1])
  #  mesh[:] = new_mesh2

def apply_permutation_to_mesh(mesh, point_permutations_from, point_permutations):
  # used on meshes between 2 set which is a 'to set' of the original mesh
  new_mesh = np.zeros_like(mesh)
  new_mesh2 = np.zeros_like(mesh)
  perm.apply_permutation(mesh[()], new_mesh, point_permutations_from[0])
  perm.apply_permutation(new_mesh, new_mesh2, point_permutations_from[1])
  perm.renumber(new_mesh2, new_mesh, point_permutations[0])
  perm.renumber(new_mesh, mesh, point_permutations[1])

def apply_renumber_to_mesh(mesh, point_permutations):
  # used when from set is not renumbered but to set has permutations
  new_mesh = np.zeros_like(mesh)
  perm.renumber(mesh[()], new_mesh, point_permutations[0])
  perm.renumber(new_mesh, mesh, point_permutations[1])


def main(filename, permdir, refdir):
  basename, ext = os.path.splitext(filename)
  outfile = basename + "_reordered" + ext
  # create a copy of the file
  os.system('cp ' + filename + ' ' + outfile)
  with h5py.File(outfile, 'r+') as freordered:

    #  invariant = {'alpha', 'bedges', 'cells', 'cfl', 'edges', 'eps', 'gam',
    #   'gm1', 'mach', 'nodes', 'p_bound', 'p_qold', 'qinf'}
    #  modified = {'p_adt', 'p_q', 'p_res', 'p_x', 'pbecell', 'pbedge', 'pcell',
    #   'pecell', 'pedge'}

    datasets = ['p_res', 'p_x', 'p_q', 'p_adt']
    # res, adt and q values are space independent, reordering is unnecessary
    for mi in [1]:
      point_permutations = [
        perm.load_permutation(
          os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
        for i in (1, 2)]
      print('Apply permutations on {}'.format(datasets[mi]))
      apply_permutation_to_dataset(freordered[datasets[mi]], point_permutations)

    meshes = ['pecell', 'pedge']
    cell_inv_permutations = [
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_1')),
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_2'))]
    for mi in range(2):
      point_permutations = [
      perm.load_permutation(
        os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
      for i in (1, 2)]
      print('Apply permutations on {}'.format(meshes[mi]))
      apply_inv_permutation_to_mesh(freordered[meshes[mi]], point_permutations,
          cell_inv_permutations)
   
    # reorder pcell: cells->nodes
    print('Apply permutations on {}'.format('pcell'))
    point_permutations = [[
    perm.load_permutation(
      os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
    for i in (1, 2)] for mi in range(2)] # from: perm of res, to: perm of x
    
    apply_permutation_to_mesh(freordered['pcell'], point_permutations[0],
        point_permutations[1])

    # reorder pbecell: bedge -> cells
    # no need to apply permutations just renumber
    print('Apply permutations on {}'.format('pbecell'))
    apply_renumber_to_mesh(freordered['pbecell'], point_permutations[0])

    # reorder pbedge: bedge -> nodes
    # no need to apply permutations just renumber
    print('Apply permutations on {}'.format('pbedge'))
    apply_renumber_to_mesh(freordered['pbedge'], point_permutations[1])



  if refdir != "":
    test_sets = ['data_res', 'data_x', 'data_q', 'data_adt']
    with h5py.File(outfile, 'r') as freordered:
      for mi in [1]:
        print('Test {}'.format(datasets[mi]))
        ref = perm.load_matrix(os.path.join(refdir, test_sets[mi]), float)
        print('PASS' if np.allclose(ref, freordered[datasets[mi]]) else 'FAIL')
        if not np.allclose(ref, freordered[datasets[mi]]):
          for i in range(10):
            print('ref: {} vs: {}'.format(ref[i], freordered[datasets[mi]][i]))

      #mesh tests
      test_sets = ['mesh_res', 'mesh_x']
      for mi in range(2):
        print('Test {}'.format(meshes[mi]))
        ref = perm.load_matrix(os.path.join(refdir, test_sets[mi]), int)[1:, :]
        print('PASS' if np.allclose(ref, freordered[meshes[mi]]) else 'FAIL')
        if not np.allclose(ref, freordered[meshes[mi]]):
          for i in range(10):
            print('ref: {} vs: {}'.format(ref[i], freordered[meshes[mi]][i]))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('filename',
      help='hdf5 file containing airfoil grid')
  parser.add_argument('permdir',
      help='directory containing permutation files')
  parser.add_argument('ref', nargs='?', default='',
      help='directory containing reference for reordered datasets')
  args = parser.parse_args()
  main(args.filename, args.permdir, args.ref)


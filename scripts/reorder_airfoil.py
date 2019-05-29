#!/usr/bin/python3
import h5py
import os
import numpy as np
import apply_perm as perm

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
      perm.apply_permutation_to_dataset(freordered[datasets[mi]],
          point_permutations)

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
      perm.apply_inv_permutation_to_mesh(freordered[meshes[mi]],
          point_permutations, cell_inv_permutations)
   
    # reorder pcell: cells->nodes
    print('Apply permutations on {}'.format('pcell'))
    point_permutations = [[
    perm.load_permutation(
      os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
    for i in (1, 2)] for mi in range(2)] # from: perm of res, to: perm of x
    
    perm.apply_permutation_to_mesh(freordered['pcell'], point_permutations[0],
        point_permutations[1])

    # reorder pbecell: bedge -> cells
    # no need to apply permutations just renumber
    print('Apply permutations on {}'.format('pbecell'))
    perm.apply_renumber_to_mesh(freordered['pbecell'], point_permutations[0])

    # reorder pbedge: bedge -> nodes
    # no need to apply permutations just renumber
    print('Apply permutations on {}'.format('pbedge'))
    perm.apply_renumber_to_mesh(freordered['pbedge'], point_permutations[1])



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


#!/usr/bin/python3
import h5py
import os
import numpy as np
import apply_perm as perm

def main(filename, permdir):
  basename, ext = os.path.splitext(filename)
  outfile = basename + "_reordered" + ext
  # create a copy of the file
  os.system('cp ' + filename + ' ' + outfile)

  with h5py.File(outfile, 'r+') as freordered:
    mi = 1 # we partitioned along el2node only
    point_permutations = [
        perm.load_permutation(
          os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
        for i in (1, 2)] 

    # reorder data on nodes
    datasets = ['ndu', 'ndv', 'ndx', 'ndy', 'ndmass', 'ndarea', 'indtype']
    datasets.extend([
      'rscratch{}'.format(num)
        for num in ['14','15', '111','112','113','114','115']])
    datasets.append('iscratch11')
    datasets.append('zscratch11')
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_permutation_to_dataset(freordered[dset], point_permutations)


    mesh = 'el2node'
    cell_inv_permutations = [
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_1')),
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_2'))]
    print('Apply permutations on {}'.format(mesh))
    perm.apply_inv_permutation_to_mesh(freordered[mesh],
        point_permutations, cell_inv_permutations)
   
    # reorder cellsToCells: el->el
    mesh = 'el2el'
    cell_permutations = cell_inv_permutations[:]
    print('Apply permutations on {}'.format(mesh))
    perm.apply_inv_permutation_to_mesh(freordered[mesh],
        cell_permutations, cell_inv_permutations) # TODO check

    # reorder data on elements
    datasets = ['rho', 'qq', 'csqrd', 'pre', 'ein', 'elmass', 'cnwt', 'cnmass',
        'elx', 'ely', 'qx', 'qy', 'ielsd', 'ielel', 'elidx', 'ielmat',
        'ielreg', 'spmass']
    datasets.extend([
      'rscratch{}'.format(num)
        for num in ['11', '12', '13', '16', '17', '18', '21', '22', '23', '24',
          '25', '26', '27', '28']])
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_renumber_to_dataset(freordered[dset], cell_permutations) # TODO check


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('filename',
      help='hdf5 file containing bookleaf grid')
  parser.add_argument('permdir',
      help='directory containing permutation files')
  args = parser.parse_args()
  main(args.filename, args.permdir)


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
    mi = 0 # we partitioned along el2node only
    point_permutations = [
        perm.load_permutation(
          os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
        for i in (1, 2)] 
    cell_inv_permutations = [
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_1')),
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_2'))]

    print("el: {} vs el: {}".format(freordered['elements'][0], len(cell_inv_permutations[0])))
    print("nodes: {} vs nodes: {}".format(freordered['nodes'][0], len(point_permutations[0])))
    #for i in freordered.keys():
    #    print(i)
    #    for j in range(min(3, np.shape(freordered[i])[0])):
    #        print(freordered[i][j])

    # reorder data on nodes
    datasets = ['ndu', 'ndv', 'ndx', 'ndy', 'ndmass', 'ndarea', 'indtype']
    datasets.extend([
      'rscratch{}'.format(num)
        for num in ['14','15']])#, '111','112','113','114','115'
    #datasets.append('iscratch11')
    #datasets.append('zscratch11')
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_permutation_to_dataset(freordered[dset], point_permutations)


    mesh = 'el2node'
    print('Apply permutations on {}'.format(mesh))
    perm.apply_inv_permutation_to_mesh(freordered[mesh],
        point_permutations, cell_inv_permutations)
   
    # reorder cellsToCells: el->el
    mesh = 'el2el'
    cell_permutations = [np.arange(len(cell_inv_permutations[0])),np.arange(len(cell_inv_permutations[0]))]
    perm.get_cell_permutation(cell_inv_permutations[0], cell_permutations[0])
    perm.get_cell_permutation(cell_inv_permutations[1], cell_permutations[1])
    print('Apply permutations on {}'.format(mesh))
    perm.apply_inv_permutation_to_mesh(freordered[mesh],
        cell_permutations, cell_inv_permutations) # TODO check

    # reorder data on elements
    datasets = ['rho', 'qq', 'csqrd', 'pre', 'ein', 'elmass', 'cnwt', 'cnmass',
        'elx', 'ely', 'qx', 'qy', 'ielsd', 'ielel', 'elidx', 'ielmat',
        'ielreg']#, 'spmass']
    datasets.extend([
      'rscratch{}'.format(num)
        for num in ['11', '12', '13', '21', '22', '23', '24',#'16', '17', '18', 
          '25', '26', '27']])#, '28'
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_permutation_to_dataset(freordered[dset], cell_permutations) # TODO check


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('filename',
      help='hdf5 file containing bookleaf grid')
  parser.add_argument('permdir',
      help='directory containing permutation files')
  args = parser.parse_args()
  main(args.filename, args.permdir)


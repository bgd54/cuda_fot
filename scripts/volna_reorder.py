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

  datasets = {'cellCenters', 'cellEdgeLR', 'cellVolumes',
      'edgeCenters', 'edgeLength', 'edgeNormals',
      'initBathymetry0', 'initBathymetry1', 'initEta', 'isBoundary',
      'nodeCoords'}
  meshes = {
      'cellsToCells', 'cellsToEdges', 'cellsToNodes', 'edgesToCells'}
  
  with h5py.File(outfile, 'r+') as freordered:
    mi = 1 # we partitioned along edgesToCells only
    point_permutations = [
        perm.load_permutation(
          os.path.join(permdir, 'point_permutation_{}_{}'.format(mi, i)))
        for i in (1, 2)] 

    # reorder data on cells
    datasets = ['cellCenters', 'cellEdgeLR', 'cellVolumes',
        'initBathymetry0', 'initBathymetry1', 'initEta'] #, 'values'] 
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_permutation_to_dataset(freordered[dset], point_permutations)


    mesh = 'edgesToCells'
    cell_inv_permutations = [
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_1')),
        perm.load_permutation(os.path.join(permdir,
          'cell_inv_permutation_all_2'))]
    print('Apply permutations on {}'.format(mesh))
    perm.apply_inv_permutation_to_mesh(freordered[mesh],
        point_permutations, cell_inv_permutations)
   
    # reorder cellsToCells: cells->cells
    print('Apply permutations on {}'.format('cellsToCells'))
    perm.apply_permutation_to_mesh(freordered['cellsToCells'],
        point_permutations, point_permutations) # TODO check

    # reorder cellsToNodes: cells->nodes
    mesh = 'cellsToNodes'
    print('Apply permutations on {}'.format(mesh))
    perm.apply_permutation_to_dataset(freordered[mesh], point_permutations)

    # reorder cellsToEdges: cells->edges
    mesh = 'cellsToCells'
    print('Apply permutations on {}'.format(mesh))
    cell_permutations = cell_inv_permutations[:]
    perm.get_cell_permutation(cell_inv_permutations, cell_permutations)
    perm.apply_permutation_to_dataset(freordered[mesh], point_permutations)
    perm.apply_renumber_to_mesh(freordered[mesh], cell_permutations) # TODO check

    # reorder data on edges
    datasets = ['edgeCenters', 'edgeLength', 'edgeNormals']
    for dset in datasets:
      print('Apply permutations on {}'.format(dset))
      perm.apply_renumber_to_dataset(freordered[dset], cell_permutations) # TODO check


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('filename',
      help='hdf5 file containing volna grid (sim12.h5)')
  parser.add_argument('permdir',
      help='directory containing permutation files')
  args = parser.parse_args()
  main(args.filename, args.permdir)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import subprocess as proc
import os

GENERATE_GRID_EXE = '../generate_grid'

def writeData(data, fname):
    f = open(fname, mode = 'w')
    f.write('\n'.join([' '.join(map(str,line)) for line in data]))
    f.flush()
    f.close()

def writeMesh(num_points, num_cells, mesh, fname):
    f = open(fname, mode = 'w')
    f.write('{} {}\n'.format(num_points, num_cells))
    f.write('\n'.join([' '.join(map(str,line)) for line in mesh]))
    f.flush()
    f.close()

def genMesh(grid_dim, fname):
    cmd = [GENERATE_GRID_EXE, fname] + [str(a) for a in grid_dim]
    print('You should generate mesh1 as well, by calling `{}`'\
            .format(' '.join(cmd)))
    # proc.run(cmd, check=True)
    # proc.run(['rm', fname + '.gps', fname + '.metis',\
    #         fname + '.metis_part'], check=True)

def gen(dirname, N, M, num_points0):
    try:
        os.mkdir(dirname)
    except FileExistsError as e:
        pass
    np.random.seed(1)
    num_points1 = N*M
    num_cells = (N-1) * M + (M-1) * N
    genMesh([N, M], dirname + '/mesh1')
    writeData(np.random.rand(num_points0, 2) * 0.001, dirname + '/point_data0')
    writeData(np.random.rand(num_points1, 4) * 0.001, dirname + '/point_data1')
    writeData(np.random.rand(num_cells, 2) * 0.5 + 0.5, dirname + '/cell_data0')
    writeData(np.random.rand(num_cells, 1) * 0.001, dirname + '/cell_data1')
    writeData(np.random.rand(num_cells, 3) * 0.001, dirname + '/cell_data2')
    mesh0 = np.concatenate(
            (np.reshape(np.arange(num_points0),(num_points0,1)),
                np.random.randint(0,num_points0,
                    size=(num_cells - num_points0,1))),
            axis = 0)
    writeMesh(num_points0, num_cells, mesh0, dirname + '/mesh0')

def main():
    gen('mmapping', 100, 200, 10000)
    gen('mmapping_small', 10, 20, 50)
    gen('mmapping_one', 1, 2, 1)

if __name__ == "__main__":
    main()

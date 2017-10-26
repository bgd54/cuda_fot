#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import subprocess as proc


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
    cmd = ['../../generate_grid', fname] + [str(a) for a in grid_dim]
    print('You should generate mesh1 as well, by calling {}'\
            .format(' '.join(cmd)))
    # proc.run(cmd, check=True)
    # proc.run(['rm', fname + '.gps', fname + '.metis',\
    #         fname + '.metis_part'], check=True)

def main():
    np.random.seed(1)
    N = 100
    M = 200
    num_points0 = 100
    num_points1 = N*M
    num_cells = (N-1) * M + (M-1) * N
    genMesh([N, M], 'mesh1')
    writeData(np.random.rand(num_points0, 2) * 0.001, 'point_data0')
    writeData(np.random.rand(num_points1, 4) * 0.001, 'point_data1')
    writeData(np.random.rand(num_cells, 2) * 0.5 + 0.5, 'cell_data0')
    writeData(np.random.rand(num_cells, 1) * 0.001, 'cell_data1')
    writeData(np.random.rand(num_cells, 3) * 0.001, 'cell_data2')
    writeMesh(num_points0, num_cells,\
            np.random.randint(0,100,size=(num_cells,1)), 'mesh0')


if __name__ == "__main__":
    main()

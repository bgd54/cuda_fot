from __future__ import print_function,division,with_statement,nested_scopes

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import sys
import argparse


# get_ipython().magic('matplotlib')

def array_sort (a):
    return np.array(sorted([(i1,i2) for i1,i2 in a]))


def transform (in_file, out_file, vis):
    el_in = np.array([[int(a) for a in line.lstrip().rstrip().split()] \
            for line in open(in_file)])
    A = scipy.sparse.csr_matrix((np.ones((el_in[0,1],)),\
            (el_in[1:,0],el_in[1:,1])),\
            shape=(el_in[0,0],el_in[0,0]))
    p = scipy.sparse.csgraph.reverse_cuthill_mckee(A)
    p = np.argsort(p)
    el_out = p[el_in[1:,:]]
    el_out = array_sort(el_out)
    bw_orig = np.max(np.abs(el_in[1:,0] - el_in[1:,1]))
    mbw_orig = np.mean(np.abs(el_in[1:,0] - el_in[1:,1]))
    bw_out = np.max(np.abs(el_out[:,0] - el_out[:,1]))
    mbw_out = np.mean(np.abs(el_out[:,0] - el_out[:,1]))
    print('Original bandwidth:',bw_orig)
    print('Original mean bandwidth:',mbw_orig)
    print('RCM''d bandwidth:',bw_out)
    print('RCM''d mean bandwidth:',mbw_out)
    with open(out_file,mode='w') as f:
        f.write('{} {}\n'.format(el_in[0,0],el_in[0,1]))
        f.write('\n'.join([' '.join(map(str,a)) for a in el_out]))
        f.flush()
    if vis:
        import matplotlib.pyplot as plt
        B = scipy.sparse.csr_matrix((np.ones((el_in[0,1],)),\
                (el_out[:,0],el_out[:,1])),\
                shape=(el_in[0,0],el_in[0,0]))
        plt.subplot(1,2,1)
        plt.spy(A,markersize=0.01)
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.spy(B,markersize=0.01)
        plt.title('RCM''d')
        plt.show()

def genArgParser ():
    parser = argparse.ArgumentParser(\
            description='apply Reverse Cuthill McKee to a graph')
    parser.add_argument('in_file',help='input filename')
    parser.add_argument('out_file',help='output filename')
    parser.add_argument('-v','--vis',action='store_true',dest='vis')
    return parser

def main ():
    parser = genArgParser()
    args = parser.parse_args()
    transform(args.in_file, args.out_file, args.vis)

if __name__ == "__main__":
    main()

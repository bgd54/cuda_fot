import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import argparse

def genArgParser ():
    parser = argparse.ArgumentParser(\
            description = 'visualise the graph on a spy diagram')
    parser.add_argument('file',nargs='+')
    return parser

def vis (files):
    fig,axarr = plt.subplots(1,len(files))
    if len(files) == 1:
        axarr = [axarr]
    for i,fname in enumerate(files):
        el = np.array([[int(a) for a in line.lstrip().rstrip().split()]\
                for line in open(fname)])
        num_points, num_edges = el[0,:]
        el = el[1:,:]
        print('-- ' + fname)
        print('Max bandwidth:', np.max(np.abs((el[:,0])-(el[:,1]))))
        print('Mean bandwidth:', np.mean(np.abs((el[:,0])-(el[:,1]))))
        data = np.array([1]*2*num_edges)
        csr = scipy.sparse.coo_matrix((data,\
                (np.r_[el[:,1],el[:,0]],np.r_[el[:,0],el[:,1]])),\
                shape=(num_points,num_points))
        axarr[i].spy(csr,markersize=3)
        axarr[i].set_title(fname)
    plt.show()

def main ():
    parser = genArgParser()
    args = parser.parse_args()
    vis(args.file)

if __name__ == "__main__":
    main()

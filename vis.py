import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

el1 = np.array([[int(a) for a in line.lstrip().rstrip().split()] for line in open('./my_edge_list')])
num_points, num_edges = el1[0,:]
el1 = el1[1:,:]
el2= np.array([[int(a) for a in line.lstrip().rstrip().split()] for line in open('./scotch_reordered_edge_list')])
el2 = el2[1:,:]
data = np.array([1]*2*num_edges)
csr1 = scipy.sparse.coo_matrix((data,\
        (np.r_[el1[:,1],el1[:,0]],np.r_[el1[:,0],el1[:,1]])),\
        shape=(num_points,num_points))
csr2 = scipy.sparse.coo_matrix((data,\
        (np.r_[el2[:,1],el2[:,0]],np.r_[el2[:,0],el2[:,1]])),\
        shape=(num_points,num_points))
plt.figure()
plt.subplot(1,2,1)
plt.title('mine')
plt.spy(csr1,markersize=3)
plt.subplot(1,2,2)
plt.title('rcm')
plt.spy(csr2,markersize=3)
plt.show()

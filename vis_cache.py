import matplotlib.pyplot as plt
import numpy as np

BLOCK_DIMS = [
    (0,32),
    (2,8),
    (4,4),
    (0,128),
    (2,32),
    (4,16),
    (8,8),
    (0,288),
    (2,72),
    (4,36),
    (12,12),
    (0,512),
    (2,128),
    (4,64),
    (8,32),
    (16,16)]

def plot_file (cache_count,names,dims):
    # shape: name, reord, AOS/SOA, dim
    assert cache_count.shape[0] == len(names)
    assert cache_count.shape[1] == 4, cache_count.shape[1]
    assert cache_count.shape[2] == 2, cache_count.shape[2]
    assert cache_count.shape[3] == len(dims), cache_count.shape[3]
    plt.figure()
    ind = np.arange(len(names)*4)
    plt.suptitle('Normalised to block size')
    xt = [a + ' ' + b for a in names \
            for b in ['row major','squared','GPS','RCM']]
    norm = np.array([128] * len(names) * 4)
    for i,d in enumerate(dims):
        plt.subplot(1,len(dims),i+1)
        plt.bar(ind,cache_count[:,:,1,i].view().flatten()/norm,0.25)
        plt.bar(ind+0.25,cache_count[:,:,0,i].view().flatten()/norm,0.25)
        plt.legend(['SOA','AOS'])
        plt.title('Dim: ' + str(d))
        plt.xticks(ind,xt,rotation=90)
    plt.show()

def plot_cache (cache_count,block_dims,dims,normalise):
    # shape: SOA,dims,block_dims
    assert cache_count.shape[0] == 2
    assert cache_count.shape[1] == len(dims)
    assert cache_count.shape[2] == len(block_dims)
    plt.figure()
    ind = np.arange(len(block_dims))
    if normalise:
        plt.suptitle('Normalised to block size')
        norm = [2*a*b if a != 0 else b for a,b in block_dims]
    else:
        plt.suptitle('Not normalised')
        norm = [1 for a,b in block_dims]
    for i,d in enumerate(dims):
        plt.subplot(1,len(dims),i+1)
        plt.bar(ind,cache_count[0,i,:]/norm, 0.25)
        plt.bar(ind + 0.25,cache_count[1,i,:]/norm, 0.25)
        plt.legend(['SOA','AOS'])
        plt.ylabel('Avg. cache lines / block')
        plt.xticks(ind,map(str,block_dims),rotation=90)
        plt.title('Dim = ' + str(d))
    plt.show()


def plot_block (cache_count):
    plot_cache(cache_count, BLOCK_DIMS, [1,2,4,8],False)
    plot_cache(cache_count, BLOCK_DIMS, [1,2,4,8],True)
    


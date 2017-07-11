import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

'''
To load bandwidths:

names = ['1025x1025','2049x1025','2049x2049','4097x2049','rotor37']
# names, reorderings, AOS/SOA, float/double, dim, implementation
bw = np.zeros((len(names),4,2,2,4,4))

bw_ = np.array([float(line.strip()) for line in open('./bandwidths')])
bw_.shape = (-1,2,2,4,4)
bw_.shape
bw_1 = bw_[:16,:,:,:,:]
bw_2 = bw_[16:,:,:,:,:]
bw_1.shape
bw_1.shape = (4,4,2,2,4,4)
bw_2.shape
bw_2.shape = (1,2,2,2,4,4)
bw.shape
bw[:4,:,:,:,:,:] = bw_1
bw[4,0,:,:,:,:] = bw_2[0,0,:,:,:,:]
bw[4,2,:,:,:,:] = bw_2[0,1,:,:,:,:]
plot_all(bf[:,:,:,0,:,:],names,y_height=6e10)  # float
plot_all(bf[:,:,:,1,:,:],names,y_height=6e10)  # double
'''

def plot_all_dims (bw, names, y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 4, 'Num of reords is not 4 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 4, 'Dim is not 4 ({})'.format(bw.shape[3])
    assert bw.shape[4] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[4])
    plt.figure(figsize=(10,4))
    plt.title('2layer')
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.plot(bw[:,:,0,i,3])
        plt.plot(bw[:,:,1,i,3],':')
        plt.title('Dim = ' + str([1,4,8,16][i]))
        legend_text = [a + ' - ' + b\
                for a in ['AOS','SOA']\
                for b in ['row_major','square_blocks','GPS','RCM']]
        i == 0 and plt.legend(legend_text)
        plt.ylim([0,y_height])
        plt.ylabel('B/s')
        plt.xticks(np.arange(len(names)),names,rotation=45)
        plt.tight_layout()
    plt.show()

def plot_2layer_global (bw, names, y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 4, 'Num of reords is not 4 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 4, 'Dim is not 4 ({})'.format(bw.shape[3])
    assert bw.shape[4] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[4])
    plt.figure(figsize=(11,6))
    plt.title('2layer')
    ss = bw.shape[0] * bw.shape[1]
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.plot(np.arange(0,ss,4),bw[:,0,0,i,3].flatten(),'bo')
        plt.plot(np.arange(1,ss,4),bw[:,1,0,i,3].flatten(),'ko')
        plt.plot(np.arange(2,ss,4),bw[:,2,0,i,3].flatten(),'yo')
        plt.plot(np.arange(3,ss,4),bw[:,3,0,i,3].flatten(),'ro')
        plt.plot(np.arange(0,ss,4),bw[:,0,0,i,2].flatten(),'bx')
        plt.plot(np.arange(1,ss,4),bw[:,1,0,i,2].flatten(),'kx')
        plt.plot(np.arange(2,ss,4),bw[:,2,0,i,2].flatten(),'yx')
        plt.plot(np.arange(3,ss,4),bw[:,3,0,i,2].flatten(),'rx')
        plt.plot(np.arange(0,ss,4),bw[:,0,1,i,3].flatten(),'b^')
        plt.plot(np.arange(1,ss,4),bw[:,1,1,i,3].flatten(),'k^')
        plt.plot(np.arange(2,ss,4),bw[:,2,1,i,3].flatten(),'y^')
        plt.plot(np.arange(3,ss,4),bw[:,3,1,i,3].flatten(),'r^')
        plt.plot(np.arange(0,ss,4),bw[:,0,1,i,2].flatten(),'bs')
        plt.plot(np.arange(1,ss,4),bw[:,1,1,i,2].flatten(),'ks')
        plt.plot(np.arange(2,ss,4),bw[:,2,1,i,2].flatten(),'ys')
        plt.plot(np.arange(3,ss,4),bw[:,3,1,i,2].flatten(),'rs')
        plt.title('Dim = ' + str([1,4,8,16][i]))
        legend_text = [a + ' - ' + b + ' - ' + c\
                for c in ['AOS','SOA']\
                for a in ['2layer','global'] \
                for b in ['row major','square','GPS','RCM']]
        i == 1 and plt.legend(legend_text)
        plt.ylim([0,y_height])
        plt.ylabel('B/s')
        plt.xticks(np.arange(0,16,4),names,rotation=45)
        plt.tight_layout()

def plot_along_dims (bw,names,y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 4, 'Num of reords is not 4 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 4, 'Dim is not 4 ({})'.format(bw.shape[3])
    assert bw.shape[4] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[3])
    fig, ax = plt.subplots(figsize = (6,8))
    ax.set_prop_cycle(cycler('linestyle',['-','--',':','-.']) *\
            cycler('color','rgbkc')*\
            cycler('linewidth',[0.5,1.5]))
    ax.plot(bw[:,:,:,:,3].view().reshape(-1,4).T)
    legend_text = [a + ' - ' + b + ' - ' + c for a in names \
            for b in ['row major','square','GPS','RCM']\
            for c in ['AOS','SOA']]
    ax.legend(legend_text,ncol=2)
    ax.set_ylim([0,2.3*y_height])
    ax.set_ylabel('B/s');
    ax.set_xlabel('Dimension')
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(map(str,[1,4,8,16]));
    ax.set_title('2layer, all reorderings, along dimensions')
    fig.tight_layout()

def plot_all (bw,names, y_height = 4e10):
    plot_all_dims(bw,names,y_height)
    plot_2layer_global(bw,names,y_height)
    plot_along_dims(bw,names,y_height)

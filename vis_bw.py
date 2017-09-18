import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

'''
To load bandwidths:

names = ['1025x1025','2049x1025','2049x2049','4097x2049','rotor37']
# names, reorderings, AOS/SOA, float/double, edgedim, pointdim, implementation
bw = np.zeros((len(names),4,2,2,4,4))

bw = np.array([float(line.strip()) for line in open('./bandwidths')])
bw.shape = (-1,2,2,2,4,4)
plot_all(bf[:,:,:,0,:,:,:],names,y_height=6e10)  # float
plot_all(bf[:,:,:,1,:,:,:],names,y_height=6e10)  # double
'''

def plot_all_dims (bw, names, y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 3, 'Num of reords is not 3 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 2, 'Num of EdgeDim variations is not 2 ({})'\
            .format(bw.shape[3])
    assert bw.shape[4] == 4, 'Dim is not 4 ({})'.format(bw.shape[4])
    assert bw.shape[5] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[5])
    plt.figure(figsize=(13,6))
    plt.title('2layer')
    for _i in range(5):
        for j in range(2):
            i = min(3,_i)
            plt.subplot(2,5,5 * j + _i + 1)
            plt.plot(bw[:,:,0,j,i,3])
            plt.plot(bw[:,:,1,j,i,3],':')
            pdim = [1,4,8,16][i]
            plt.title('Dim = {}, EdgeDim = {}'\
                    .format(str(pdim),str([1,pdim][j])))
            plt.ylim([0,y_height])
            plt.ylabel('B/s')
            plt.xticks(np.arange(len(names)),names,rotation=45,ha='right')
            plt.tight_layout()
            legend_text = [a + ' - ' + b\
                    for a in ['AOS','SOA']\
                    for b in ['original','GPS','METIS']]
            _i == 4 and plt.legend(legend_text)
    plt.show()

def plot_2layer_global (bw, names, y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 3, 'Num of reords is not 3 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 2, 'Num of EdgeDim variations is not 2 ({})'\
            .format(bw.shape[3])
    assert bw.shape[4] == 4, 'Dim is not 4 ({})'.format(bw.shape[4])
    assert bw.shape[5] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[5])
    plt.figure(figsize=(13,8))
    plt.title('2layer')
    ss = bw.shape[0] * bw.shape[1]
    for _i in range(5):
        for j in range(2):
            i = min(3,_i)
            plt.subplot(2,5,j * 5 + _i + 1)
            plt.plot(np.arange(0,ss,3),bw[:,0,0,j,i,3].flatten(),'bo')
            plt.plot(np.arange(1,ss,3),bw[:,1,0,j,i,3].flatten(),'ko')
            plt.plot(np.arange(2,ss,3),bw[:,2,0,j,i,3].flatten(),'yo')
            plt.plot(np.arange(0,ss,3),bw[:,0,0,j,i,2].flatten(),'bx')
            plt.plot(np.arange(1,ss,3),bw[:,1,0,j,i,2].flatten(),'kx')
            plt.plot(np.arange(2,ss,3),bw[:,2,0,j,i,2].flatten(),'yx')
            plt.plot(np.arange(0,ss,3),bw[:,0,1,j,i,3].flatten(),'b^')
            plt.plot(np.arange(1,ss,3),bw[:,1,1,j,i,3].flatten(),'k^')
            plt.plot(np.arange(2,ss,3),bw[:,2,1,j,i,3].flatten(),'y^')
            plt.plot(np.arange(0,ss,3),bw[:,0,1,j,i,2].flatten(),'bs')
            plt.plot(np.arange(1,ss,3),bw[:,1,1,j,i,2].flatten(),'ks')
            plt.plot(np.arange(2,ss,3),bw[:,2,1,j,i,2].flatten(),'ys')
            pdim = [1,4,8,16][i]
            plt.title('Dim = {}, EdgeDim = {}'\
                    .format(str(pdim),str([1,pdim][j])))
            plt.ylim([0,y_height])
            plt.ylabel('B/s')
            plt.xticks(np.arange(bw.shape[1]//2,ss,bw.shape[1]),names,\
                    rotation=45,ha='right')
            legend_text = [a + ' - ' + b + ' - ' + c\
                    for c in ['AOS','SOA']\
                    for a in ['2layer','global'] \
                    for b in ['original','GPS','METIS']]
            _i == 4 and plt.legend(legend_text)
    plt.tight_layout()

def plot_cache_bw (bw, names, y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 3, 'Num of reords is not 3 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 2, 'Num of EdgeDim variations is not 2 ({})'\
            .format(bw.shape[3])
    assert bw.shape[4] == 4, 'Dim is not 4 ({})'.format(bw.shape[4])
    assert bw.shape[5] == 2, 'Last axis should be BW and cache BW ({})'\
            .format(bw.shape[5])
    plt.figure(figsize=(13,8))
    plt.title('2layer')
    ss = bw.shape[0] * bw.shape[1]
    for _i in range(5):
        for j in range(2):
            i = min(3,_i)
            plt.subplot(2,5,j * 5 + _i + 1)
            plt.plot(np.arange(0,ss,3),bw[:,0,0,j,i,0].flatten(),'bo')
            plt.plot(np.arange(1,ss,3),bw[:,1,0,j,i,0].flatten(),'ko')
            plt.plot(np.arange(2,ss,3),bw[:,2,0,j,i,0].flatten(),'yo')
            plt.plot(np.arange(0,ss,3),bw[:,0,0,j,i,1].flatten()*100,'bx')
            plt.plot(np.arange(1,ss,3),bw[:,1,0,j,i,1].flatten()*100,'kx')
            plt.plot(np.arange(2,ss,3),bw[:,2,0,j,i,1].flatten()*100,'yx')
            plt.plot(np.arange(0,ss,3),bw[:,0,1,j,i,0].flatten(),'b^')
            plt.plot(np.arange(1,ss,3),bw[:,1,1,j,i,0].flatten(),'k^')
            plt.plot(np.arange(2,ss,3),bw[:,2,1,j,i,0].flatten(),'y^')
            plt.plot(np.arange(0,ss,3),bw[:,0,1,j,i,1].flatten()*100,'bs')
            plt.plot(np.arange(1,ss,3),bw[:,1,1,j,i,1].flatten()*100,'ks')
            plt.plot(np.arange(2,ss,3),bw[:,2,1,j,i,1].flatten()*100,'ys')
            pdim = [1,4,8,16][i]
            plt.title('Dim = {}, EdgeDim = {}'\
                    .format(str(pdim),str([1,pdim][j])))
            plt.ylim([0,y_height])
            plt.ylabel('B/s')
            plt.xticks(np.arange(bw.shape[1]//2,ss,bw.shape[1]),names,\
                    rotation=45,ha='right')
            legend_text = [a + ' - ' + b + ' - ' + c\
                    for c in ['AOS','SOA']\
                    for a in ['bandwidth','cache-line bw * 100'] \
                    for b in ['original','GPS','METIS']]
            _i == 4 and plt.legend(legend_text)
    plt.tight_layout()

def plot_along_dims (bw,names,y_height = 4e10):
    assert bw.shape[0] == len(names), (bw.shape[0], len(names))
    assert bw.shape[1] == 3, 'Num of reords is not 4 ({})'.format(bw.shape[1])
    assert bw.shape[2] == 2, 'SAO/AOS missing ({})'.format(bw.shape[2])
    assert bw.shape[3] == 2, 'Num of EdgeDim variations is not 2 ({})'\
            .format(bw.shape[3])
    assert bw.shape[4] == 4, 'Dim is not 4 ({})'.format(bw.shape[4])
    assert bw.shape[5] == 4, 'Num of implementations is not 4 ({})'\
            .format(bw.shape[5])
    fig, ax = plt.subplots(figsize = (12,8))
    ax.set_prop_cycle(cycler('linestyle',['-','--',':','-.']) *\
            cycler('color','rgbkc')*\
            cycler('linewidth',[0.5,1.0,1.5]))
    ax.plot(bw[:,:,:,:,:,3].view().reshape(-1,4).T)
    legend_text = ['{} - {} - {} - {}'.format(a,b,c,d)\
            for a in names \
            for b in ['original','GPS','METIS']\
            for c in ['AOS','SOA']\
            for d in ['EDim=1','Edim=Pdim']]
    ax.legend(legend_text,ncol=4)
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

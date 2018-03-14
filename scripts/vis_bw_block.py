import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot

'''
To load bandwidths:
bw = np.zeros((2,4,16,2))
names = ['0x32','2x8','4x4','0x128', '2x32', '4x16', '8x8', '0x288', '2x72', '4x36', '12x12', '0x512', '2x128', '4x64', '8x32', '16x16']

bw_ = np.array([float(line.strip()) for line in open('./bandwidths')])

bw_.shape = (-1,16,2)
bw_.shape

bw_1 = bw_[:4,:,:]
bw_2 = bw_[4:,:,:]
bw[0,:,:,:] = bw_1
bw[1,:,:,:] = bw_2

rf = np.array([float(line.strip()) for line in open('./reuse_factors')])
'''


def plot_block_versions(bw, names, offset=1):
    plt.figure()
    offsets = ['float', 'double']
    plt.suptitle('Hierarchical blocks (' + offsets[offset] + ')')
    blocksizes = ['32', '128', '288', '512']
    ind = 0;
    for i in range(len(blocksizes)):
        plt.subplot(4,1,i+1)
        plt.title('bs: ' + blocksizes[i])
        start = ind
        plt.plot(bw[offset,0,:,ind,1])
        ind += 1
        plt.plot(bw[offset,0,:,ind,1])
        ind += 1
        plt.plot(bw[offset,0,:,ind,1])
        ind += 1
        if i > 0:
            plt.plot(bw[offset,0,:,ind,1])
            ind += 1
        if i > 1:
            plt.plot(bw[offset,0,:,ind,1])
            ind += 1
        ind = start
        plt.plot(bw[offset,1,:,ind,1],':')
        ind += 1
        plt.plot(bw[offset,1,:,ind,1],':')
        ind += 1
        plt.plot(bw[offset,1,:,ind,1],':')
        ind += 1
        if i > 0:
            plt.plot(bw[offset,1,:,ind,1],':')
            ind += 1
        if i > 1:
            plt.plot(bw[offset,1,:,ind,1],':')
            ind += 1
        legend_text = [a + ' - ' + b\
                for a in ['SOA','AOS']\
                for b in names[start : ind]]
        plt.legend(legend_text)
        plt.ylabel('B/s')
        plt.xlabel('Dim')
        plt.xticks(range(4),[1, 2, 4, 8])
    plt.show()

def plot_block_versions_by_dim(bw, names, offset=1):
    plt.figure()
    offsets = ['float', 'double']
    plt.suptitle('Hierarchical blocks (' + offsets[offset] + ')')
    width = 0.1
    index = np.arange(len(bw[offset,0,:,0,1]))
    color = plt.cm.rainbow(np.linspace(0,1,len(names)))
    for i,c in zip(range(len(names)),color):
        plt.plot(index*2+i*width, bw[offset,0,:,i,1], color=c,marker='o',linestyle='none')
        plt.plot(index*2+i*width, bw[offset,1,:,i,1], color=c,marker='x',linestyle='none')

    legend_text = [a + ' - ' + b\
            for b in names\
            for a in ['SOA','AOS']]
    plt.legend(legend_text,ncol=2)
    plt.ylabel('B/s')
    plt.xlabel('Dim')
    plt.xticks(index*2+8*width,[1, 2, 4, 8])
    
    plt.show()


def plot_reuse_factors(rf, names):
    plt.figure()
    plt.suptitle('Reuse factors')

    bar_width = 0.35
    index = np.arange(len(rf))
    plt.bar(index, rf, bar_width,
                 color='b')
    plt.xticks(index, names)
    plt.ylabel('reuse factor')
    plt.xlabel('version')    
    plt.show()

def plot_all():
    global bw, rf, names
    plot_block_versions(bw,names,0)
    plot_block_versions(bw,names,1)
    plot_reuse_factors(rf,names)
    plot_block_versions_by_dim(bw, names, 1)
    plot_block_versions_by_dim(bw, names, 0)

def plot_3D(bw, blocks, rf, cl, nc):
    N = len(blocks)
    mx = np.max(bw)
    plt.figure(figsize = (5.5, 3.5))
    host = host_subplot(111, axes_class = AA.Axes)
    plt.subplots_adjust(right = 0.83)
    par1 = host.twinx()
    par2 = host.twinx()
    offset = 35
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis['right'] = new_fixed_axis(loc = 'right',
                                        axes = par2,
                                        offset = (offset, 0))
    par2.axis['right'].toggle(all = True)
    par1.axis['right'].toggle(all = True)

    host.set_ylabel('Bandwidth (GB/s)')
    host.set_xlabel('Block dimensions')
    # host.bar(np.arange(N)-0.2, bw[0,:]/1e9,0.4, hatch = '//', fill = False)
    host.bar(np.arange(N)-0.2, bw[0,:]/1e9,0.4, color = (0.65, 0.65, 0.65))
    # host.bar(np.arange(N)+0.2, bw[1,:]/1e9,0.4, fill = False, hatch = '\\\\')
    host.bar(np.arange(N)+0.2, bw[1,:]/1e9,0.4, color = (0.45, 0.45, 0.45))

    par1.set_ylabel('Thread colour')
    par1.plot(np.arange(N), nc[0,:], 'ks')
    par1.set_ylim([0, 1.4 * np.max(nc)])

    par2.set_ylabel('Reuse factor')
    par2.plot(np.arange(N), 1/rf[0,:], 'kv', markersize = 7)
    par2.set_ylim([0, 1.3 * np.max(1/rf)])

    # plt.plot(np.arange(2 * N)/2-0.2,cl.T.flatten()*1e7, 'y*')
    # plt.legend(['reuse','threadcol','cacheline','SOA','AOS'],loc='upper left')
    host.set_xticks(np.arange(N))
    host.set_xticklabels(map(str,blocks), rotation = 90, ha = 'right')
    host.set_ylim([0, 1.50 * mx/1e9])

    host.legend(['SOA', 'AOS', 'Thread colour', 'Reuse factor'],
                loc = 'upper left')

    plt.draw()
    plt.show()
    plt.tight_layout()

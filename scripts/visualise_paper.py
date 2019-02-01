import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.ticker import MultipleLocator

plt.rc('axes', axisbelow = True)

# No global or nonreordered versions for  miniAero
# Also too high bandwidths are plotted as zero
MINI_AERO=False

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'

bss_ = '128 196 256 288 320 352 384 416 448 480 512'.split()
bss = list(map(int,bss_))
data_sizes = {
  'res_calc' : (721801 * 2 * 8 * 1
               + 720000 * 4 * 8 * 2
               + 720000 * 5 * 8 * 1
               + 1438600 * 2 * 4 * 2),
  'res_calc_big' : (2883601 * 2 * 8 * 1
               + 2880000 * 4 * 8 * 2
               + 2880000 * 5 * 8 * 1
               + 5757200 * 2 * 4 * 2),
  'bookleaf' : (4005001 * 4 * 8 * 2
               + 4000000 * 17 * 8 * 1
               + 4000000 * 4 * 4 * 1),
  'volna' : (2392352 * 4 * 4 * 2
               + 2392352 * 1 * 4 * 1
               + 3589735 * 7 * 4 * 1
               + 3589735 * 1 * 4 * 1
               + 3589735 * 2 * 4 * 1),
  'lulesh' : (5000211 * 3 * 8 * 2
               + 5000211 * 3 * 8 * 1
               + 4913000 * 3 * 8 * 1
               + 4913000 * 1 * 8 * 1
               + 4913000 * 8 * 4 * 1),
  'mini_aero' : (2097152 * 5 * 8 * 2
               + 2097152 * 28 * 8 * 1
               + 6242304 * 12 * 8 * 1
               + 6242304 * 2 * 4 * 1),
  'lulesh_block' : (2146689 * 3 * 8 * 2
                    + 2146689 * 3 * 8 * 1
                    + 2097152 * 3 * 8 * 1
                    + 2097152 * 1 * 8 * 1
                    + 2097152 * 8 * 4 * 1),
  'mini_aero_block' : (715563 * 5 * 8 * 2
                       + 715563 * 28 * 8 * 1
                       + 2118954 * 12 * 8 * 1
                       + 2118954 * 2 * 4 * 1),
}

orig_bw_airfoil = data_sizes['res_calc_big'] / 3.82 * 2000
orig_bw_volna = data_sizes['volna'] / 0.56 * 600
orig_bw_bookleaf = data_sizes['bookleaf'] / 0.15 * 40
orig_bw_lulesh = 9.66601e10
# orig_bw_mini_aero = 8.34999e+10
orig_bw_mini_aero_atomic = 9.58461e+10
# orig_bw_mini_aero2 = 6.64361e+10
orig_bw_mini_aero_array = 6.94593e+10

# Volta stats
orig_bw_lulesh_volta = 1.48009e+11
orig_bw_mini_aero_atomic_volta = 1.04613e+11
orig_bw_mini_aero_array_volta = 1.40627e+11
orig_bw_airfoil_volta = data_sizes['res_calc_big'] / 2.517 * 2000

# def autolabel (rects):
#     for ii,rect in enumerate(rects):
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width() / 2, 1.05 * height, str(height))


def get_bw (fname, data_size):
    times = np.array([float(line.strip().split(':')[1].strip().split()[0])
                      for line in open(fname)
                      if '  time:      ' in line])
    if MINI_AERO:
        times.shape = (len(bss),2,2,3)
    else:
        times.shape = (len(bss),2,3,5)
    bw = data_size / times * 1000 * 500
    bw_hier = bw[:,:,:,0:3]
    if MINI_AERO:
        bw_glob = None
    else:
        bw_glob = bw[:,:,:,3:5]
    return (bw_hier, bw_glob)


def bw_vs_bs (fname, data_size):
    bw_hier, bw_glob = get_bw(fname, data_size)
    if MINI_AERO:
        bw_hier[bw_hier > 7e11] = 0

    figsize = (5, 2.5)

    plt.figure(figsize = figsize)
    # gs = gridspec.GridSpec(2, 1,
    #                        height_ratios = [2, 1])
    gs = gridspec.GridSpec(1, 2)
    plt.subplot(gs[0])
    plt.plot(np.arange(len(bss)), bw_hier[:,0,0,0] / 1e9, ':', lw=2.5)
    plt.plot(np.arange(len(bss)), bw_hier[:,0,1,0] / 1e9, '-')
    if not MINI_AERO:
        plt.plot(np.arange(len(bss)), bw_hier[:,0,2,0] / 1e9, '--')
    plt.title('AOS')
    plt.ylabel('GB/s')
    bw_max = np.max(bw_hier[:,:,:,0]) / 1e9
    plt.ylim([0.0 * bw_max,1.1*bw_max])
    plt.grid(True)
    # plt.xticks(np.arange(0,len(bss),5),bss_[::5],rotation=90)
    plt.xticks(np.arange(len(bss)),bss_,rotation=90)
    plt.xlabel('Block size')
    if MINI_AERO:
        plt.legend(['GPS','partitioned'])
    else:
        plt.legend(['non-reordered','GPS','partitioned'])
    plt.subplot(gs[1])
    plt.plot(np.arange(len(bss)), bw_hier[:,1,0,0] / 1e9, ':', lw=2.5)
    plt.plot(np.arange(len(bss)), bw_hier[:,1,1,0] / 1e9, '-')
    if not MINI_AERO:
        plt.plot(np.arange(len(bss)), bw_hier[:,1,2,0] / 1e9, '--')
    plt.title('SOA')
    plt.ylabel('GB/s')
    plt.ylim([0.0*bw_max, 1.1*bw_max])
    plt.grid(True)
    plt.xticks(np.arange(len(bss)),bss_,rotation=90)
    plt.xlabel('Block size')
    # plt.legend(['non-reordered','GPS','partitioned'])
    plt.tight_layout()
    plt.show()

    # Don't need global for now
    # plt.figure(figsize = figsize)
    # gs = gridspec.GridSpec(2, 1,
    #                        height_ratios = [2, 1])
    # plt.subplot(gs[0])
    # plt.plot(np.arange(len(bss)), bw_glob[:,0,0,0], ':', lw=2.5)
    # plt.plot(np.arange(len(bss)), bw_glob[:,0,1,0], '-')
    # plt.plot(np.arange(len(bss)), bw_glob[:,0,2,0], '--')
    # plt.title('AOS')
    # plt.ylabel('B/s')
    # bw_max = np.max(bw_glob[:,:,:,0])
    # plt.grid(True)
    # plt.ylim([0*bw_max,0.9*bw_max])
    # plt.xticks(np.arange(len(bss) + 1),bss_)
    # plt.xlabel('Block size')
    # plt.legend(['non-reordered','GPS','partitioned'])
    # plt.subplot(gs[1])
    # plt.plot(np.arange(len(bss)), bw_glob[:,1,0,0], ':', lw=2.5)
    # plt.plot(np.arange(len(bss)), bw_glob[:,1,1,0], '-')
    # plt.plot(np.arange(len(bss)), bw_glob[:,1,2,0], '--')
    # plt.title('SOA')
    # plt.ylabel('B/s')
    # plt.ylim([0.0*bw_max,1.1*bw_max])
    # plt.grid(True)
    # plt.xticks(np.arange(len(bss) + 1),bss_)
    # plt.xlabel('Block size')
    # # plt.legend(['non-reordered','GPS','partitioned'])
    # plt.tight_layout()
    # plt.show()

def speedup (fname, data_size, orig_bw):
    bw_hier, bw_glob = get_bw(fname,data_size)
    if MINI_AERO:
        bw_hier[bw_hier > 7e11] = 0

    hier_bs = np.argmax(bw_hier[:,:,:,0],axis=0)
    hier_bs_ = [bss_[a] for a in hier_bs.T.flatten()]
    if not MINI_AERO:
        glob_bs = np.argmax(bw_glob[:,:,:,0],axis=0)
        glob_bs_ =[bss_[a] for a in glob_bs.T.flatten()]


    data = [
            bw_hier[hier_bs[0,0], 0, 0, 0], bw_hier[hier_bs[1,0], 1, 0, 0],
            bw_hier[hier_bs[0,1], 0, 1, 0], bw_hier[hier_bs[1,1], 1, 1, 0],
            ]
    if not MINI_AERO:
        data += [
            bw_hier[hier_bs[0,2], 0, 2, 0], bw_hier[hier_bs[1,2], 1, 2, 0],
            bw_glob[glob_bs[0,0], 0, 0, 0], bw_glob[glob_bs[1,0], 1, 0, 0],
            bw_glob[glob_bs[0,1], 0, 1, 0], bw_glob[glob_bs[1,1], 1, 1, 0],
            bw_glob[glob_bs[0,2], 0, 2, 0], bw_glob[glob_bs[1,2], 1, 2, 0],
            ]

    data = np.array(data)
    data /= orig_bw
    xticks = [
            'Hier\nGPS\nAOS', 'Hier\nGPS\nSOA',
            'Hier\npart.\nAOS','Hier\npart.\nSOA',
            ]
    if not MINI_AERO:
        xticks = [
            'Hier\nNR\nAOS', 'Hier\nNR\nSOA', 'Hier\nGPS\nAOS', 'Hier\nGPS\nSOA',
            'Hier\npart.\nAOS','Hier\npart.\nSOA',
            ] + xticks + [
            'Glob\nNR\nAOS','Glob\nNR\nSOA', 'Glob\nGPS\nAOS','Glob\nGPS\nSOA',
            'Glob\npart.\nAOS','Glob\npart.\nSOA'
            ]
    if MINI_AERO:
        xticks = [a + '\n(' + str(b) + ')' for a,b in zip(xticks,hier_bs_)]
        xticks += ['orig.\n(array)\nSOA\n(256)']
    else:
        xticks = [a + '\n(' + str(b) + ')' for a,b in zip(xticks,hier_bs_ + glob_bs_)]

    if MINI_AERO:
        plt.figure(figsize = (4.2, 2.5))
    else:
        plt.figure(figsize = (5, 2.5))
    plt.subplot(111)
    plt.grid(True)
    num_cols = 2 if MINI_AERO else 6
    plt.bar(2 * np.arange(num_cols),data[::2], 0.8,
            color = (0.65, 0.65, 0.65))
    plt.bar(2 * np.arange(num_cols) + 1.0,data[1::2], 0.8,
            color = (0.45, 0.45, 0.45))
    if MINI_AERO:
        plt.bar(4, orig_bw_mini_aero_array / orig_bw, 0.8,
                color = (0.25, 0.25, 0.25))
    plt.plot([-0.4,4.4 if MINI_AERO else 11.4],[1,1],'k')
    plt.xticks(np.arange(11 if MINI_AERO else 12), xticks)
    if MINI_AERO:
        plt.legend(['original','AOS','SOA', 'original (helper array)'])
    else:
        plt.legend(['original','AOS','SOA'])
    plt.tight_layout()
    plt.show()

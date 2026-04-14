import time, numpy as np, arepo, os, gc, sys, math
import astropy.units as u, astropy.constants as constants
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

def gastemp(s):
    xe = s['ne']
    U = np.array(s['InternalEnergy'])*u.km**2/u.s**2

    if 'gmet' in s.data:
        Xh = s['gmet'][:,0]
    else:
        Xh = 0.76
    gamma = 5./3.

    # mu = (1 + Xh /(1-Xh)) / (1 + Xh/(4*(1-Xh)) + xe)*constants.m_p
    mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p
    temp = (gamma - 1)*U/constants.k_B*mu

    return temp.to('K').value

def find_neighbors(tri):    
    l = tri.vertex_neighbor_vertices
    neib = []
    for i in range(len(l[0])-1):
        neib.append(list(l[1][l[0][i]:l[0][i+1]]))
    return neib

def get_allclouds(coldidxs,neighbors):
    coldidxs = np.array(coldidxs, dtype=int)
    idx_map = {orig:i for i, orig in enumerate(coldidxs)}

    rows, cols = [], []
    for i in coldidxs:
        ui = idx_map[i]
        for j in neighbors[i]:
            vj = idx_map.get(j)
            if vj is not None:
                rows.append(ui)
                cols.append(vj)

    # build symmetric sparse graph
    data = np.ones(len(rows), dtype=bool)
    G = coo_matrix((data, (rows, cols)), shape=(len(coldidxs), len(coldidxs)))
    G = G + G.T  # ensure symmetry

    n_components, labels = connected_components(csgraph=G, directed=False)
    cloudidxs = -np.ones(len(neighbors), dtype=int)
    cloudidxs[coldidxs] = labels

    return cloudidxs

def get_snum(filename):
    # return int((filename.split('_')[1]).split('.')[0])
    return int(filename.split('_')[1][:3])

##########
## main ##
##########

## Arguments ##
## 1: path to snapshots
## 2: path to allclouds files
## 3: overwrite boolean (1 = overwrite)
## 4: single snapshot number (optional, -1 to do all)

if __name__ == '__main__':

    folder = sys.argv[1]
    filelist = os.listdir(folder)
    filelist = list(np.array(filelist)[[f.startswith('snap_') & f.endswith('.hdf5') for f in filelist]])
    filelist.sort(key=lambda x:get_snum(x))
    if (len(filelist) == 0):
        filelist = os.listdir(folder)
        filelist = list(np.array(filelist)[[f.startswith('snapdir_') for f in filelist]])
        snums = np.array([get_snum(x) for x in filelist])
        filelist = [filelist[i]+'/snap_{:03}.0.hdf5'.format(snums[i]) for i in range(len(snums))]
        filelist = np.array(filelist)[np.argsort(snums)]
    print("Found {} files in {}".format(len(filelist),folder))

    ddens = 2
    outdir = sys.argv[2]+"/denscut_{}".format(ddens)
    print("Outdir: {}".format(outdir))
    if (not os.path.exists(outdir)):
        os.makedirs(outdir)
    overwrite = int(sys.argv[3])
    print("Overwrite: {}".format(bool(overwrite)))

    if (len(sys.argv) > 4):
        onlysnum = int(sys.argv[4])
        if (onlysnum >= 0):
            print("Only finding clouds for snapshot number: {}".format(onlysnum))
    else:
        onlysnum = -1
    print("")

    for fname in filelist:
        snum = get_snum(fname)
        if (onlysnum > 0):
            if (snum != onlysnum):
                continue
        if (not overwrite):
            if (os.path.exists(outdir+"/allclouds_{}_IDs_dens1.npy".format(snum))):
                print("Skipping {}...".format(fname))
                continue
        print("Starting {}...".format(fname),flush=True)
        try:
            s5 = arepo.Snapshot(folder+'/'+fname,parttype=[5],combineFiles=True)
            snap_center = s5.part5.pos[0]
            s5.close()
        except:
            print("Failed {} (no BHs)!".format(fname))
            continue
        s = arepo.Snapshot(folder+'/'+fname,parttype=[0],combineFiles=True)

        subtemp = gastemp(s)
        radii = np.linalg.norm(s.pos-snap_center,axis=1)
        mask = radii < 200

        stime = time.time()
        delmesh = Delaunay(s.pos[mask],qhull_options="Qbb Qc Qz Q12 Q3 Q5 Q8")
        print("Delaunay: {:.2f} s".format(time.time()-stime),flush=True)
        stime = time.time()
        neighbors = find_neighbors(delmesh)
        print("Neighbors: {:.2f} s".format(time.time()-stime),flush=True)

        mindens = min(s.part0.rho.value)*10
        maxdens = max(s.part0.rho.value)
        nlevels = int(np.floor(math.log(maxdens/mindens,ddens)))
        for i in range(1,nlevels+1):
            denscut = mindens*ddens**i
            densmask = s.part0.rho.value[mask] > denscut
            densidxs = np.nonzero(densmask)[0]
            print("Level {}".format(i))

            stime = time.time()
            cloudidxs = get_allclouds(densidxs,neighbors)
            print("\tGet cloudidxs: {:.2f} s".format(time.time()-stime),flush=True)

            stime = time.time()

            # 1) pick off only the positions you actually care about
            ids    = s.part0.id[mask]               # shape = M 
            labels = cloudidxs                # shape = M (same alignment as ids)
            valid  = labels >= 0
            ids    = ids[valid]
            labels = labels[valid]

            # 2) sort everything by label
            order = np.argsort(labels)
            labels = labels[order]
            ids    = ids[order]

            # 3) split into per-cloud arrays
            #   SciPy gave you labels in 0..(K-1), so we can do a bincount
            counts    = np.bincount(labels)          # length = K
            split_idx = np.cumsum(counts)[:-1]       # split points between each group
            cloudids  = np.split(ids, split_idx)     # list of length K

            final = np.empty(len(cloudids),dtype=object)
            final[:] = cloudids
            np.save(outdir+"/allclouds_{}_IDs_dens{}.npy".format(snum,i),final)
            print("\tGet IDs and save: {:.2f} s".format(time.time()-stime),flush=True)

        s.close()
        del cloudidxs,cloudids,final,s,s5
        gc.collect()
        print("{} done.\n".format(fname),flush=True)

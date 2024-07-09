import time, numpy as np, arepo, os, gc, sys
import astropy.units as u, astropy.constants as constants
from matplotlib.colors import LogNorm
from scipy.spatial import Delaunay

def gastemp(s):
    xe = s['ne']
    U = np.array(s['InternalEnergy'])*u.km**2/u.s**2

    Xh = 0.76
    gamma = 5./3.

    mu = (1 + Xh /(1-Xh)) / (1 + Xh/(4*(1-Xh)) + xe)*constants.m_p
    # mu = 4./(1 + 3*Xh + 4*Xh*xe)*constants.m_p.cgs.value
    temp = (gamma - 1)*U/constants.k_B*mu

    return temp.to('K').value

def find_neighbors(tri):    
    l = tri.vertex_neighbor_vertices
    neib = []
    for i in range(len(l[0])-1):
        neib.append(list(l[1][l[0][i]:l[0][i+1]]))
    return neib

def get_allclouds(allmask,neighbors,verbose=False):
    allis = np.nonzero(allmask)[0]
    allclouds = []
    for j in allis:
        # print("\r{}/{}...".format(j+1,len(allis)),end='',flush=True)
        if verbose:
            print('\nStarting',j)
            print(allclouds)
        if len(allclouds)>0 and j in np.hstack(allclouds):
            continue
        cloud = get_cold_neighbors(j,allmask,neighbors,None,[],verbose)
        allclouds.append(cloud)
    print('done')
    return allclouds

def get_cold_neighbors(istart,allmask,neighbors,tosearch=None,cloud=[],verbose=False):
    if (tosearch is None):
        tosearch = [istart]
        if (allmask[istart]):
            cloud.append(istart)
        else:
            raise Exception('istart ({}) not in mask'.format(istart))
    elif len(tosearch) == 0:
        return np.sort(cloud)
    tosearch_next = []
    if verbose:
        print('tosearch: ',tosearch)
    for i in tosearch:
        if (i>=len(neighbors)):
            print(istart,i,len(neighbors))
        neib = neighbors[i]
        if verbose:
            print(i,'neighbors',neib)
        for n in neib:
            if (allmask[n]):
                if not n in cloud:
                    cloud.append(n)
                    tosearch_next.append(n)
        if verbose:
            # print('cloud',cloud)
            print('len(tosearch_next)',len(tosearch_next))
    return get_cold_neighbors(istart,allmask,neighbors,np.unique(tosearch_next),cloud,verbose)

##########
## main ##
##########

## Arguments ##
## 1: path to snapshots
## 2: path to allclouds files
## 3: overwrite boolean (1 = overwrite)

if __name__ == '__main__':

    folder = sys.argv[1]
    filelist = os.listdir(folder)
    filelist = list(np.array(filelist)[[f.endswith('_cutout.hdf5') for f in filelist]])
    filelist.sort(key=lambda x:int(x.split('_')[2]))

    outdir = sys.argv[2]
    overwrite = bool(sys.argv[3])

    for fname in filelist:
        if (not overwrite):
            if (os.path.exists(outdir+"/allclouds_subbox0_{}_IDs.npy".format(fname.split('_')[2]))):
                continue
        print("Starting {}...".format(fname),flush=True)
        s = arepo.Snapshot(folder+'/'+fname,parttype=[0])

        subtemp = gastemp(s)
        radii = np.linalg.norm(s.pos,axis=1)
        mask = radii < 200
        Tmask = subtemp[mask] < 10**4.5

        stime = time.time()
        delmesh = Delaunay(s.pos[mask])
        print("Delaunay: {:.2f} s".format(time.time()-stime),flush=True)
        stime = time.time()
        neighbors = find_neighbors(delmesh)
        print("Neighbors: {:.2f} s".format(time.time()-stime),flush=True)

        allclouds = get_allclouds(Tmask,neighbors)
        cloudids = [s.id[mask][c] for c in allclouds]

        final = np.empty(len(cloudids),dtype=object)
        final[:] = cloudids
        np.save(outdir+"/allclouds_subbox0_{}_IDs.npy".format(fname.split('_')[2]),final)

        del delmesh,neighbors,allclouds,final
        gc.collect()
        print("{} done.\n".format(fname),flush=True)

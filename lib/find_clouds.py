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
    # print('done')
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

def get_snum(filename):
    return int((filename.split('_')[1]).split('.')[0])

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
    filelist = list(np.array(filelist)[[f.startswith('snap_') for f in filelist]])
    filelist.sort(key=lambda x:get_snum(x))
    print("Found {} files in {}".format(len(filelist),folder))

    outdir = sys.argv[2]
    print("Outdir: {}".format(outdir))
    if (not os.path.exists(outdir)):
        os.makedirs(outdir)
    overwrite = int(sys.argv[3])
    print("Overwrite: {}".format(bool(overwrite)))
    print("")

    for fname in filelist:
        snum = get_snum(fname)
        if (not overwrite):
            if (os.path.exists(outdir+"/allclouds_{}_IDs.npy".format(snum))):
                print("Skipping {}...".format(fname))
                continue
        print("Starting {}...".format(fname),flush=True)
        s5 = arepo.Snapshot(folder+'/'+fname,parttype=[5])
        snap_center = s5.part5.pos[0]
        s5.close()
        s = arepo.Snapshot(folder+'/'+fname,parttype=[0])

        subtemp = gastemp(s)
        radii = np.linalg.norm(s.pos-snap_center,axis=1)
        mask = radii < 200
        Tmask = subtemp[mask] < 10**4.5

        stime = time.time()
        delmesh = Delaunay(s.pos[mask],qhull_options="Qbb Qc Qz Q12 Q3 Q5 Q8")
        print("Delaunay: {:.2f} s".format(time.time()-stime),flush=True)
        stime = time.time()
        neighbors = find_neighbors(delmesh)
        print("Neighbors: {:.2f} s".format(time.time()-stime),flush=True)

        stime = time.time()
        allclouds = get_allclouds(Tmask,neighbors,verbose=False)
        # allclouds = get_allclouds(s,Tmask,verbose=False)
        print("Get allclouds: {:.2f} s".format(time.time()-stime),flush=True)
        cloudids = [s.id[mask][c] for c in allclouds]

        final = np.empty(len(cloudids),dtype=object)
        final[:] = cloudids
        np.save(outdir+"/allclouds_{}_IDs.npy".format(snum),final)

        s.close()
        del allclouds,final,s,s5
        gc.collect()
        print("{} done.\n".format(fname),flush=True)

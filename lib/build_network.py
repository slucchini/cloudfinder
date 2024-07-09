import arepo, gc, time, pickle, sys
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
import networkx as nx
import astropy.units as u, astropy.constants as constants, astropy.coordinates as coord
from tqdm import tqdm
from scipy.optimize import linprog
from .CloudObj import *
import multiprocess as mp

# from https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def get_age_snap(s):
    return get_age(s.HubbleParam,s.Omega0,s.OmegaLambda,s.time)

# from https://www.astro.ucla.edu/~wright/CC.python
def get_age(h,WM,WV,z):
    Tyr = 977.8; a = 1.0; H0 = h*100; age = 0.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    n=1000         # number of points in integrals
    for i in range(n):
        a = az*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 1./adot

    zage = az*age/n
    zage_Gyr = (Tyr/H0)*zage
    return zage_Gyr

def check_overlap(i):
    edges = []
    co = cloud_objs_prev[i]
    p = s_prev.pos[snapmasks_prev[i]]; v = s_prev.vel[snapmasks_prev[i]]
    newp = p*u.kpc+v*u.km/u.s*dt
    future_cloud = newp.to(u.kpc).value
    
    for j,cn in enumerate(cloud_objs):
        overlap = np.array([in_hull(future_cloud,cp) for cp in s.pos[snapmasks[j]]])
        if (len(overlap[overlap==True])/len(overlap) > 0.1):
            edges.append([co,cn])
    return edges

##########
## main ##
##########

## Arguments ##
## 1: path to snapshots
## 2: path to allclouds files
## 3: snapnumber to start at
## 4: snapnumber to end at
## 5: number of cores

if __name__ == '__main__':

    ncores = int(sys.argv[5])

    G = nx.Graph()
    folder = sys.argv[1]
    outdir = sys.argv[2]
    snapvals = np.arange(int(sys.argv[3]),int(sys.argv[4])+1)
    stime = time.time()
    for si in snapvals:
        print("Starting {}...".format(si),flush=True,end='')
        stime1 = time.time()

        s = arepo.Snapshot(folder+'/snap_subbox0_{}_cutout.hdf5'.format(si),parttype=[0])
        allclouds = np.load(outdir+"/allclouds_subbox0_{}_IDs.npy".format(si),allow_pickle=True)
        allcloudlen = np.array([len(cl) for cl in allclouds])
        cloudids = allclouds[(allcloudlen >= 2) & (allcloudlen < max(allcloudlen))]
        
        cloud_objs = [CloudObj(c,si) for c in cloudids]
        argsort = np.argsort(s.id)
        sids = s.id[argsort]
        snapmasks = [co.get_snapmask(sids,argsort) for co in cloud_objs]
        G.add_nodes_from(cloud_objs,snapnum=si)
        
        if si > snapvals[0]:
            dt = (get_age_snap(s) - get_age_snap(s_prev))*u.Gyr
            with mp.Pool(ncores) as pool:
                for edges in list(pool.imap(check_overlap,range(len(cloud_objs_prev)))):
                    G.add_edges_from(edges)

        s_prev = s
        cloud_objs_prev = cloud_objs
        snapmasks_prev = snapmasks
        gc.collect()
        print('done ({:.2f} s)'.format(time.time()-stime1),flush=True)

    print("Loop time: {:.0f} sec".format(time.time()-stime))
    print("Saving graph...",end='',flush=True)
    with open("graph_{}-{}_min2.pkl".format(snapvals[0],snapvals[-1]),'wb') as f:
        pickle.dump(G,f)
    print('done',flush=True)

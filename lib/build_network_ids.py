import arepo, gc, time, pickle, sys, os
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from CloudObj import *


def compute_all_snapmasks(cloud_objs, sids, argsort):
    """
    Compute snapmasks for ALL clouds at once using vectorized numpy ops.
    Replaces the per-cloud get_snapmask which did O(N) 'in' checks on 50M arrays.
    """
    if len(cloud_objs) == 0:
        return []

    # Concatenate all cloud member IDs with cloud labels
    all_members = np.concatenate([co.members for co in cloud_objs])
    cloud_labels = np.concatenate([
        np.full(len(co.members), i, dtype=np.int32)
        for i, co in enumerate(cloud_objs)
    ])

    # Use searchsorted on the sorted sids array: O(M log N) instead of O(M*N)
    idx = np.searchsorted(sids, all_members)
    idx_clipped = np.minimum(idx, len(sids) - 1)
    found = sids[idx_clipped] == all_members

    # Compute positions for found particles
    found_idx = idx_clipped[found]
    found_labels = cloud_labels[found]
    pos_in_original = argsort[found_idx]

    # Split results by cloud
    snapmasks = [None] * len(cloud_objs)
    # Group by cloud label
    order = np.argsort(found_labels)
    sorted_labels = found_labels[order]
    sorted_pos = pos_in_original[order]

    # Find boundaries between clouds
    changes = np.concatenate([[0], np.where(np.diff(sorted_labels) != 0)[0] + 1, [len(sorted_labels)]])
    for k in range(len(changes) - 1):
        ci = sorted_labels[changes[k]]
        snapmasks[ci] = sorted_pos[changes[k]:changes[k+1]]

    # Fill any clouds that had no matches
    for i in range(len(cloud_objs)):
        if snapmasks[i] is None:
            snapmasks[i] = np.array([], dtype=np.intp)

    return snapmasks


def find_edges_by_id_overlap(cloud_objs_prev, cloud_objs):
    """
    Find edges between previous and current clouds based on shared particle IDs.
    Fully vectorized using numpy — no Python loops over particles.
    """
    if len(cloud_objs_prev) == 0 or len(cloud_objs) == 0:
        return []

    # Build flat arrays for current clouds
    all_curr_ids = np.concatenate([co.members for co in cloud_objs])
    all_curr_labels = np.concatenate([
        np.full(len(co.members), j, dtype=np.int32)
        for j, co in enumerate(cloud_objs)
    ])

    # Build flat arrays for previous clouds
    all_prev_ids = np.concatenate([co.members for co in cloud_objs_prev])
    all_prev_labels = np.concatenate([
        np.full(len(co.members), i, dtype=np.int32)
        for i, co in enumerate(cloud_objs_prev)
    ])

    # Sort current IDs for searchsorted lookup
    sort_idx = np.argsort(all_curr_ids)
    sorted_curr_ids = all_curr_ids[sort_idx]
    sorted_curr_labels = all_curr_labels[sort_idx]

    # Find prev particles in current set: O(M log N)
    insert_pos = np.searchsorted(sorted_curr_ids, all_prev_ids)
    insert_pos_clipped = np.minimum(insert_pos, len(sorted_curr_ids) - 1)
    found = sorted_curr_ids[insert_pos_clipped] == all_prev_ids

    if not np.any(found):
        return []

    rows = all_prev_labels[found]
    cols = sorted_curr_labels[insert_pos_clipped[found]]

    # Deduplicate edges using sparse matrix
    data = np.ones(len(rows), dtype=np.int32)
    overlap_matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(cloud_objs_prev), len(cloud_objs))
    ).tocsr().tocoo()

    edges = [(cloud_objs_prev[i], cloud_objs[j])
             for i, j in zip(overlap_matrix.row, overlap_matrix.col)]

    return edges


##########
## main ##
##########

## Arguments ##
## 1: path to snapshots
## 2: path to allclouds files
## 3: snapnumber to start at
## 4: snapnumber to end at

if __name__ == '__main__':

    G = nx.Graph()
    folder = sys.argv[1]
    outdir = sys.argv[2]
    snapvals = np.arange(int(sys.argv[3]), int(sys.argv[4])+1)
    stime = time.time()

    snapbase = folder+'/snap_{:03}.hdf5'
    if (not os.path.exists(snapbase.format(snapvals[0]))):
        snapbase = folder+'/snapdir_{0:03}/snap_{0:03}.0.hdf5'
        if (not os.path.exists(snapbase.format(snapvals[0]))):
            raise Exception("Cannot find files at {} or {}".format(
                folder+'/snap_{:03}.hdf5'.format(snapvals[0]),
                snapbase.format(snapvals[0])))

    cloud_objs_prev = None

    for si in snapvals:
        print("Starting {}...".format(si), flush=True, end='')
        stime1 = time.time()

        s = arepo.Snapshot(snapbase.format(si), parttype=[0], combineFiles=True,
                           fields=['id'])
        allclouds = np.load(outdir+"/allclouds_{}_IDs.npy".format(si), allow_pickle=True)
        allcloudlen = np.array([len(cl) for cl in allclouds])
        cloudids = allclouds[(allcloudlen >= 2) & (allcloudlen < max(allcloudlen))]

        cloud_objs = [CloudObj(c, si) for c in cloudids]
        argsort = np.argsort(s.id)
        sids = s.id[argsort]

        t0 = time.time()
        snapmasks = compute_all_snapmasks(cloud_objs, sids, argsort)
        print('snapmasks {:.1f}s...'.format(time.time()-t0), end='', flush=True)

        G.add_nodes_from(cloud_objs, snapnum=si)

        with open(outdir+"/s{}_snapmask_min2.pkl".format(si), 'wb') as f:
            pickle.dump(snapmasks, f)

        if cloud_objs_prev is not None:
            t0 = time.time()
            edges = find_edges_by_id_overlap(cloud_objs_prev, cloud_objs)
            print('edges {:.1f}s ({})...'.format(time.time()-t0, len(edges)), end='', flush=True)
            G.add_edges_from(edges)

        cloud_objs_prev = cloud_objs
        del s, sids, argsort
        gc.collect()
        print('done ({:.2f} s)'.format(time.time()-stime1), flush=True)

    print("Loop time: {:.0f} sec".format(time.time()-stime))
    print("Saving graph...", end='', flush=True)
    with open(outdir+"graph_{}-{}_min2.pkl".format(snapvals[0], snapvals[-1]), 'wb') as f:
        pickle.dump(G, f)
    print('done', flush=True)

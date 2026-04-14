import time, re, numpy as np, arepo, os, gc, sys
import astropy.units as u, astropy.constants as constants
from scipy.spatial import Delaunay

_SNAP_RE = re.compile(r'^snap_\d{3}\.hdf5$')


def get_snum(filename):
    return int(filename.split('_')[1][:3])


def gather_neighbors_csr(indptr, indices, cells):
    """Flat gather of Delaunay neighbors for an array of cell indices, using
    the raw CSR arrays from Delaunay.vertex_neighbor_vertices.

    Returns (neigh_flat, counts) where counts[i] = # neighbors of cells[i].
    """
    if len(cells) == 0:
        return np.empty(0, dtype=indices.dtype), np.empty(0, dtype=np.int64)
    starts = indptr[cells]
    ends = indptr[cells + 1]
    counts = (ends - starts).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=indices.dtype), counts
    # Build flat positions into `indices`:
    #   for each cell i, emit starts[i] .. starts[i]+counts[i]-1.
    cum = np.empty(len(counts), dtype=np.int64)
    cum[0] = 0
    np.cumsum(counts[:-1], out=cum[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(cum, counts)
    pos = np.repeat(starts.astype(np.int64), counts) + within
    return indices[pos], counts


def split_by_label(labels, values, ncl, out_dtype):
    """Split `values` into a list of per-label arrays of length ncl.
    Assumes labels in [0, ncl). Empty groups get an empty array."""
    out = [np.array([], dtype=out_dtype) for _ in range(ncl)]
    if len(labels) == 0:
        return out
    order = np.argsort(labels, kind='stable')
    ls = labels[order]
    vs = values[order]
    changes = np.concatenate(([0], np.where(np.diff(ls) != 0)[0] + 1, [len(ls)]))
    for seg in range(len(changes) - 1):
        lbl = int(ls[changes[seg]])
        out[lbl] = vs[changes[seg]:changes[seg + 1]]
    return out


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
    filelist = list(np.array(filelist)[[bool(_SNAP_RE.match(f)) for f in filelist]])
    filelist.sort(key=lambda x: get_snum(x))
    if (len(filelist) == 0):
        filelist = os.listdir(folder)
        filelist = list(np.array(filelist)[[f.startswith('snapdir_') for f in filelist]])
        snums = np.array([get_snum(x) for x in filelist])
        filelist = [filelist[i] + '/snap_{:03}.0.hdf5'.format(snums[i]) for i in range(len(snums))]
        filelist = np.array(filelist)[np.argsort(snums)]
    print("Found {} files in {}".format(len(filelist), folder))

    outdir = sys.argv[2]
    print("Outdir: {}".format(outdir))
    if (not os.path.exists(outdir)):
        os.makedirs(outdir)
    plotdir = outdir + "/plots/"
    if (not os.path.exists(plotdir)):
        os.makedirs(plotdir)
    overwrite = int(sys.argv[3])
    print("Overwrite: {}".format(bool(overwrite)),flush=True)

    if (len(sys.argv) > 4):
        onlysnum = int(sys.argv[4])
        if (onlysnum >= 0):
            print("Only finding clouds for snapshot number: {}".format(onlysnum),flush=True)
    else:
        onlysnum = -1
    print("")

    nlevels = 10

    for fname in filelist:
        snum = get_snum(fname)
        if (onlysnum > 0 and snum != onlysnum):
            continue
        if (not overwrite):
            if (os.path.exists(outdir + "/allclouds_{}_boundary.npy".format(snum))):
                print("Skipping {}...".format(fname), flush=True)
                continue
        print("Starting {}...".format(fname), flush=True)
        try:
            s5 = arepo.Snapshot(folder + '/' + fname, parttype=[5], combineFiles=True)
            snap_center = s5.part5.pos[0]
            s5.close()
        except:
            print("Failed {} (no BHs)!".format(fname))
            continue
        s = arepo.Snapshot(folder + '/' + fname, parttype=[0], combineFiles=True)

        radii = np.linalg.norm(s.pos - snap_center, axis=1)
        mask = radii < 200

        stime = time.time()
        delmesh = Delaunay(s.pos[mask], qhull_options="Qbb Qc Qz Q12 Q3 Q5 Q8")
        print("Delaunay: {:.2f} s".format(time.time() - stime), flush=True)

        # Raw CSR neighbor arrays — no list-of-lists conversion.
        indptr, indices = delmesh.vertex_neighbor_vertices

        cloudids = np.load(outdir + "/allclouds_{}_IDs.npy".format(snum), allow_pickle=True)
        simids = s.part0.id[mask]
        argsort = np.argsort(simids)
        sids = simids[argsort]
        N = len(simids)

        stime = time.time()

        current_clids = [np.asarray(c) for c in cloudids]
        ncl = len(current_clids)
        boundary_list = [[] for _ in range(ncl)]

        for k in range(nlevels):
            lengths = np.array([len(c) for c in current_clids], dtype=np.int64)
            if lengths.sum() == 0:
                for j in range(ncl):
                    boundary_list[j].append(np.array([], dtype=simids.dtype))
                continue

            # Batched ID -> mask-local cell index across ALL clouds at once.
            all_ids = np.concatenate(current_clids)
            labels = np.repeat(np.arange(ncl, dtype=np.int64), lengths)
            pos = np.searchsorted(sids, all_ids)
            pos_clipped = np.minimum(pos, N - 1)
            found = sids[pos_clipped] == all_ids
            cells_flat = argsort[pos_clipped[found]]
            cell_labels = labels[found]

            # Batched Delaunay-neighbor gather via CSR.
            neigh_flat, ncounts = gather_neighbors_csr(indptr, indices, cells_flat)
            neigh_labels = np.repeat(cell_labels, ncounts)

            # Dedupe (label, neighbor-cell) pairs across clouds in one shot.
            if len(neigh_flat) > 0:
                keys = neigh_labels * N + neigh_flat.astype(np.int64)
                ukeys = np.unique(keys)
                u_labels = ukeys // N
                u_cells = (ukeys % N).astype(indices.dtype)
            else:
                u_labels = np.empty(0, dtype=np.int64)
                u_cells = np.empty(0, dtype=indices.dtype)

            # Originals (unique label,cell pairs) for boundary exclusion.
            orig_keys = np.unique(cell_labels * N + cells_flat.astype(np.int64))

            # Boundary = neighbors \ originals (combined key space handles per-cloud).
            is_orig = np.isin(ukeys, orig_keys) if len(ukeys) else np.array([], dtype=bool)
            bnd_labels = u_labels[~is_orig]
            bnd_cells = u_cells[~is_orig]

            bnd_ids = simids[bnd_cells] if len(bnd_cells) else np.array([], dtype=simids.dtype)
            per_cloud_bnd = split_by_label(bnd_labels, bnd_ids, ncl, simids.dtype)
            for j in range(ncl):
                boundary_list[j].append(per_cloud_bnd[j])

            # Next-level clids = simids[neighbors] per cloud (same semantics as original).
            next_ids = simids[u_cells] if len(u_cells) else np.array([], dtype=simids.dtype)
            current_clids = split_by_label(u_labels, next_ids, ncl, simids.dtype)

        final = np.empty(len(boundary_list), dtype=object)
        final[:] = boundary_list
        np.save(outdir + "/allclouds_{}_boundary.npy".format(snum), final)
        print("Build boundary list and save: {:.2f} s".format(time.time() - stime), flush=True)

        s.close()
        del s, s5, simids, argsort, sids, boundary_list, final, current_clids
        gc.collect()
        print("{} done.\n".format(fname), flush=True)

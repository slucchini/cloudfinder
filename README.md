# CloudFinder

Catalogs cold clouds in the circumgalactic medium (CGM) of cosmological galaxy simulations run with [Arepo](https://arepo-code.org/), and tracks them across snapshots.

## Usage

See `run.sh` for an example:

```sh
SNAP_DIR=/path/to/simulation/output/
OUT_DIR=${SNAP_DIR}allclouds/

python3 lib/find_clouds.py $SNAP_DIR $OUT_DIR 0
python3 lib/build_network_ids.py $SNAP_DIR $OUT_DIR 200 300
```

## Core scripts

### `lib/find_clouds.py`

Identifies clouds in a single snapshot. Selects cold gas cells, builds a Delaunay triangulation to determine neighbors, and uses connected components on the neighbor graph to group cold cells into discrete clouds. Writes a catalog of clouds (and their member particle IDs) to `OUT_DIR`.

By default, identifies clouds within 200 comoving kpc/h centered around the black hole particle with the temperature cutoff at $10^{4.5}$ K. This cutoff was chosen for use with the [IllustrisTNG](https://tng-project.org) simulations which have a pressure floor corresponding to a temperature of $10^4$ K.

**Arguments:** `SNAP_DIR OUT_DIR OVERWRITE [SNAPNUM]` (OVERWRITE is 0/1; SNAPNUM optional, -1 for all)

### `lib/find_clouds_dens_levels.py`

Variant of `find_clouds.py` that identifies clouds by density rather than temperature. Builds the Delaunay triangulation once per snapshot, then sweeps through density thresholds in multiplicative steps (`ddens=2` by default) from `10 × min(rho)` up to `max(rho)`. At each level, runs connected components on the subset of cells above the threshold and writes `allclouds_{snum}_IDs_dens{i}.npy` into a `denscut_{ddens}` subdirectory of `OUT_DIR`. Useful for building nested cloud catalogs across a range of density cuts.

**Arguments:** `SNAP_DIR OUT_DIR OVERWRITE [SNAPNUM]` (OVERWRITE is 0/1; SNAPNUM optional, -1 for all)

### `lib/build_network_ids.py` (recommended)

Tracks clouds across snapshots by matching Arepo gas cell particle IDs between consecutive snapshot catalogs. Builds a `networkx` graph where nodes are clouds and edges connect clouds in adjacent snapshots that share any particle. Uses vectorized `searchsorted` on sorted ID arrays, giving roughly O(total_particles) per snapshot pair.

**Arguments:** `SNAP_DIR OUT_DIR FIRST_SNAP LAST_SNAP`

### `lib/build_network.py` (legacy, slow)

Original cloud tracker. Advects particle positions forward in time and tests cloud overlap by solving a linear program (`scipy.optimize.linprog`) for each particle against the convex hull of each candidate cloud in the next snapshot. Cost is roughly O(N_prev · N_curr · particles_per_cloud) per snapshot pair, which becomes prohibitive for large catalogs — prefer `build_network_ids.py`. Kept for reference and for comparisons that require geometric (rather than ID-based) matching.

**Arguments:** `SNAP_DIR OUT_DIR FIRST_SNAP LAST_SNAP`

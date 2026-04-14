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

Both stages are typically run on a cluster via the provided SLURM scripts (`cloudfinder.sbatch.sh`, `buildnetwork_ids.sbatch.sh`).

## Core scripts

### `lib/find_clouds.py`

Identifies clouds in a single snapshot. Selects cold gas cells, builds a Delaunay triangulation to determine neighbors, and uses connected components on the neighbor graph to group cold cells into discrete clouds. Writes a catalog of clouds (and their member particle IDs) to `OUT_DIR`.

**Arguments:** `SNAP_DIR OUT_DIR SNAPNUM`

### `lib/build_network_ids.py` (recommended)

Tracks clouds across snapshots by matching Arepo gas cell particle IDs between consecutive snapshot catalogs. Builds a `networkx` graph where nodes are clouds and edges connect clouds in adjacent snapshots that share any particle. Uses vectorized `searchsorted` on sorted ID arrays, giving roughly O(total_particles) per snapshot pair.

**Arguments:** `SNAP_DIR OUT_DIR FIRST_SNAP LAST_SNAP`

### `lib/build_network.py` (legacy, slow)

Original cloud tracker. Advects particle positions forward in time and tests cloud overlap by solving a linear program (`scipy.optimize.linprog`) for each particle against the convex hull of each candidate cloud in the next snapshot. Cost is roughly O(N_prev · N_curr · particles_per_cloud) per snapshot pair, which becomes prohibitive for large catalogs — prefer `build_network_ids.py`. Kept for reference and for comparisons that require geometric (rather than ID-based) matching.

**Arguments:** `SNAP_DIR OUT_DIR FIRST_SNAP LAST_SNAP`

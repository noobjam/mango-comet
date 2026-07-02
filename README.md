# Mango Comet

Portable viewer for retrospective agronomic story activity. The repository is
intentionally code-only: field geometry, Parquet runs, manifests, generated
bundles, and credentials are not published.

## VM quick start

```bash
git clone https://github.com/noobjam/mango-comet.git
cd mango-comet

python3 --version  # Python 3.11 or newer
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt

cp server/env.example server/.env
```

Edit `server/.env` and point `STORY_MAP_RUN_DIR` at the existing run directory
on the VM. A raw run works for current-frame exploration; historical footprints
require an optimized bundle.

```bash
python server/story_map_server.py
```

The safe default listens only on the VM loopback interface. In a second local
terminal, create an SSH tunnel and open `http://127.0.0.1:8877`:

```bash
ssh -N -L 8877:127.0.0.1:8877 VM_USER@VM_HOST
```

This prototype has no authentication. Bind it to `0.0.0.0` only after adding
explicit VM firewall/security-group restrictions or a protected reverse proxy.

## Build an optimized release

Build into a new versioned directory rather than overwriting the directory used
by a running process:

```bash
python server/build_story_map_bundle.py \
  --run-dir /srv/story-map-data/source-run \
  --out-dir /srv/story-map-data/releases/mango-comet-v1
```

Then set this in `server/.env` and restart the server:

```dotenv
STORY_MAP_RUN_DIR=/srv/story-map-data/releases/mango-comet-v1
```

The builder validates required schemas, geometry integrity, unique field IDs,
and geometry coverage before installing a release.

## Verify on the VM

```bash
python -m py_compile server/*.py
python -m unittest discover -s server -p 'test_*.py'
curl --fail http://127.0.0.1:8877/api/health
```

For interpretation, API endpoints, performance controls, optional GPU
precomputation, and the production vector-tile direction, see
[`server/README.md`](server/README.md).

## Data policy

Keep run artifacts outside the repository. The `.gitignore` blocks common
geospatial, Parquet, bundle, environment, and cache paths as a second line of
defense, but inspect `git status` before every push.

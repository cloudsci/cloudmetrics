# Performance benchmarks

[pytest-benchmark](https://pytest-benchmark.readthedocs.io/) timings of the
metrics in `cloudmetrics.mask` on deterministic synthetic cloud masks (see
`conftest.py`):

| mask             | size       | objects | cloud fraction | kind                                       |
|------------------|------------|---------|----------------|--------------------------------------------|
| `noise-small`    | 256x256 px | 199     | 0.3            | thresholded smoothed white noise           |
| `noise-medium`   | 512x512 px | 686     | 0.3            | thresholded smoothed white noise           |
| `cumulus-small`  | 256x256 px | 171     | 0.15           | clustered, heavy-tailed object sizes       |
| `cumulus-medium` | 512x512 px | 607     | 0.15           | clustered, heavy-tailed object sizes       |

They are not part of the regular test-suite (see `testpaths` in
`pyproject.toml`) but are run by the `performance benchmarks` workflow
(`.github/workflows/ci-performance.yml`).

## Running locally

```bash
pip install pytest-benchmark
OMP_NUM_THREADS=1 pytest benchmarks --benchmark-only \
    --benchmark-min-rounds=5 --benchmark-warmup=on --benchmark-disable-gc \
    --benchmark-json=head.json
```

Repeat with the version to compare against installed (e.g.
`pip install --no-deps <path-to-master-checkout>`) to get `base.json`, then

```bash
python benchmarks/compare.py --base base.json --head head.json \
    --max-slowdown 1.5 --summary summary.md
```

## CI check

On a pull-request the workflow benchmarks the base and the head commit on the
same runner, with the same dependencies and the benchmark files of the head,
and writes the comparison to the job's Summary tab. It fails if a benchmark's
median on the head exceeds **1.5x** the base median, or if a benchmark is
missing on the head. Runners are noisy, so treat a failure as a prompt to look
closer (and re-run) rather than as a precise measurement.

The benchmarks are skipped if nothing under `cloudmetrics/` or in
`pyproject.toml` changed.

Manual runs (`Actions` -> `performance benchmarks` -> `Run workflow`) always
benchmark and take the inputs `base_ref` (default `master`), `head_ref`
(default: the ref the workflow is run from) and `max_slowdown` (default
`1.5`), e.g. to compare a branch against `master` without opening a
pull-request.

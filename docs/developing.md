# Developing cloudmetrics

`cloudmetrics` is automatically checked with tests that reside in
`tests/`. Every function with a name starting with `test_` in every file
in this directory with filename starting with `test_` is considered a test
and will be run. These tests are run automatically on all pull-requests
against the git repository at https://github.com/cloudsci/cloudmetrics and
can be run locally with `pytest` from the root of the repository.

Everything needed to get a working developing environment set up is stored
inside `pyproject.toml`. The dependency versions used in CI are pinned in
`uv.lock`, so the recommended way to set up is with
[uv](https://docs.astral.sh/uv/), which creates a virtual environment in
`.venv` with `cloudmetrics` installed in editable mode together with the
`dev` extras (pytest, nbval, pre-commit, ipython):

```bash
uv sync --extra dev
uv run pytest
```

Alternatively you can install everything with pip into an environment of your
choice:

```bash
python -m pip install -e ".[dev]"
```

Performance benchmarks of the metrics reside in `benchmarks/` and are not run
with the regular tests, but by a separate github action that compares the
timings of a pull-request against its base branch. See
[`benchmarks/README.md`](../benchmarks/README.md) for how to run them locally.

Linting is done with [pre-commit](https://pre-commit.com/), run the following
command to have linting run automatically for each git commit:

```bash
uvx pre-commit install   # or `pre-commit install` if installed with pip
```

If you add or change dependencies in `pyproject.toml`, update the lockfile with
`uv lock` and commit the updated `uv.lock` (CI installs with
`uv sync --locked`, which fails if the lockfile is out of date). Dependabot
opens quarterly pull-requests (only for versions released at least 7 days ago) to keep `uv.lock`, the GitHub Actions used in
`.github/workflows` and the hook revisions in `.pre-commit-config.yaml` up to date.

If the computer you are running on has multiple CPUs it can be advantageous to
run the tests in parallel to speed up the testing process. To do this you will
first need to install `pytest-xdist` and run pytest with `-n` to indicate the
number of parallel workers:

```bash
python -m pip install pytest-xdist
python -m pytest -n <n_cpus>
```

You can also speed up testing by reducing the number of test being run (if
you're for example working on fixing just a single breaking test) by using the
`-k` flag which where you provide a regex pattern for the name of tests you
want to run, e.g.

```bash
python -m pytest -k orientation
```

Finally, it is useful to have a `ipdb`-debugger open up inline on failing
tests. This can be achieved by first installing `ipdb` and setting the
`PYTEST_ADDOPPS` environment variable:

```bash
export PYTEST_ADDOPTS='--pdb --pdbcls=IPython.terminal.debugger:Pdb'
```

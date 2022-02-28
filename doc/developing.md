# Developing cloudmetrics

`cloudmetrics` is automatically checked with tests that reside in
`tests/`. Every function with a name starting with `test_` in every file
in this directory with filename starting with `test_` is considered a test
and will be run. These tests are run automatically on all pull-requests
against the git repository at https://github.com/cloudsci/cloudmetrics and
can be run locally with `pytest` from the root of the repository.

Everything needed to get a working developing environment set up is stored
inside `setup.cfg` can be installed with pip by calling:

```bash
python -m pip install -e ".[dev]"
```

Linting is done with [pre-commit](https://pre-commit.com/), run the following
command to have linting run automatically for each git commit:

```bash
pre-commit install
```

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

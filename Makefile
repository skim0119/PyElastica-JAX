#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
PYTHON_VERSION :=

#* Installation
.PHONY: install
install:
ifdef PYTHON_VERSION
	uv sync --python $(PYTHON_VERSION)
else
	uv sync
endif

.PHONY: install-dev-deps
install-dev-deps:
ifdef PYTHON_VERSION
	uv sync --all-groups --all-extras --python $(PYTHON_VERSION)
else
	uv sync --all-groups --all-extras
endif


.PHONY: build
build:
	uv build

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

#* Formatters
.PHONY: ruff-format
ruff-format:
	uv run --no-sync ruff --version
	uv run --no-sync ruff format src/elastica_jax tests examples benchmark

.PHONY: ruff-format-check
ruff-format-check:
	uv run --no-sync ruff --version
	uv run --no-sync ruff format --check src/elastica_jax tests examples benchmark

.PHONY: ruff-check
ruff-check:
	uv run --no-sync ruff --version
	uv run --no-sync ruff check src/elastica_jax tests examples benchmark

.PHONY: format-codestyle
format-codestyle: ruff-format

.PHONY: mypy
mypy:
	uv run --no-sync mypy --config-file pyproject.toml src/elastica_jax  # Main
	uv run --no-sync mypy --config-file pyproject.toml --explicit-package-bases \
		examples/AxialStretchingCase \
		examples/ButterflyCase \
		examples/CatenaryCase \
		examples/ContinuumSnakeCase

.PHONY: test
test:
	uv run --no-sync pytest -c pyproject.toml tests

.PHONY: test_coverage
test_coverage:
	NUMBA_DISABLE_JIT=1 uv run --no-sync pytest --cov=src/elastica_jax -c pyproject.toml tests

.PHONY: test_coverage_xml
test_coverage_xml:
	NUMBA_DISABLE_JIT=1 uv run --no-sync pytest --cov=src/elastica_jax --cov-report=xml -c pyproject.toml tests

.PHONY: check-codestyle
check-codestyle: ruff-format-check ruff-check

.PHONY: formatting
formatting: format-codestyle

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/ dist/

.PHONY: doc-remove
doc-remove:
	rm -rf docs/_build docs/gen_modules/ docs/sg_execution_times.rst docs/_gallery/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove mypycache-remove build-remove doc-remove

all: format-codestyle cleanup test

ci: check-codestyle

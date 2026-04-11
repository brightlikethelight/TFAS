# Makefile for the s1s2 project.
#
# Standard targets:
#
#   make install       # editable install + dev tooling + pre-commit hooks
#   make test          # full pytest suite
#   make test-fast     # skip @pytest.mark.slow tests
#   make lint          # ruff check
#   make format        # black + ruff --fix
#   make smoke         # end-to-end smoke test on synthetic data
#   make benchmark     # generate the benchmark JSONL
#   make extract       # cache activations for all configured models
#   make probes        # run the probing workstream
#   make sae           # run the SAE workstream
#   make attention     # run the attention workstream
#   make geometry      # run the geometry workstream
#   make causal        # run the causal interventions workstream
#   make metacog       # run the metacognitive monitoring workstream
#   make all-analyses  # probes, sae, attention, geometry, causal, metacog
#   make figures       # regenerate publication figures
#   make clean         # remove caches and ephemeral artifacts

PYTHON ?= python
PIP ?= pip

.PHONY: install test test-fast lint format smoke clean \
	benchmark extract probes sae attention geometry causal metacog \
	all-analyses figures help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  install        editable install + dev extras + pre-commit hooks"
	@echo "  test           run full pytest suite"
	@echo "  test-fast      skip @pytest.mark.slow tests"
	@echo "  lint           ruff check src/ tests/ scripts/"
	@echo "  format         black + ruff --fix"
	@echo "  smoke          end-to-end smoke test on synthetic data"
	@echo "  benchmark      generate the benchmark JSONL from templates"
	@echo "  extract        cache activations for all configured models"
	@echo "  probes / sae / attention / geometry / causal / metacog"
	@echo "                 run an individual workstream"
	@echo "  all-analyses   run every analysis workstream"
	@echo "  figures        regenerate publication figures"
	@echo "  clean          remove caches and ephemeral artifacts"

install:
	$(PIP) install -e ".[dev]"
	pre-commit install || echo "pre-commit hook install skipped (pre-commit not installed?)"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/ --line-length=100
	ruff check --fix src/ tests/ scripts/

smoke:
	$(PYTHON) scripts/smoke_test.py

benchmark:
	$(PYTHON) -m s1s2.benchmark.cli generate

extract:
	$(PYTHON) scripts/extract_all.py

probes:
	$(PYTHON) scripts/run_probes.py

sae:
	$(PYTHON) scripts/run_sae.py

attention:
	$(PYTHON) scripts/run_attention.py

geometry:
	$(PYTHON) scripts/run_geometry.py

causal:
	$(PYTHON) scripts/run_causal.py

metacog:
	$(PYTHON) scripts/run_metacog.py

all-analyses: probes sae attention geometry causal metacog

figures:
	$(PYTHON) scripts/build_figures.py

clean:
	rm -rf data/activations/*.h5 || true
	rm -rf results/**/*.json || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

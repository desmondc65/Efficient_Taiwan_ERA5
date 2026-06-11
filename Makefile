# Convenience targets. `make help` lists them.
PY ?= .venv/bin/python
STEPS ?= 744

.PHONY: help setup check synthetic download zarr bench-codecs bench-chunking \
        bench-dataloader bench-ddp report all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## install core deps into the active env
	$(PY) -m pip install -r requirements.txt

check: ## run pre-flight checks
	$(PY) scripts/00_check_setup.py

synthetic: ## generate synthetic ERA5 (no CDS key needed)
	$(PY) scripts/02_make_synthetic.py --steps $(STEPS)

download: ## download real ERA5 from the CDS (needs ~/.cdsapirc)
	$(PY) scripts/01_download_era5.py

zarr: ## build the production Zarr from raw data
	$(PY) scripts/03_build_zarr.py

bench-codecs: ## benchmark compression codecs
	$(PY) scripts/04_bench_codecs.py

bench-chunking: ## benchmark chunking strategies (keeps stores)
	$(PY) scripts/05_bench_chunking.py --keep

bench-dataloader: ## benchmark PyTorch DataLoader worker scaling
	$(PY) scripts/06_bench_dataloader.py

bench-ddp: ## DDP strong-scaling sweep on this node
	./run_ddp.sh

report: ## summarize all metrics (+plots)
	$(PY) scripts/08_report.py --plots

all: ## full synthetic pipeline end-to-end
	./run_all.sh

clean: ## remove generated data + results
	rm -rf data results

.PHONY: run test check

run test check:
	@test -n "$(DATASET)" || { echo "Set DATASET=/path/to/ebay_boys_girls_shirts"; exit 2; }
	uv run --python 3.11 python scripts/run_notebook.py "$(DATASET)"

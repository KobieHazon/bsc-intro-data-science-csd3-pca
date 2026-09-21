# BSc Intro Data Science - CSD3 PCA

- Course: BSc Computer Science.

## Contents

Case-study coursework around image-vector loading, dimensionality reduction, and PCA answer selection.

## Files

Template or reference material:

- `assignment/CSD3.ipynb`

Runnable implementation:

- `solutions/CSD3.ipynb`: complete PCA workflow

My submitted answers:

- `results/csd3_answers.csv`

## Tech Stack

- Python notebooks.
- Main Python packages: matplotlib, numpy, pandas, requests, scikit-image, scikit-learn, notebook.
- Jupyter-compatible local review flow.

## Notes

- The exercise notebook and submitted answer CSV are included.
- The image dataset is supplied separately. The runnable notebook uses a local dataset directory.

## Run

Supply the extracted `ebay_boys_girls_shirts` course dataset:

```sh
uv run --python 3.11 python scripts/run_notebook.py /path/to/ebay_boys_girls_shirts
```

This runs the complete PCA workflow on 64 real images per class: loading, centering, ten-component PCA, projection, reconstruction checks, plots, and score exports. The exercise template and submitted answer CSV are unchanged. The runnable implementation completes the coding exercises; it is not the original submitted notebook.

Add `--full` to use 2,000 training images per class. Use `--output-dir PATH` to keep generated plots and CSVs; otherwise they are temporary. Submitted files are never overwritten. For interactive use, set `SHIRTS_DATASET` before opening the solution notebook.

"""Execute every solution code cell and check computed numerical results."""

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "dataset", type=Path, help="Extracted ebay_boys_girls_shirts directory"
)
parser.add_argument("--full", action="store_true", help="Use original sample sizes")
parser.add_argument("--samples-per-class", type=int, default=64)
parser.add_argument(
    "--output-dir", type=Path, help="New or empty directory for plots and CSVs"
)
args = parser.parse_args()
if args.samples_per_class < 16:
    parser.error("Use at least 16 training images per class")
dataset = args.dataset.expanduser().resolve()
for filename in (
    "boys_train.csv",
    "girls_train.csv",
    "boys_test.csv",
    "girls_test.csv",
):
    if not (dataset / filename).is_file():
        parser.error("Missing dataset file: " + filename)
os.environ["SHIRTS_DATASET"] = str(dataset)
root = Path(__file__).resolve().parents[1]
notebook = root / "solutions" / "CSD3.ipynb"
cells = json.loads(notebook.read_text())["cells"]


class SampleSize(ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "n_train":
                node.value = ast.Constant(args.samples_per_class)
            elif node.targets[0].id == "n_test":
                node.value = ast.Constant(min(32, args.samples_per_class))
        return node


def execute(output):
    import numpy as np

    namespace = {"__name__": "__main__"}
    completed = 0
    figures = 0
    original_cwd = Path.cwd()
    try:
        os.chdir(output)
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(
                line for line in cell["source"] if not line.lstrip().startswith("%")
            )
            if not source.strip():
                continue
            tree = ast.parse(source)
            if not args.full:
                tree = SampleSize().visit(tree)
            ast.fix_missing_locations(tree)
            print(f"Executing cell {index}", flush=True)
            exec(compile(tree, f"{notebook.name}:cell{index}", "exec"), namespace)
            completed += 1
            if "plt" in namespace:
                plt = namespace["plt"]
                for figure_number in plt.get_fignums():
                    plt.figure(figure_number).savefig(
                        f"cell-{index}-figure-{figure_number}.png"
                    )
                    figures += 1
                plt.close("all")
        x = namespace["x_train_centered"]
        pca = namespace["pca"]
        reduced = namespace["x_train_reduced"]
        assert x.shape[1] == 30000 and reduced.shape == (len(x), 10)
        assert np.isfinite(reduced).all()
        np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(
            pca.components_ @ pca.components_.T, np.eye(10), atol=1e-10
        )
        np.testing.assert_allclose(reduced, x @ pca.components_.T, atol=1e-8)
        rebuilt = pca.inverse_transform(reduced)
        assert np.mean((x - rebuilt) ** 2) <= np.mean(x**2) + 1e-8
        assert np.all(pca.explained_variance_ratio_ >= 0)
        assert pca.explained_variance_ratio_.sum() <= 1 + 1e-10
        high, low = namespace["plot_highest_lowest_on_PC"](0)
        assert len(high) == len(low) == 16
        assert min(reduced[high, 0]) >= max(reduced[low, 0])
        try:
            namespace["plot_highest_lowest_on_PC"](10)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-range component accepted")
        namespace["plt"].close("all")
        exported = namespace["pd"].read_csv("pca_scores.csv")
        np.testing.assert_allclose(exported.iloc[:, 1:].values, reduced)
        assert Path("pca_variance.csv").is_file()
        print(
            f"PASS: {completed} solution code cells; {figures} plots; real-image computation and numerical checks"
        )
    finally:
        os.chdir(original_cwd)


if args.output_dir:
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    execute(output)
else:
    with tempfile.TemporaryDirectory(prefix="course-notebook-") as directory:
        execute(Path(directory))

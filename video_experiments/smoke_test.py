"""One-cell smoke test: does the restructured/cleaned solver code reproduce
the recorded ballet test_ari for k=3, fold=0, sigma=0.0 across all 3 methods?
"""
import csv

import cluster_ballet as cb
from common import run_row

dancer_segments = cb.load_dancer_segments()
name_to_dir = {p.name: p for p in cb.sequence_dirs()}
test_mats = cb.fold_test_matrices(dancer_segments, name_to_dir, k=3, sigma=0.0, fold=0)
print(f"n test matrices: {len(test_mats)}", flush=True)

rows = list(csv.DictReader(open("hyperparameters_ballet.csv")))
for method in ("TKSS", "WTV-SSC", "OSC"):
    row = [r for r in rows if r["method"] == method and r["k"] == "3"
           and r["fold"] == "0" and r["sigma"] == "0.0"][0]
    score = run_row(row, test_mats)
    print(f"{method} reproduced: {score}  recorded: {row['test_ari']}", flush=True)
print("ALL DONE", flush=True)

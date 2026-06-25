"""
Sample 100,000 rows from the full 11M-row HIGGS dataset and save to a local CSV.
Uses reservoir sampling over the gzip stream so we never load all 11M rows into memory.
"""
import gzip
import random
import csv
import os

SEED = 42
N_SAMPLE = 100_000
SRC = os.path.join(os.path.dirname(__file__), "..", "HIGGS.csv.gz")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "higgs_sample_100k.csv")

COLUMNS = ["label"] + [f"f{i}" for i in range(1, 29)]


def reservoir_sample(path, k, seed):
    rng = random.Random(seed)
    reservoir = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            row = line.rstrip("\n")
            if i < k:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = row
            if (i + 1) % 1_000_000 == 0:
                print(f"  scanned {i + 1:,} rows...")
    return reservoir


def main():
    print(f"Reservoir sampling {N_SAMPLE:,} rows from {SRC} ...")
    rows = reservoir_sample(SRC, N_SAMPLE, SEED)
    print(f"Sampled {len(rows):,} rows. Writing to {OUT} ...")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row.split(","))
    print("Done.")


if __name__ == "__main__":
    main()

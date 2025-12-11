#!/usr/bin/env python

import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STRUCT_DIR = BASE_DIR / "structures"
DOCKING_DIR = BASE_DIR / "docking"
RECEPTOR_DIR = DOCKING_DIR / "receptors"
POCKET_CSV = BASE_DIR / "results" / "pockets_fpocket.csv"


def main():
    import pandas as pd

    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

    if not POCKET_CSV.exists():
        raise FileNotFoundError(f"{POCKET_CSV} not found")

    df = pd.read_csv(POCKET_CSV)
    protein_ids = sorted(set(df["protein_id"].astype(str)))

    print("=== Preparing receptor PDBQT files with Meeko (mk_prepare_receptor.py) ===\n")
    print(f"Found {len(protein_ids)} unique protein IDs in {POCKET_CSV.name}\n")

    for prot in protein_ids:
        pdb_in = STRUCT_DIR / f"{prot}.pdb"
        if not pdb_in.exists():
            print(f"[SKIP] {prot}: structure not found at {pdb_in}")
            continue

        pdbqt_out = RECEPTOR_DIR / f"{prot}.pdbqt"
        if pdbqt_out.exists():
            print(f"[SKIP] {prot}: {pdbqt_out.name} already exists")
            continue

        print(f"[RUN] {prot}: {pdb_in.name} → {pdbqt_out.name}")

        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(pdb_in),      # input PDB
            "--write_pdbqt", str(pdbqt_out) # output PDBQT
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE_DIR,  # just in case
        )

        if result.returncode != 0:
            print(f"[ERROR] Meeko receptor prep failed for {prot}")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        else:
            print(f"[OK] Wrote {pdbqt_out}\n")

    print("=== Done receptor preparation ===")


if __name__ == "__main__":
    main()
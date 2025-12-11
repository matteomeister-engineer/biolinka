#!/usr/bin/env python

import subprocess
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
LIGAND_DIR = BASE_DIR / "docking" / "ligands"


def main():
    if not LIGAND_DIR.exists():
        print(f"[ERROR] Ligand directory does not exist: {LIGAND_DIR}")
        sys.exit(1)

    sdfs = sorted(LIGAND_DIR.glob("*.sdf"))
    if not sdfs:
        print(f"[WARN] No .sdf files found in {LIGAND_DIR}")
        sys.exit(0)

    print(f"=== Converting {len(sdfs)} ligands with Meeko (mk_prepare_ligand.py) ===")

    for sdf_path in sdfs:
        out_pdbqt = sdf_path.with_suffix(".pdbqt")
        print(f"\n[RUN] {sdf_path.name} → {out_pdbqt.name}")

        cmd = [
            "mk_prepare_ligand.py",
            "-i",
            str(sdf_path),
            "-o",
            str(out_pdbqt),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print(
                "[FATAL] mk_prepare_ligand.py not found in PATH.\n"
                "Make sure Meeko is installed in this environment, e.g.:\n"
                "  pip install meeko\n"
            )
            sys.exit(1)

        if result.returncode != 0:
            print("[ERROR] Meeko failed for", sdf_path.name)
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
        else:
            print("[OK] Wrote", out_pdbqt)


if __name__ == "__main__":
    main()
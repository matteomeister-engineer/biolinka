#!/usr/bin/env python

import argparse
import subprocess
import sys
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DOCKING_DIR = BASE_DIR / "docking"
STRUCT_DIR = BASE_DIR / "structures"

# ---------------------------------------------------------------------
# Vina result parsing
# ---------------------------------------------------------------------

VINA_RESULT_RE = re.compile(r"REMARK VINA RESULT:\s*([-0-9\.]+)")

def parse_vina_best_affinity(pdbqt_path: Path) -> float | None:
    """
    Parse the best binding energy (kcal/mol) from a Vina PDBQT output file.
    Looks for lines like:
        REMARK VINA RESULT:    -6.7      0.0      0.0
    Returns a float (negative) or None if not found.
    """
    if not pdbqt_path.exists():
        return None

    best = None
    try:
        with pdbqt_path.open() as f:
            for line in f:
                if "REMARK VINA RESULT" not in line:
                    continue
                m = VINA_RESULT_RE.search(line)
                if not m:
                    continue
                val = float(m.group(1))
                if best is None or val < best:
                    best = val
    except Exception:
        return None

    return best

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run AutoDock Vina for all selected pockets of an analyte."
    )
    parser.add_argument(
        "--analyte",
        required=True,
        help="Analyte name used in prepare_docking_setup (e.g. glucose, cortisol).",
    )
    parser.add_argument(
        "--vina",
        default="vina",
        help="Vina executable (default: 'vina' on PATH).",
    )
    parser.add_argument(
        "--ligand_name",
        default=None,
        help="Ligand name for PDBQT file (default: analyte, spaces->underscores).",
    )
    args = parser.parse_args()

    analyte_raw = args.analyte.strip()
    analyte_safe = analyte_raw.replace(" ", "_")

    ligand_name = args.ligand_name or analyte_safe

    docking_subdir = DOCKING_DIR / analyte_safe
    targets_csv = docking_subdir / "docking_targets.csv"
    ligand_pdbqt = DOCKING_DIR / "ligands" / f"{ligand_name}.pdbqt"
    receptors_dir = DOCKING_DIR / "receptors"
    docked_dir = docking_subdir / "docked"

    print(f"=== Vina batch docking for analyte '{analyte_raw}' ===")
    print(f"[INFO] Docking folder: {docking_subdir}")

    if not targets_csv.exists():
        print(f"[ERROR] docking_targets.csv not found at {targets_csv}")
        sys.exit(1)

    if not ligand_pdbqt.exists():
        print(f"[ERROR] Ligand PDBQT not found at {ligand_pdbqt}")
        sys.exit(1)

    docked_dir.mkdir(parents=True, exist_ok=True)

    # Load pocket targets prepared by prepare_docking_setup.py
    df_targets = pd.read_csv(targets_csv)

    required_cols = {"protein_id", "pocket_id", "box_file"}
    missing = required_cols - set(df_targets.columns)
    if missing:
        print(f"[ERROR] docking_targets.csv is missing columns: {missing}")
        sys.exit(1)

    vina_rows = []

    # -----------------------------------------------------------------
    # Loop over pockets
    # -----------------------------------------------------------------
    for _, row in df_targets.iterrows():
        pid = str(row["protein_id"])
        pocket_id = str(row["pocket_id"])
        box_file = Path(row["box_file"])

        receptor_pdbqt = receptors_dir / f"{pid}.pdbqt"
        if not receptor_pdbqt.exists():
            print(f"[WARN] Receptor PDBQT missing for {pid}: {receptor_pdbqt}")
            continue

        if not box_file.exists():
            print(f"[WARN] Box/config file missing for {pid} pocket {pocket_id}: {box_file}")
            continue

        out_pdbqt = docked_dir / f"{pid}_pocket{pocket_id}_{analyte_safe}_out.pdbqt"

        print(
            f"[RUN] {pid} pocket {pocket_id}: "
            f"vina --receptor {receptor_pdbqt.name} "
            f"--ligand {ligand_pdbqt.name} "
            f"--config {box_file.name}"
        )

        cmd = [
            args.vina,
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--config", str(box_file),
            "--out", str(out_pdbqt),
        ]

        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[ERROR] Vina failed for {pid} pocket {pocket_id}")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            affinity = None
        else:
            affinity = parse_vina_best_affinity(out_pdbqt)
            if affinity is None:
                # Fallback: try to parse from stdout
                for line in result.stdout.splitlines():
                    if "REMARK VINA RESULT" in line:
                        m = VINA_RESULT_RE.search(line)
                        if m:
                            affinity = float(m.group(1))
                            break

            if affinity is not None:
                print(f"[OK] Best affinity for {pid} pocket {pocket_id}: {affinity:.2f} kcal/mol")
            else:
                print(f"[WARN] Could not parse affinity for {pid} pocket {pocket_id}")

        vina_rows.append(
            {
                "analyte": analyte_raw,
                "protein_id": pid,
                "pocket_id": pocket_id,
                "receptor_pdbqt": str(receptor_pdbqt),
                "ligand_pdbqt": str(ligand_pdbqt),
                "config_file": str(box_file),
                "pose_pdbqt": str(out_pdbqt),
                "vina_affinity": affinity,
            }
        )

    if not vina_rows:
        print("[WARN] No docking runs completed; no vina_scores.csv written.")
        sys.exit(0)

    df_vina = pd.DataFrame(vina_rows)

    # Sort: best (most negative) energy first; NaN at bottom
    df_vina["vina_affinity"] = pd.to_numeric(df_vina["vina_affinity"], errors="coerce")
    df_vina = df_vina.sort_values("vina_affinity", ascending=True, na_position="last")

    out_csv = docking_subdir / "vina_scores.csv"
    df_vina.to_csv(out_csv, index=False)
    print(f"[OK] Vina scores written to {out_csv}")


if __name__ == "__main__":
    main()
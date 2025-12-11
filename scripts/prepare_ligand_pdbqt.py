#!/usr/bin/env python

import argparse
from pathlib import Path

from meeko import MoleculePreparation
from rdkit import Chem

BASE_DIR = Path(__file__).resolve().parents[1]
DOCKING_DIR = BASE_DIR / "docking"
LIGAND_DIR = DOCKING_DIR / "ligands"


def prepare_ligand_pdbqt(analyte: str) -> int:
    """Convert docking/ligands/<analyte>.sdf -> .pdbqt using Meeko+RDKit."""
    analyte = analyte.strip()
    if not analyte:
        print("[ERROR] Empty analyte name.")
        return 1

    analyte_folder_name = analyte.replace(" ", "_")
    sdf_path = LIGAND_DIR / f"{analyte_folder_name}.sdf"
    out_path = LIGAND_DIR / f"{analyte_folder_name}.pdbqt"

    if not sdf_path.exists():
        print(f"[ERROR] SDF not found: {sdf_path}")
        print("Run prepare_docking_setup.py for this analyte first.")
        return 1

    print(f"[INFO] Reading ligand SDF: {sdf_path}")
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = suppl[0] if len(suppl) > 0 else None
    if mol is None:
        print("[ERROR] RDKit failed to read SDF or no molecules present.")
        return 1

    print("[INFO] Preparing ligand with Meeko …")
    prep = MoleculePreparation()
    mol_set = prep.prepare(mol)
    pdbqt_string = prep.write_pdbqt_string(mol_set)

    LIGAND_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pdbqt_string)

    print(f"[OK] Wrote ligand PDBQT: {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert docking/ligands/<analyte>.sdf to .pdbqt using Meeko."
    )
    parser.add_argument(
        "--analyte",
        required=True,
        help="Analyte name (e.g. glucose, cortisol).",
    )
    args = parser.parse_args()
    raise SystemExit(prepare_ligand_pdbqt(args.analyte))


if __name__ == "__main__":
    main()
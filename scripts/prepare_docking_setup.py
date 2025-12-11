#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
STRUCT_DIR = BASE_DIR / "structures"
DOCKING_DIR = BASE_DIR / "docking"

POCKET_CSV = RESULTS_DIR / "pockets_fpocket.csv"

# ---------------------------------------------------------------------
# 1) Utilities: biosensor score (same logic as in GUI)
# ---------------------------------------------------------------------

def compute_biosensor_score(row: pd.Series) -> float:
    drug = float(row.get("druggability_score", 0.0))
    vol = float(row.get("volume", np.nan))
    hyd = float(row.get("hydrophobicity_score", np.nan))
    pol = float(row.get("polarity_score", np.nan))
    alpha = float(row.get("alpha_sphere_density", np.nan))
    flex = float(row.get("flexibility", np.nan))

    vol_term = 0.0
    if not np.isnan(vol):
        if vol < 150:
            vol_term = 0.2
        elif vol < 400:
            vol_term = 0.8
        elif vol < 900:
            vol_term = 1.0
        else:
            vol_term = 0.7

    hyd_term = 0.0
    if not np.isnan(hyd):
        hyd_norm = np.clip(hyd / 60.0, 0.0, 1.0)
        hyd_term = 0.3 + 0.7 * hyd_norm

    pol_term = 0.0
    if not np.isnan(pol):
        pol_norm = np.clip(pol / 15.0, 0.0, 1.0)
        pol_term = 0.3 + 0.7 * pol_norm

    alpha_term = 0.0
    if not np.isnan(alpha):
        alpha_norm = np.clip(alpha / 10.0, 0.0, 1.0)
        alpha_term = 0.3 + 0.7 * alpha_norm

    flex_term = 0.0
    if not np.isnan(flex):
        if 0.7 <= flex <= 1.0:
            flex_term = 1.0
        elif 0.5 <= flex < 0.7 or 1.0 < flex <= 1.3:
            flex_term = 0.7
        else:
            flex_term = 0.4

    drug_term = np.clip(drug, 0.0, 1.0)

    w_drug = 0.25
    w_vol = 0.25
    w_hyd = 0.20
    w_pol = 0.15
    w_alpha = 0.10
    w_flex = 0.05

    score = (
        w_drug * drug_term +
        w_vol * vol_term +
        w_hyd * hyd_term +
        w_pol * pol_term +
        w_alpha * alpha_term +
        w_flex * flex_term
    )
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------
# 2) PubChem call: get CID + properties + CanonicalSMILES
# ---------------------------------------------------------------------

def fetch_ligand_pubchem(analyte_name: str) -> dict | None:
    """
    Return:
      {
        name, cid, smiles,
        mw, xlogp, tpsa, charge, hbd, hba, rotb
      }
    or None.
    """
    name = (analyte_name or "").strip()
    if not name:
        return None

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # name -> CID
    cid_url = f"{base}/compound/name/{quote(name)}/cids/JSON"
    try:
        r_cid = requests.get(cid_url, timeout=8)
    except Exception as e:
        print(f"[ERROR] Request error (CID) for '{name}': {e}")
        return None

    if r_cid.status_code != 200:
        print(f"[WARN] CID lookup HTTP {r_cid.status_code} for '{name}'")
        return None

    try:
        cid_data = r_cid.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            print(f"[WARN] No CIDs found for '{name}'")
            return None
        cid = cids[0]
    except Exception as e:
        print(f"[ERROR] Parsing CID JSON for '{name}': {e}")
        return None

    # CID -> properties including CanonicalSMILES
    props = (
        "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,"
        "Charge,TPSA,RotatableBondCount,CanonicalSMILES"
    )
    prop_url = f"{base}/compound/cid/{cid}/property/{props}/JSON"

    try:
        r = requests.get(prop_url, timeout=8)
    except Exception as e:
        print(f"[ERROR] Request error (properties) for CID {cid}: {e}")
        return None

    if r.status_code != 200:
        print(f"[WARN] Property lookup HTTP {r.status_code} for CID {cid}")
        return None

    try:
        data = r.json()
        p = data["PropertyTable"]["Properties"][0]
    except Exception as e:
        print(f"[ERROR] Parsing property JSON for CID {cid}: {e}")
        return None

    def _float(key):
        v = p.get(key, None)
        if v is None:
            return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    def _int(key):
        v = p.get(key, None)
        if v is None:
            return 0
        try:
            return int(v)
        except Exception:
            return 0

    mw = _float("MolecularWeight")
    xlogp = _float("XLogP")
    tpsa = _float("TPSA")
    hbd = _int("HBondDonorCount")
    hba = _int("HBondAcceptorCount")
    charge = _int("Charge")
    rotb = _int("RotatableBondCount")
    smiles = p.get("CanonicalSMILES", None)

    info = {
        "name": name,
        "cid": cid,
        "smiles": smiles,
        "mw": mw,
        "xlogp": xlogp,
        "tpsa": tpsa,
        "charge": charge,
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
    }
    print(f"[INFO] PubChem: CID={cid}, MW={mw:.2f} (if not NaN), SMILES={smiles}")
    return info


# ---------------------------------------------------------------------
# 3) RDKit: build 3D conformer from SMILES
# ---------------------------------------------------------------------

def build_ligand_3d(smiles: str, out_sdf: Path) -> bool:
    """
    Generate a 3D conformer and write to SDF.
    Returns True on success.
    """
    if not smiles:
        print("[ERROR] No SMILES provided for ligand.")
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("[ERROR] RDKit failed to parse SMILES.")
        return False

    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if success != 0:
        print("[WARN] EmbedMolecule failed, trying again with random seed...")
        success = AllChem.EmbedMolecule(mol, AllChem.ETKDG(randomSeed=42))
        if success != 0:
            print("[ERROR] Failed to embed molecule in 3D.")
            return False

    AllChem.UFFOptimizeMolecule(mol)

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(out_sdf))
    w.write(mol)
    w.close()

    print(f"[OK] 3D ligand written to {out_sdf}")
    return True


# ---------------------------------------------------------------------
# 4) Ligand–pocket compatibility & QuickShape
# ---------------------------------------------------------------------

def compute_ligand_pocket_compat(
    row: pd.Series,
    mw: float,
    tpsa: float,
    xlogp: float,
    charge: float
) -> float:
    """
    Heuristic ligand–pocket compatibility score in [0, 1].
    """
    # ligand size class
    if not np.isnan(mw):
        if mw < 150:
            lig_size = "very_small"
        elif mw < 300:
            lig_size = "small"
        elif mw < 500:
            lig_size = "medium"
        else:
            lig_size = "large"
    else:
        lig_size = "unknown"

    # ligand polarity class (from logP + TPSA)
    if not np.isnan(tpsa) and tpsa > 90:
        lig_pol = "polar"
    elif not np.isnan(xlogp) and xlogp > 1.5 and (np.isnan(tpsa) or tpsa < 60):
        lig_pol = "hydrophobic"
    else:
        lig_pol = "mixed"

    # ligand charge class
    if charge > 0:
        lig_charge_class = "positive"
    elif charge < 0:
        lig_charge_class = "negative"
    else:
        lig_charge_class = "neutral"

    # polarity strength
    if not np.isnan(tpsa):
        if tpsa < 60:
            lig_pol_strength = "low"
        elif tpsa < 90:
            lig_pol_strength = "medium"
        else:
            lig_pol_strength = "high"
    else:
        lig_pol_strength = "unknown"

    # pocket
    vol = row.get("volume", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    pol_score = row.get("polarity_score", np.nan)
    chg_score = row.get("charge_score", 0.0)
    drug = row.get("druggability_score", 0.0)

    # pocket size
    if np.isnan(vol):
        pocket_size = "unknown"
    elif vol < 250:
        pocket_size = "very_small"
    elif vol < 500:
        pocket_size = "small"
    elif vol < 900:
        pocket_size = "medium"
    else:
        pocket_size = "large"

    # pocket environment
    if np.isnan(hyd):
        pocket_env = "mixed"
    elif hyd < 15:
        pocket_env = "polar"
    elif hyd < 40:
        pocket_env = "mixed"
    else:
        pocket_env = "hydrophobic"

    # pocket polarity strength
    if np.isnan(pol_score):
        pocket_pol_strength = "unknown"
    elif pol_score < 5:
        pocket_pol_strength = "low"
    elif pol_score < 10:
        pocket_pol_strength = "medium"
    else:
        pocket_pol_strength = "high"

    # pocket charge
    if chg_score > 1:
        pocket_charge_class = "positive"
    elif chg_score < -1:
        pocket_charge_class = "negative"
    else:
        pocket_charge_class = "neutral"

    # size match
    size_matrix = {
        ("very_small", "very_small"): 1.0,
        ("very_small", "small"): 0.9,
        ("small", "small"): 1.0,
        ("small", "very_small"): 0.8,
        ("small", "medium"): 0.8,
        ("medium", "medium"): 1.0,
        ("medium", "small"): 0.7,
        ("medium", "large"): 0.8,
        ("large", "large"): 1.0,
        ("large", "medium"): 0.8,
    }
    size_score = size_matrix.get((lig_size, pocket_size), 0.6)

    # hydrophobic match
    hydro_matrix = {
        ("hydrophobic", "hydrophobic"): 1.0,
        ("hydrophobic", "mixed"): 0.85,
        ("hydrophobic", "polar"): 0.4,
        ("mixed", "mixed"): 1.0,
        ("mixed", "hydrophobic"): 0.8,
        ("mixed", "polar"): 0.8,
        ("polar", "polar"): 1.0,
        ("polar", "mixed"): 0.85,
        ("polar", "hydrophobic"): 0.4,
    }
    hydro_score = hydro_matrix.get((lig_pol, pocket_env), 0.7)

    # polarity strength
    pol_strength_matrix = {
        ("low", "low"): 1.0,
        ("low", "medium"): 0.8,
        ("low", "high"): 0.5,
        ("medium", "medium"): 1.0,
        ("medium", "low"): 0.8,
        ("medium", "high"): 0.8,
        ("high", "high"): 1.0,
        ("high", "medium"): 0.9,
        ("high", "low"): 0.4,
    }
    pol_score_match = pol_strength_matrix.get((lig_pol_strength, pocket_pol_strength), 0.7)

    # charge complementarity
    if lig_charge_class == "positive" and pocket_charge_class == "negative":
        charge_score = 1.0
    elif lig_charge_class == "negative" and pocket_charge_class == "positive":
        charge_score = 1.0
    elif lig_charge_class == "neutral" and pocket_charge_class == "neutral":
        charge_score = 0.9
    elif lig_charge_class == "neutral" or pocket_charge_class == "neutral":
        charge_score = 0.75
    elif lig_charge_class == pocket_charge_class:
        charge_score = 0.4
    else:
        charge_score = 0.6

    drug_bonus = 0.5 + 0.5 * max(0.0, min(1.0, float(drug)))

    w_size = 0.25
    w_hydro = 0.30
    w_pol = 0.20
    w_charge = 0.20
    w_drug = 0.05

    score = (
        w_size * size_score +
        w_hydro * hydro_score +
        w_pol * pol_score_match +
        w_charge * charge_score +
        w_drug * drug_bonus
    )
    return float(max(0.0, min(1.0, score)))


def estimate_ligand_radius(mw: float) -> float:
    if np.isnan(mw) or mw <= 0:
        return np.nan
    if mw < 150:
        return 2.0
    elif mw < 300:
        return 3.0
    elif mw < 500:
        return 4.0
    elif mw < 800:
        return 5.0
    else:
        return 6.0


def compute_quickshape(row: pd.Series, mw: float) -> float:
    if np.isnan(mw) or mw <= 0:
        return np.nan

    pocket_vol = row.get("volume", np.nan)
    pocket_rad = row.get("com_max_dist", np.nan)

    if np.isnan(pocket_vol) or np.isnan(pocket_rad):
        return np.nan

    lig_rad = estimate_ligand_radius(mw)
    if np.isnan(lig_rad):
        return np.nan

    lig_vol = mw  # super simple proxy

    vol_ratio = pocket_vol / max(lig_vol, 1e-6)
    rad_ratio = pocket_rad / max(lig_rad, 1e-6)

    rad_score = np.exp(-((rad_ratio - 1.5) / 0.75) ** 2)
    vol_score = np.exp(-((vol_ratio - 4.0) / 2.0) ** 2)

    score = 0.6 * rad_score + 0.4 * vol_score
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------
# 5) Build docking boxes from pocket PDB
# ---------------------------------------------------------------------

def compute_box_from_pocket_pdb(pdb_path: Path, padding: float = 4.0):
    """
    Read pocketX_atm.pdb, compute bounding box:
    returns (center, size) where each is (x, y, z).
    """
    xs, ys, zs = [], [], []

    if not pdb_path.exists():
        print(f"[WARN] Pocket PDB not found: {pdb_path}")
        return None, None

    with pdb_path.open() as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xs.append(x)
                ys.append(y)
                zs.append(z)
            except ValueError:
                continue

    if not xs:
        print(f"[WARN] No atoms parsed in {pdb_path}")
        return None, None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    center_z = 0.5 * (min_z + max_z)

    size_x = (max_x - min_x) + 2 * padding
    size_y = (max_y - min_y) + 2 * padding
    size_z = (max_z - min_z) + 2 * padding

    center = (center_x, center_y, center_z)
    size = (size_x, size_y, size_z)

    return center, size


# ---------------------------------------------------------------------
# 6) Main procedure
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare docking setup (ligand 3D + pocket boxes) for a given analyte."
    )
    parser.add_argument("--analyte", required=True, help="Analyte name, e.g. glucose")
    parser.add_argument("--top_n", type=int, default=8, help="Number of top pockets to export")
    args = parser.parse_args()

    analyte = args.analyte
    top_n = args.top_n

    print(f"=== Docking setup for analyte: '{analyte}' ===")

    # 1) Load pockets
    if not POCKET_CSV.exists():
        print(f"[ERROR] pockets file not found at {POCKET_CSV}")
        sys.exit(1)

    df = pd.read_csv(POCKET_CSV)

    # recompute biosensor score
    df["biosensor_score"] = df.apply(compute_biosensor_score, axis=1)

    # 2) PubChem
    lig = fetch_ligand_pubchem(analyte)
    if lig is None:
        print("[ERROR] Could not fetch ligand from PubChem, aborting.")
        sys.exit(1)

    mw = lig["mw"]
    xlogp = lig["xlogp"]
    tpsa = lig["tpsa"]
    charge = lig["charge"]
    smiles = lig["smiles"]

    print(f"[INFO] Using MW={mw}, XlogP={xlogp}, TPSA={tpsa}, charge={charge}")
    print(f"[INFO] Canonical SMILES: {smiles}")

    # 3) Generate 3D ligand
    ligand_dir = DOCKING_DIR / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    safe_name = analyte.replace(" ", "_")
    ligand_sdf = ligand_dir / f"{safe_name}.sdf"

    ok_3d = build_ligand_3d(smiles, ligand_sdf)
    if not ok_3d:
        print("[WARN] 3D ligand generation failed. You can still use SMILES elsewhere.")
    else:
        print("[INFO] 3D ligand ready.")

    # 4) Compute compatibility & quickshape
    df["ligand_compat_score"] = df.apply(
        lambda row: compute_ligand_pocket_compat(
            row,
            mw=mw,
            tpsa=tpsa,
            xlogp=xlogp,
            charge=charge,
        ),
        axis=1,
    )

    df["quickshape_score"] = df.apply(
        lambda row: compute_quickshape(row, mw=mw),
        axis=1,
    )

    # 5) Select top N pockets
    df_sorted = df.sort_values(
        ["ligand_compat_score", "quickshape_score", "biosensor_score"],
        ascending=False,
    ).head(top_n)

    print(f"[INFO] Selecting top {len(df_sorted)} pockets for docking boxes.")

    # 6) For each pocket, compute box and write config file for Vina
    out_root = DOCKING_DIR / safe_name
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for _, row in df_sorted.iterrows():
        pid = str(row["protein_id"])
        pocket_id = str(row["pocket_id"])

        pocket_pdb = STRUCT_DIR / f"{pid}_out" / "pockets" / f"pocket{pocket_id}_atm.pdb"
        center, size = compute_box_from_pocket_pdb(pocket_pdb)

        if center is None or size is None:
            print(f"[WARN] Skipping pocket {pid} pocket {pocket_id} (no box).")
            continue

        cx, cy, cz = center
        sx, sy, sz = size

        out_box = out_root / f"{pid}_pocket{pocket_id}_box.txt"

        # --- Write a proper Vina CONFIG file ---
        with out_box.open("w") as f:
            f.write(f"center_x = {cx:.3f}\n")
            f.write(f"center_y = {cy:.3f}\n")
            f.write(f"center_z = {cz:.3f}\n\n")

            f.write(f"size_x = {sx:.3f}\n")
            f.write(f"size_y = {sy:.3f}\n")
            f.write(f"size_z = {sz:.3f}\n\n")

            # reasonable defaults – tweak if you like
            f.write("exhaustiveness = 16\n")
            f.write("num_modes = 9\n")
            f.write("energy_range = 3\n")

        print(f"[OK] Box written for {pid} pocket {pocket_id}: {out_box}")

        summary_rows.append({
            "analyte": analyte,
            "protein_id": pid,
            "pocket_id": pocket_id,
            "box_file": str(out_box),
            "ligand_compat_score": row["ligand_compat_score"],
            "quickshape_score": row["quickshape_score"],
            "biosensor_score": row["biosensor_score"],
        })

    # 7) Save summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_root / "docking_targets.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Summary written to {summary_csv}")
    else:
        print("[WARN] No boxes generated. Check that fpocket outputs exist and pockets_fpocket.csv is populated.")


if __name__ == "__main__":
    main()
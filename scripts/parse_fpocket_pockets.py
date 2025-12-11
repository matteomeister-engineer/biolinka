from pathlib import Path
import csv

STRUCT_DIR = Path("structures")
RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "pockets_fpocket.csv"

# We'll capture quite a few useful fields:
FIELDNAMES = [
    "protein_id",
    "pocket_id",
    "score",
    "druggability_score",
    "nb_alpha_spheres",
    "total_sasa",
    "polar_sasa",
    "apolar_sasa",
    "volume",
    "mean_local_hydrophobic_density",
    "mean_alpha_sphere_radius",
    "mean_alpha_sphere_solvent_access",
    "apolar_alpha_sphere_proportion",
    "hydrophobicity_score",
    "volume_score",
    "polarity_score",
    "charge_score",
    "proportion_polar_atoms",
    "alpha_sphere_density",
    "com_max_dist",
    "flexibility",
]

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

def find_info_file(protein_id: str, out_dir: Path) -> Path | None:
    """
    For your runs, the pattern is:
      <protein>_out/<protein>_info.txt
    """
    p0 = out_dir / f"{protein_id}_info.txt"
    if p0.exists():
        return p0

    # Fallbacks for other fpocket layouts:
    p1 = out_dir / "pockets_info.txt"
    if p1.exists():
        return p1

    p2 = out_dir / "pockets" / "pockets_info.txt"
    if p2.exists():
        return p2

    p3 = out_dir / "pockets" / f"{protein_id}_info.txt"
    if p3.exists():
        return p3

    return None

def parse_info_file(protein_id: str, info_path: Path):
    """
    Parse a fpocket <protein>_info.txt file with 'Pocket N :' blocks like:

    Pocket 1 :
        Score :   0.150
        Druggability Score :   0.004
        ...
    """
    pockets = []

    # Map from (normalized) text keys to our csv field names
    key_map = {
        "score": "score",
        "druggability score": "druggability_score",
        "number of alpha spheres": "nb_alpha_spheres",
        "total sasa": "total_sasa",
        "polar sasa": "polar_sasa",
        "apolar sasa": "apolar_sasa",
        "volume": "volume",
        "mean local hydrophobic density": "mean_local_hydrophobic_density",
        "mean alpha sphere radius": "mean_alpha_sphere_radius",
        "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_access",
        "apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
        "hydrophobicity score": "hydrophobicity_score",
        "volume score": "volume_score",
        "polarity score": "polarity_score",
        "charge score": "charge_score",
        "proportion of polar atoms": "proportion_polar_atoms",
        "alpha sphere density": "alpha_sphere_density",
        "cent. of mass - alpha sphere max dist": "com_max_dist",
        "flexibility": "flexibility",
    }

    current = None  # dict for the current pocket

    with info_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                # blank line -> end of pocket block
                if current is not None:
                    pockets.append(current)
                    current = None
                continue

            # Start of a new pocket block: "Pocket N :"
            if line.lower().startswith("pocket"):
                # If a previous pocket wasn't pushed yet, push it
                if current is not None:
                    pockets.append(current)

                # Extract the pocket number
                # Format: "Pocket 1 :"
                parts = line.split()
                pocket_id = None
                for p in parts:
                    if p.isdigit():
                        pocket_id = int(p)
                        break

                current = {field: None for field in FIELDNAMES}
                current["protein_id"] = protein_id
                current["pocket_id"] = pocket_id
                continue

            # Otherwise, this should be a "Key : Value" line
            if current is None:
                # We haven't hit "Pocket N :" yet; skip
                continue

            if ":" not in line:
                continue

            left, _, right = line.partition(":")
            key = left.strip().lower()
            value_str = right.strip()

            # Some lines like "Volume score:   4.200" → we want 4.200
            # Sometimes there might be units; we just take the first token
            if not value_str:
                continue
            val_token = value_str.split()[0]

            field_name = key_map.get(key)
            if field_name is None:
                # Unknown key -> ignore
                continue

            # Decide if this should be int or float:
            if field_name in ("nb_alpha_spheres",):
                try:
                    current[field_name] = int(val_token)
                except Exception:
                    current[field_name] = safe_float(val_token)
            else:
                current[field_name] = safe_float(val_token)

    # In case the file doesn't end with a blank line:
    if current is not None:
        pockets.append(current)

    return pockets

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_rows = []

    for pdb_path in STRUCT_DIR.glob("*.pdb"):
        protein_id = pdb_path.stem
        out_dir = STRUCT_DIR / f"{protein_id}_out"
        if not out_dir.exists():
            print(f"[WARN] No _out directory for {protein_id}, skipping.")
            continue

        info_path = find_info_file(protein_id, out_dir)
        if info_path is None:
            print(f"[WARN] No info file found for {protein_id} in {out_dir}, skipping.")
            continue

        print(f"[INFO] Parsing pockets for {protein_id} from {info_path}")
        pockets = parse_info_file(protein_id, info_path)
        print(f"  -> found {len(pockets)} pockets")
        all_rows.extend(pockets)

    if not all_rows:
        print("[WARN] No pocket data found. Did fpocket run successfully?")
        return

    print(f"[INFO] Writing {len(all_rows)} pocket rows to {OUT_CSV}")
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

if __name__ == "__main__":
    main()

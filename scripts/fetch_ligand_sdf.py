#!/usr/bin/env python

import sys
import requests
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parents[1]
DOCKING_DIR = BASE_DIR / "docking"
LIGAND_DIR = DOCKING_DIR / "ligands"


def fetch_pubchem_sdf_by_name(name: str) -> str | None:
    """
    Fetch a 3D (or 2D fallback) SDF for a small molecule from PubChem by name.
    Returns the SDF text, or None if it fails.
    """
    name = name.strip()
    if not name:
        return None

    try:
        # 1) name -> CID
        cid_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{quote(name)}/cids/JSON"
        )
        cid_resp = requests.get(cid_url, timeout=15)
        if cid_resp.status_code != 200:
            print(f"[ERROR] CID request failed ({cid_resp.status_code})")
            return None

        cid_data = cid_resp.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            print("[ERROR] No CIDs returned for this name.")
            return None
        cid = cids[0]
        print(f"[INFO] Using CID {cid}")

        # 2) CID -> 3D SDF (fall back to 2D if 3D not available)
        for record_type in ("3d", "2d"):
            sdf_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                f"compound/cid/{cid}/record/SDF/?record_type={record_type}"
            )
            print(f"[INFO] Trying {record_type} SDF from: {sdf_url}")
            sdf_resp = requests.get(sdf_url, timeout=30)
            if sdf_resp.status_code == 200 and sdf_resp.text.strip():
                print(f"[OK] Got {record_type} SDF.")
                return sdf_resp.text

        print("[ERROR] No valid SDF returned from PubChem.")
        return None

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_ligand_sdf.py <analyte_name>")
        sys.exit(1)

    name = sys.argv[1]
    analyte_folder_name = name.replace(" ", "_")

    print(f"=== Fetching SDF for analyte: '{name}' ===")

    sdf_text = fetch_pubchem_sdf_by_name(name)
    if sdf_text is None:
        print("[FAIL] Could not fetch SDF from PubChem.")
        sys.exit(1)

    LIGAND_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LIGAND_DIR / f"{analyte_folder_name}.sdf"
    out_path.write_text(sdf_text)

    print(f"[OK] Wrote SDF to: {out_path}")


if __name__ == "__main__":
    main()
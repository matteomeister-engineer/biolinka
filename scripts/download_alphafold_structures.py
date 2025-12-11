import requests
from pathlib import Path
import time

# Adjust this if your file is named differently
IDS_FILE = Path("data/raw/biosensor_panel_uniprot.txt")
OUT_DIR = Path("structures")
OUT_DIR.mkdir(exist_ok=True)

API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"


def download_af_model(uniprot_id: str) -> bool:
    """
    Use the AlphaFold API to get the PDB URL for a UniProt ID,
    then download the PDB file and save it locally.
    Returns True on success, False on failure.
    """
    uniprot_id = uniprot_id.strip()
    if not uniprot_id or uniprot_id.startswith("#"):
        return False

    out_path = OUT_DIR / f"{uniprot_id}.pdb"
    if out_path.exists():
        print(f"[SKIP] {uniprot_id} already exists at {out_path}")
        return True

    api_url = API_TEMPLATE.format(uniprot_id=uniprot_id)
    print(f"\n=== {uniprot_id} ===")
    print(f"→ Calling API: {api_url}")

    try:
        resp = requests.get(api_url, timeout=60)
    except Exception as e:
        print(f"   ⚠️ Request error for {uniprot_id}: {e}")
        return False

    if resp.status_code != 200:
        print(f"   ⚠️ API returned HTTP {resp.status_code} for {uniprot_id}")
        return False

    try:
        data = resp.json()
    except ValueError as e:
        print(f"   ⚠️ Failed to parse JSON for {uniprot_id}: {e}")
        return False

    if not data:
        print(f"   ⚠️ Empty result for {uniprot_id}")
        return False

    # API returns a list; we take the first prediction
    entry = data[0]
    pdb_url = entry.get("pdbUrl")
    entry_id = entry.get("entryId")

    if not pdb_url:
        print(f"   ⚠️ No 'pdbUrl' in API response for {uniprot_id}")
        return False

    print(f"   → Found entryId: {entry_id}")
    print(f"   → PDB URL: {pdb_url}")

    try:
        pdb_resp = requests.get(pdb_url, timeout=60)
    except Exception as e:
        print(f"   ⚠️ Error downloading PDB for {uniprot_id}: {e}")
        return False

    if pdb_resp.status_code != 200:
        print(f"   ⚠️ Failed to download PDB (HTTP {pdb_resp.status_code}) for {uniprot_id}")
        return False

    out_path.write_bytes(pdb_resp.content)
    print(f"   ✅ Saved PDB as {out_path}")
    return True


def main():
    if not IDS_FILE.exists():
        raise FileNotFoundError(
            f"{IDS_FILE} not found. Create it and put one UniProt ID per line."
        )

    ids = []
    with IDS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # support lines like "P0AEU7   # comment"
            token = line.split()[0]
            ids.append(token)

    print(f"[INFO] Found {len(ids)} IDs in {IDS_FILE}")

    for i, uniprot_id in enumerate(ids, start=1):
        print(f"\n--- ({i}/{len(ids)}) {uniprot_id} ---")
        download_af_model(uniprot_id)
        time.sleep(1.0)  # be polite to the server

    print("\n[DONE] Download run complete.")


if __name__ == "__main__":
    main()

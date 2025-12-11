import requests
from pathlib import Path

IDS_FILE = Path("data/raw/ids.txt")
OUT_DIR = Path("structures")

API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

def download_af_model(uniprot_id: str):
    """
    Use the AlphaFold API to get the PDB URL for a UniProt ID,
    then download the PDB file and save it locally.
    """
    OUT_DIR.mkdir(exist_ok=True)

    api_url = API_TEMPLATE.format(uniprot_id=uniprot_id)
    print(f"\n=== {uniprot_id} ===")
    print(f"→ Calling API: {api_url}")

    try:
        resp = requests.get(api_url)
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
        pdb_resp = requests.get(pdb_url)
    except Exception as e:
        print(f"   ⚠️ Error downloading PDB for {uniprot_id}: {e}")
        return False

    if pdb_resp.status_code != 200:
        print(f"   ⚠️ Failed to download PDB (HTTP {pdb_resp.status_code}) for {uniprot_id}")
        return False

    out_path = OUT_DIR / f"{uniprot_id}.pdb"
    out_path.write_bytes(pdb_resp.content)
    print(f"   ✅ Saved PDB as {out_path}")
    return True

def main():
    if not IDS_FILE.exists():
        raise FileNotFoundError(
            f"{IDS_FILE} not found. Create it and put one UniProt ID per line."
        )

    with IDS_FILE.open() as f:
        for line in f:
            uniprot_id = line.strip()
            if not uniprot_id:
                continue
            download_af_model(uniprot_id)

if __name__ == "__main__":
    main()

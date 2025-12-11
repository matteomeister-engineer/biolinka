import requests
from pathlib import Path

IDS_FILE = Path("data/raw/ids.txt")
OUT_FASTA = Path("data/raw/test_seqs.fasta")

BASE_URL = "https://rest.uniprot.org/uniprotkb/"

def main():
    if not IDS_FILE.exists():
        raise FileNotFoundError(f"{IDS_FILE} does not exist.")

    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)

    with IDS_FILE.open() as ids, OUT_FASTA.open("w") as out:
        for line in ids:
            acc = line.strip()
            if not acc:
                continue
            url = f"{BASE_URL}{acc}.fasta"
            print(f"→ Fetching {acc} ...")
            r = requests.get(url)

            if r.status_code == 200:
                out.write(r.text + "\n")
                print(f"   Saved FASTA entry\n")
            else:
                print(f"   !! Failed for {acc}: HTTP {r.status_code}")

    print(f"🎉 Done! FASTA sequences saved to: {OUT_FASTA}")

if __name__ == "__main__":
    main()

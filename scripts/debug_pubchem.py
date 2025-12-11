import requests
from urllib.parse import quote

def debug_pubchem_name(name: str):
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    name_q = quote(name)
    cid_url = f"{base}/compound/name/{name_q}/cids/JSON"
    print("CID URL:", cid_url)

    try:
        r_cid = requests.get(cid_url, timeout=10)
    except Exception as e:
        print("ERROR during CID request:", repr(e))
        return

    print("CID status:", r_cid.status_code)
    print("CID headers:", r_cid.headers)
    print("CID text (first 500 chars):")
    print(r_cid.text[:500])
    print("-" * 60)

    if r_cid.status_code != 200:
        print("Non-200 status -> PubChem name->CID failed.")
        return

    try:
        data = r_cid.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        print("Parsed CIDs:", cids)
        if not cids:
            print("No CID found in JSON.")
            return
        cid = cids[0]
    except Exception as e:
        print("ERROR parsing CID JSON:", repr(e))
        return

    # Now fetch properties for this CID
    props = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,Charge,TPSA,RotatableBondCount"
    prop_url = f"{base}/compound/cid/{cid}/property/{props}/JSON"
    print("PROP URL:", prop_url)

    try:
        r_prop = requests.get(prop_url, timeout=10)
    except Exception as e:
        print("ERROR during PROP request:", repr(e))
        return

    print("PROP status:", r_prop.status_code)
    print("PROP headers:", r_prop.headers)
    print("PROP text (first 500 chars):")
    print(r_prop.text[:500])

    if r_prop.status_code != 200:
        print("Non-200 status -> PubChem CID->properties failed.")
        return

    try:
        pdata = r_prop.json()
        props = pdata["PropertyTable"]["Properties"][0]
        print("Parsed properties dict:")
        for k, v in props.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("ERROR parsing PROP JSON:", repr(e))

if __name__ == "__main__":
    debug_pubchem_name("glucose")
import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol
import sys
import subprocess
import zipfile
import io
import base64
import re

from pathlib import Path
from urllib.parse import quote
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from math import sqrt


# -----------------------------
# MUTATION SUGGESTION ENGINE
# -----------------------------

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}

SAFE_SUBS = {
    "hydrophobic": ["VAL","LEU","ILE","MET","ALA"],
    "aromatic":    ["PHE","TYR","TRP"],
    "H-bond":      ["SER","THR","ASN","GLN","TYR"],
    "salt bridge": ["LYS","ARG","HIS","ASP","GLU"],  # filtered by charge sign below
    "special":     ["ALA","SER","THR"],             # gentle alternatives
    "other":       ["ALA","SER","THR","VAL","LEU"]
}

POS = {"LYS","ARG","HIS"}
NEG = {"ASP","GLU"}

def suggest_mutations(resn: str, interaction: str, lig_charge: float | None = None, top_n: int = 4):
    """
    Returns (safe_list, exploratory_list) where each list is a list of tuples:
      (mut_aa3, rationale)
    """
    r = (resn or "").upper().strip()
    interaction = (interaction or "other").strip()

    # --- SAFE: keep same interaction class
    pool = SAFE_SUBS.get(interaction, SAFE_SUBS["other"]).copy()

    # If salt-bridge, keep the same charge by default
    if interaction == "salt bridge":
        if r in POS:
            pool = [aa for aa in pool if aa in POS]
        elif r in NEG:
            pool = [aa for aa in pool if aa in NEG]

    # Remove original, dedupe
    safe = []
    for aa in pool:
        if aa == r:
            continue
        safe.append((aa, "Conservative: preserve interaction chemistry"))
    safe = safe[:top_n]

    # --- EXPLORATORY: pocket reshaping / polarity / charge flips
    exploratory = []

    # Pocket enlargement: mutate bulky → smaller
    if r in {"TRP","PHE","TYR","LEU","ILE","MET"}:
        exploratory.append(("ALA", "Exploratory: enlarge pocket / reduce steric bulk"))
        exploratory.append(("GLY", "Exploratory: enlarge pocket (higher backbone risk)"))

    # Add bulk: fill cavity / increase packing
    if r in {"ALA","VAL","SER","THR"}:
        exploratory.append(("LEU", "Exploratory: increase hydrophobic packing"))
        exploratory.append(("PHE", "Exploratory: add aromatic packing / π contacts"))

    # Polarity flips
    if interaction == "hydrophobic":
        exploratory.append(("SER", "Exploratory: introduce polarity / new H-bonding"))
    if interaction == "H-bond":
        exploratory.append(("VAL", "Exploratory: remove polarity / favor hydrophobic ligands"))

    # Charge flips (guided by ligand charge if known)
    if lig_charge is not None:
        if lig_charge < 0:
            # ligand anionic -> try adding positive
            exploratory.append(("LYS", "Exploratory: add + charge (favor anionic ligand)"))
            exploratory.append(("ARG", "Exploratory: add + charge (strong salt-bridge potential)"))
        elif lig_charge > 0:
            # ligand cationic -> try adding negative
            exploratory.append(("ASP", "Exploratory: add – charge (favor cationic ligand)"))
            exploratory.append(("GLU", "Exploratory: add – charge (salt-bridge potential)"))
    else:
        # unknown ligand charge -> generic flip options
        exploratory.append(("LYS", "Exploratory: charge flip (+)"))
        exploratory.append(("ASP", "Exploratory: charge flip (–)"))

    # Remove original + dedupe, then trim
    seen = set([r])
    exp2 = []
    for aa, why in exploratory:
        if aa in seen:
            continue
        seen.add(aa)
        exp2.append((aa, why))
    exp2 = exp2[:top_n]

    return safe, exp2

# ---------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------

def design_priority_score(res: dict) -> float:
    """
    Heuristic score in [0,1] for how interesting a residue is to engineer.
    Higher = better mutation candidate.
    """
    # distance term (closer = better)
    d = res.get("min_dist_A", np.nan)
    if np.isnan(d):
        dist_score = 0.0
    else:
        dist_score = max(0.0, min(1.0, 1.0 - (d / 6.0)))

    # interaction term
    interaction = res.get("interaction", "other")
    interaction_weights = {
        "hydrophobic": 0.7,
        "aromatic": 0.75,
        "H-bond": 0.8,
        "salt bridge": 0.9,
        "special": 0.3,
        "other": 0.5,
    }
    inter_score = interaction_weights.get(interaction, 0.5)

    # risk term
    risk = res.get("risk", "medium")
    risk_weights = {
        "low": 1.0,
        "medium": 0.6,
        "high": 0.2,
    }
    risk_score = risk_weights.get(risk, 0.6)

    # penalize structural residues
    resn = res.get("resn", "")
    structural_penalty = 0.4 if resn in {"GLY", "PRO", "CYS"} else 1.0

    score = (
        0.4 * dist_score
        + 0.3 * inter_score
        + 0.3 * risk_score
    ) * structural_penalty

    return float(max(0.0, min(1.0, score)))

AROMATIC = {"PHE", "TYR", "TRP"}
SPECIAL  = {"GLY", "PRO", "CYS"}

def residue_interaction_hint(resn: str, lig_charge: float | None = None) -> tuple[str, str, str]:
    """
    Returns (interaction, risk, notes)
    interaction: hydrophobic / H-bond / salt bridge / aromatic / special / other
    risk: low / medium / high
    notes: short design hint
    """
    r = (resn or "").upper().strip()

    # defaults
    interaction = "other"
    risk = "medium"
    notes = ""

    if r in HYDROPHOBIC:
        interaction = "hydrophobic"
        risk = "low"
        notes = "Hydrophobic contact; tune pocket size/shape via mutations."
        if r in {"PHE", "TRP"}:
            risk = "medium"

    if r in POLAR:
        interaction = "H-bond"
        risk = "medium"
        notes = "Polar contact; mutations can change H-bond network."

    if r in CHARGED_POS:
        interaction = "salt bridge"
        risk = "high"
        notes = "Charged residue; likely key for binding specificity."
        if lig_charge is not None and lig_charge < 0:
            notes = "Cationic site; favorable for anionic ligand (salt bridge)."

    if r in CHARGED_NEG:
        interaction = "salt bridge"
        risk = "high"
        notes = "Charged residue; likely key for binding specificity."
        if lig_charge is not None and lig_charge > 0:
            notes = "Anionic site; favorable for cationic ligand (salt bridge)."

    if r in AROMATIC:
        interaction = "aromatic"
        risk = "medium"
        notes = "π-stacking / hydrophobic packing; mutations can strongly alter affinity."

    if r in SPECIAL:
        interaction = "special"
        risk = "high" if r in {"CYS"} else "medium"
        if r == "GLY":
            notes = "Glycine: structural; mutations often destabilize."
        elif r == "PRO":
            notes = "Proline: backbone kink; mutations often destabilize."
        elif r == "CYS":
            notes = "Cysteine: reactive/disulfide risk; mutate with caution."

    return interaction, risk, notes


def risk_color(risk: str) -> tuple[str, str]:
    """Return (bg, fg) for a small pill."""
    r = (risk or "").lower()
    if r == "low":
        return ("#DCFCE7", "#166534")
    if r == "medium":
        return ("#FEF9C3", "#854D0E")
    if r == "high":
        return ("#FEE2E2", "#991B1B")
    return ("#E5E7EB", "#374151")

HYDROPHOBIC = {"ALA","VAL","LEU","ILE","MET","PHE","TRP","PRO"}
POLAR       = {"SER","THR","ASN","GLN","TYR","CYS"}
CHARGED_POS = {"LYS","ARG","HIS"}
CHARGED_NEG = {"ASP","GLU"}

def residue_class(resn: str) -> str:
    r = (resn or "").upper()
    if r in CHARGED_POS:
        return "charged +"
    if r in CHARGED_NEG:
        return "charged -"
    if r in POLAR:
        return "polar"
    if r in HYDROPHOBIC:
        return "hydrophobic"
    return "other"

def parse_pdb_atoms(pdb_text: str):
    """
    Very small PDB parser.
    Returns list of dicts with keys: chain, resi (int), resn, atom, x,y,z.
    """
    atoms = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            atom = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip() or "A"
            resi = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except Exception:
            continue
        atoms.append(
            {"chain": chain, "resi": resi, "resn": resn, "atom": atom, "x": x, "y": y, "z": z}
        )
    return atoms

def centroid_xyz(atoms):
    if not atoms:
        return None
    sx = sum(a["x"] for a in atoms)
    sy = sum(a["y"] for a in atoms)
    sz = sum(a["z"] for a in atoms)
    n = len(atoms)
    return (sx/n, sy/n, sz/n)

def dist(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def ligand_center_from_pdbqt(pdbqt_path):
    """
    Compute ligand center from a docked pdbqt-like file (ATOM/HETATM coords).
    Works even if it's pdbqt; coords are in the same columns.
    """
    try:
        text = Path(pdbqt_path).read_text()
    except Exception:
        return None
    atoms = parse_pdb_atoms(text)
    return centroid_xyz(atoms)

def compute_lining_residues(protein_pdb_path, pocket_pdb_path, center_xyz=None, cutoff=6.0):
    """
    Returns list of residues near pocket (within cutoff Å of ANY pocket atom),
    plus a per-residue distance to center_xyz (ligand center if available, else pocket centroid).
    """
    protein_text = Path(protein_pdb_path).read_text()
    pocket_text  = Path(pocket_pdb_path).read_text()

    prot_atoms = parse_pdb_atoms(protein_text)
    poc_atoms  = parse_pdb_atoms(pocket_text)

    if not poc_atoms or not prot_atoms:
        return [], None

    pocket_center = centroid_xyz(poc_atoms)
    use_center = center_xyz or pocket_center

    pocket_pts = [(a["x"], a["y"], a["z"]) for a in poc_atoms]

    # group atoms by residue
    res_map = {}
    for a in prot_atoms:
        key = (a["chain"], a["resi"], a["resn"])
        res_map.setdefault(key, []).append((a["x"], a["y"], a["z"]))

    lining = []
    for (chain, resi, resn), coords in res_map.items():
        # min distance residue atoms -> pocket atoms
        min_to_pocket = 1e9
        for c in coords:
            for p in pocket_pts:
                d = dist(c, p)
                if d < min_to_pocket:
                    min_to_pocket = d
                if min_to_pocket <= cutoff:
                    break
            if min_to_pocket <= cutoff:
                break

        if min_to_pocket <= cutoff:
            # distance to ligand/pocket center (use min atom distance to center)
            min_to_center = min(dist(c, use_center) for c in coords)
            lining.append({
                "chain": chain,
                "resi": resi,
                "resn": resn,
                "label": f"{resn}{resi}",
                "class": residue_class(resn),
                "min_dist_A": float(min_to_center),
            })

    # sort closest first
    lining.sort(key=lambda r: r["min_dist_A"])
    return lining, use_center

def remove_pubmed_refs(text: str) -> str:
    if not text:
        return text
    # Remove patterns like (PubMed:12345, PubMed:67890)
    text = re.sub(r"\s*\(PubMed:[^)]+\)", "", text)
    # Remove inline PubMed:12345
    text = re.sub(r"PubMed:\d+", "", text)
    return text

def pill_html(label: str, value: str, color: str = "#111827", bg: str = "#F3F4F6") -> str:
    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding:6px 12px;
        border-radius:999px;
        background:{bg};
        color:{color};
        font-size:0.85rem;
        font-weight:600;
        white-space:nowrap;
    ">
        <span style="opacity:0.75; font-weight:600;">{label}</span>
        <span style="font-weight:800;">{value}</span>
    </span>
    """
def to_streamlit_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix PyArrow object-type column errors for st.dataframe().
    Convert all object columns to plain strings.
    (Kept here in case you want it elsewhere; not used for the rank table.)
    """
    if df.empty:
        return df

    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(lambda x: "" if pd.isna(x) else str(x))

    arr = df2.to_numpy(dtype=object, copy=True)
    return pd.DataFrame(arr, columns=df2.columns).reset_index(drop=True)


def estimate_ligand_radius(mw: float) -> float:
    """Crude mapping from MW to an effective radius (Å)."""
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
    """
    Approximate shape/volume compatibility score in [0,1].
    Uses pocket volume + max distance from COM, and ligand MW.
    """
    if np.isnan(mw) or mw <= 0:
        return np.nan

    pocket_vol = row.get("volume", np.nan)
    pocket_rad = row.get("com_max_dist", np.nan)

    if np.isnan(pocket_vol) or np.isnan(pocket_rad):
        return np.nan

    lig_rad = estimate_ligand_radius(mw)
    if np.isnan(lig_rad):
        return np.nan

    lig_vol = mw  # super simple proxy for ligand volume

    vol_ratio = pocket_vol / max(lig_vol, 1e-6)
    rad_ratio = pocket_rad / max(lig_rad, 1e-6)

    rad_score = np.exp(-((rad_ratio - 1.5) / 0.75) ** 2)
    vol_score = np.exp(-((vol_ratio - 4.0) / 2.0) ** 2)

    score = 0.6 * rad_score + 0.4 * vol_score
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
STRUCT_DIR = BASE_DIR / "structures"
DOCKING_ROOT = BASE_DIR / "docking"

POCKET_CSV = RESULTS_DIR / "pockets_fpocket.csv"
LOGO_PATH = BASE_DIR / "assets" / "biolinka_logo.svg"  


# ---------------------------------------------------------------------
# STATIC UNIPROT / ALPHAFOLD METADATA (extend as needed)
# ---------------------------------------------------------------------

UNIPROT_INFO = {
    "Q99798": {
        "protein_name": "Aconitate hydratase, mitochondrial",
        "organism": "Homo sapiens (Human)",
        "function": (
            "Catalyzes the reversible isomerization of citrate to isocitrate "
            "via cis-aconitate in the TCA cycle; contributes to cellular energy "
            "production and metabolic regulation in mitochondria."
        ),
    },
    "P69905": {
        "protein_name": "Hemoglobin subunit alpha",
        "organism": "Homo sapiens (Human)",
        "function": "Oxygen transport from lung to tissues as part of hemoglobin.",
    },
    "P68871": {
        "protein_name": "Hemoglobin subunit beta",
        "organism": "Homo sapiens (Human)",
        "function": "Oxygen transport; binds heme and cooperatively binds O2.",
    },
    "P0A7A9": {
        "protein_name": "ATP synthase subunit alpha",
        "organism": "Escherichia coli",
        "function": "Key catalytic subunit of F1-F0 ATP synthase; synthesizes ATP.",
    },
}


@st.cache_data(show_spinner=False)
def fetch_uniprot_metadata(pid: str):
    """
    Fetch protein name & organism from UniProt REST API.
    Returns dict with {"protein_name": ..., "organism": ..., "function": ...}
    or None if request fails.
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{pid}.json"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        j = r.json()

        prot_name = (
            j.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", "Unknown protein")
        )

        organism = j.get("organism", {}).get("scientificName", "unknown")

        comments = j.get("comments", [])
        function_text = ""
        for c in comments:
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
                break

        return {
            "protein_name": prot_name,
            "organism": organism,
            "function": function_text or "Function not available.",
        }

    except Exception:
        return None


# ---------------------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------------------



FAVICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "favicon.png"

st.set_page_config(
    page_title="Biolinka",
    page_icon=str(FAVICON_PATH),
    layout="wide",
)

st.markdown(
    """
    <style>
      .dock-kpi-title {
        color:#6B7280;
        font-size:0.85rem;
        margin-bottom:0.25rem;
      }
      .dock-kpi-value {
        font-size:0.95rem;
        font-weight:500;
        margin-bottom:0.4rem;
      }
      .dock-pill {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        font-size:0.78rem;
        font-weight:600;
        color:white;
      }
      .dock-subtitle {
        color:#6B7280;
        font-size:0.9rem;
        line-height:1.45;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Hide Streamlit element toolbars (includes the chain-link icon) */
    div[data-testid="stElementToolbar"] { 
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Backward/alternate testids used in some versions */
    div[data-testid="stToolbar"] { 
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Ultra-safe fallback: hide any "copy/link" icon buttons */
    button[aria-label*="Copy link"],
    button[title*="Copy link"],
    button[aria-label*="Link"],
    button[title*="Link"] {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_pockets() -> pd.DataFrame:
    df = pd.read_csv(POCKET_CSV)

    def biosensor_score(row):
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
            w_drug * drug_term
            + w_vol * vol_term
            + w_hyd * hyd_term
            + w_pol * pol_term
            + w_alpha * alpha_term
            + w_flex * flex_term
        )
        return max(0.0, min(1.0, score))

    df["biosensor_score"] = df.apply(biosensor_score, axis=1)

    def resolve_uniprot_field(pid: str):
        if pid in UNIPROT_INFO:
            return UNIPROT_INFO[pid]

        meta = fetch_uniprot_metadata(pid)
        if meta:
            return meta

        return {
            "protein_name": "Unknown protein",
            "organism": "unknown",
            "function": "Function not available.",
        }

    meta_series = df["protein_id"].map(resolve_uniprot_field)

    df["protein_name"] = meta_series.map(lambda m: m["protein_name"])
    df["source_organism"] = meta_series.map(lambda m: m["organism"])
    df["biological_function"] = meta_series.map(lambda m: m["function"])

    return df


df_all = load_pockets()


# ---------------------------------------------------------------------
# PUBCHEM LIGAND FETCH
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_ligand_from_pubchem(name: str):
    """
    Fetch basic physicochemical properties for a ligand name from PubChem.
    """
    name = (name or "").strip()
    if not name:
        return None

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    cid_url = f"{base}/compound/name/{quote(name)}/cids/JSON"
    try:
        r_cid = requests.get(cid_url, timeout=6)
    except Exception:
        return None

    if r_cid.status_code != 200:
        return None

    try:
        cid_data = r_cid.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        cid = cids[0]
    except Exception:
        return None

    props = (
        "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,"
        "Charge,TPSA,RotatableBondCount"
    )
    prop_url = f"{base}/compound/cid/{cid}/property/{props}/JSON"

    try:
        r = requests.get(prop_url, timeout=6)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
        p = data["PropertyTable"]["Properties"][0]
    except Exception:
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

    if np.isnan(mw):
        size_class = "unknown"
    elif mw < 180:
        size_class = "small"
    elif mw < 400:
        size_class = "medium"
    else:
        size_class = "large"

    if not np.isnan(tpsa) and tpsa > 90:
        polarity = "polar"
    elif not np.isnan(xlogp) and xlogp > 1.5 and (np.isnan(tpsa) or tpsa < 60):
        polarity = "hydrophobic"
    else:
        polarity = "mixed"

    if charge > 0:
        charge_class = "positive"
    elif charge < 0:
        charge_class = "negative"
    else:
        charge_class = "neutral"

    return {
        "name": name,
        "cid": cid,
        "mw": mw,
        "xlogp": xlogp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "charge": charge,
        "rotb": rotb,
        "size_class": size_class,
        "polarity": polarity,
        "charge_class": charge_class,
    }


def fetch_pubchem_sdf_by_name(name: str) -> str | None:
    """
    Fetch a 3D (or 2D fallback) SDF for a small molecule from PubChem by name.
    """
    name = (name or "").strip()
    if not name:
        return None

    try:
        cid_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{quote(name)}/cids/JSON"
        )
        cid_resp = requests.get(cid_url, timeout=12)
        if cid_resp.status_code != 200:
            return None

        cid_data = cid_resp.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        cid = cids[0]

        for record_type in ("3d", "2d"):
            sdf_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                f"compound/cid/{cid}/record/SDF/?record_type={record_type}"
            )
            sdf_resp = requests.get(sdf_url, timeout=20)
            if sdf_resp.status_code == 200 and sdf_resp.text.strip():
                return sdf_resp.text
    except Exception:
        return None

    return None


# ---------------------------------------------------------------------
# LIGAND–POCKET COMPATIBILITY
# ---------------------------------------------------------------------

def compute_ligand_pocket_compat(
    row: pd.Series,
    ligand_info: dict | None,
    profile: dict | None,
    mw_num: float,
    xlogp_num: float,
    tpsa_num: float,
    charge_num: float,
) -> float:
    """
    Heuristic ligand–pocket compatibility score in [0, 1].
    """
    if ligand_info is None and profile is None:
        return np.nan

    if not np.isnan(mw_num):
        if mw_num < 150:
            lig_size = "very_small"
        elif mw_num < 300:
            lig_size = "small"
        elif mw_num < 500:
            lig_size = "medium"
        else:
            lig_size = "large"
    elif profile is not None:
        lig_size = profile.get("size", "unknown")
    else:
        lig_size = "unknown"

    if ligand_info is not None:
        lig_pol = ligand_info.get("polarity", "mixed")
    elif profile is not None:
        lig_pol = profile.get("polarity", "mixed")
    else:
        lig_pol = "mixed"

    if ligand_info is not None:
        lig_charge_class = ligand_info.get("charge_class", "neutral")
    elif profile is not None:
        lig_charge_class = profile.get("charge", "neutral")
    else:
        lig_charge_class = "neutral"

    if not np.isnan(tpsa_num):
        if tpsa_num < 60:
            lig_pol_strength = "low"
        elif tpsa_num < 90:
            lig_pol_strength = "medium"
        else:
            lig_pol_strength = "high"
    else:
        lig_pol_strength = "unknown"

    vol = row.get("volume", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    pol_score = row.get("polarity_score", np.nan)
    chg_score = row.get("charge_score", 0.0)
    drug = row.get("druggability_score", 0.0)

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

    if np.isnan(hyd):
        pocket_env = "mixed"
    elif hyd < 15:
        pocket_env = "polar"
    elif hyd < 40:
        pocket_env = "mixed"
    else:
        pocket_env = "hydrophobic"

    if np.isnan(pol_score):
        pocket_pol_strength = "unknown"
    elif pol_score < 5:
        pocket_pol_strength = "low"
    elif pol_score < 10:
        pocket_pol_strength = "medium"
    else:
        pocket_pol_strength = "high"

    if chg_score > 1:
        pocket_charge_class = "positive"
    elif chg_score < -1:
        pocket_charge_class = "negative"
    else:
        pocket_charge_class = "neutral"

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
    pol_score_match = pol_strength_matrix.get(
        (lig_pol_strength, pocket_pol_strength), 0.7
    )

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
        w_size * size_score
        + w_hydro * hydro_score
        + w_pol * pol_score_match
        + w_charge * charge_score
        + w_drug * drug_bonus
    )

    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------
# LIGAND PROPERTY DESCRIPTORS + COLOR HELPERS
# ---------------------------------------------------------------------

def describe_mw(mw: float) -> str:
    if np.isnan(mw):
        return "size: unknown"
    if mw < 150:
        return "very small molecule"
    elif mw < 300:
        return "small molecule"
    elif mw < 500:
        return "medium-sized ligand"
    else:
        return "large / bulky ligand"


def describe_logp(x: float, fallback_polarity: str | None = None) -> str:
    if np.isnan(x):
        if fallback_polarity:
            if fallback_polarity == "polar":
                return "expected polar ligand"
            elif fallback_polarity == "hydrophobic":
                return "expected hydrophobic ligand"
            elif fallback_polarity == "mixed":
                return "amphipathic ligand"
        return "polarity: unknown"
    if x < -1.5:
        return "extremely hydrophilic"
    elif x < 0.5:
        return "hydrophilic"
    elif x < 2.5:
        return "amphipathic"
    elif x < 4.5:
        return "hydrophobic"
    else:
        return "very hydrophobic"


def describe_tpsa(t: float) -> str:
    if np.isnan(t):
        return "H-bonding: unknown"
    if t < 20:
        return "very low polarity"
    elif t < 60:
        return "low polar surface"
    elif t < 90:
        return "moderate polar surface"
    elif t < 120:
        return "highly polar"
    else:
        return "very highly polar"


def describe_charge(q: float | str, fallback: str | None = None) -> str:
    if isinstance(q, str):
        if q == "positive":
            return "positively charged"
        elif q == "negative":
            return "negatively charged"
        elif q == "neutral":
            return "overall neutral"
        return "charge: unknown"

    if np.isnan(q):
        if fallback:
            return describe_charge(fallback)
        return "charge: unknown"

    if q > 0:
        return "positively charged"
    elif q < 0:
        return "negatively charged"
    else:
        return "overall neutral"


def color_for_mw(mw: float) -> str:
    if np.isnan(mw):
        return "#E5E7EB"
    if mw < 150:
        return "#10B981"
    elif mw < 300:
        return "#2563EB"
    elif mw < 500:
        return "#7E22CE"
    else:
        return "#4C1D95"


def color_for_logp(x: float) -> str:
    if np.isnan(x):
        return "#E5E7EB"
    if x > 4.5:
        return "#4C1D95"
    elif x > 2.5:
        return "#7E22CE"
    elif x > 0.5:
        return "#2563EB"
    elif x > -1.5:
        return "#10B981"
    else:
        return "#0D9488"


def color_for_tpsa(t: float) -> str:
    if np.isnan(t):
        return "#E5E7EB"
    if t < 20:
        return "#4C1D95"
    elif t < 60:
        return "#7E22CE"
    elif t < 90:
        return "#2563EB"
    elif t < 120:
        return "#10B981"
    else:
        return "#0D9488"


def color_for_charge(q: float | str) -> str:
    if isinstance(q, str):
        if q == "positive":
            return "#DC2626"
        if q == "negative":
            return "#2563EB"
        if q == "neutral":
            return "#10B981"
        return "#E5E7EB"

    if np.isnan(q):
        return "#E5E7EB"

    if q > 0:
        return "#DC2626"
    elif q < 0:
        return "#2563EB"
    else:
        return "#10B981"


def soft_color_for_compat(score: float) -> tuple[str, str]:
    """Returns (bg_color, text_color) for soft pastel pills."""
    if score is None or np.isnan(score):
        return "#E5E7EB", "#374151"

    s = max(0.0, min(1.0, float(score)))

    if s < 0.2:
        return "#FEE2E2", "#991B1B"
    elif s < 0.4:
        return "#FFEDD5", "#9A3412"
    elif s < 0.6:
        return "#FEF9C3", "#854D0E"
    elif s < 0.8:
        return "#DCFCE7", "#166534"
    else:
        return "#BBF7D0", "#14532D"


# ---------------------------------------------------------------------
# POCKET INTERPRETATION
# ---------------------------------------------------------------------

def interpret_pocket(row: pd.Series) -> str:
    """Return HTML-formatted description of the pocket."""
    prot = row.get("protein_id", "–")
    pname = row.get("protein_name", "Unknown protein")
    org = row.get("source_organism", "unknown organism")
    vol = row.get("volume", np.nan)
    drug = row.get("druggability_score", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    pol = row.get("polarity_score", np.nan)
    chg = row.get("charge_score", np.nan)
    bios = row.get("biosensor_score", np.nan)
    fun_raw = row.get("biological_function", "")
    fun = remove_pubmed_refs(fun_raw)

    lines: list[str] = []

    lines.append(
        f"<p><b>Protein</b> <code>{prot}</code> — "
        f"<i>{pname}</i> ({org}).</p>"
    )

    if fun:
        lines.append(f"<p><b>Biological role:</b> {fun}</p>")

    bio_lines: list[str] = []

    if not np.isnan(vol):
        if vol < 250:
            bio_lines.append(
                f"Volume ≈ {vol:.0f} Å³ — compact pocket suited for small ligands."
            )
        elif vol < 600:
            bio_lines.append(
                f"Volume ≈ {vol:.0f} Å³ — medium-sized pocket, compatible with many metabolites or drugs."
            )
        else:
            bio_lines.append(
                f"Volume ≈ {vol:.0f} Å³ — large cavity, can host bulky ligands or multi-contact binding."
            )

    if not np.isnan(hyd):
        if hyd < 15:
            bio_lines.append(
                "Hydrophobicity is <b>low</b> → pocket is mostly polar / H-bonding."
            )
        elif hyd < 40:
            bio_lines.append(
                "Hydrophobicity is <b>moderate</b> → mixed polar/non-polar environment."
            )
        else:
            bio_lines.append(
                "Hydrophobicity is <b>high</b> → pocket favors hydrophobic or amphipathic ligands."
            )

    if not np.isnan(pol):
        if pol < 5:
            bio_lines.append(
                "Polarity score is <b>low</b> → few strong polar contacts expected."
            )
        elif pol < 10:
            bio_lines.append(
                "Polarity score is <b>moderate</b> → can support several polar interactions."
            )
        else:
            bio_lines.append(
                "Polarity score is <b>high</b> → rich in polar groups, strong H-bond network possible."
            )

    if not np.isnan(chg):
        if chg > 1:
            bio_lines.append(
                "Pocket charge is <b>overall positive</b> → favorable to anionic / acidic ligands."
            )
        elif chg < -1:
            bio_lines.append(
                "Pocket charge is <b>overall negative</b> → favorable to cationic / basic ligands."
            )
        else:
            bio_lines.append(
                "Pocket charge is <b>near neutral</b> → not strongly biased toward charged ligands."
            )

    if not np.isnan(drug):
        if drug > 0.7:
            bio_lines.append(
                "Druggability score is <b>high</b> → classical small-molecule drug–like site."
            )
        elif drug > 0.3:
            bio_lines.append(
                "Druggability score is <b>moderate</b> → may still be usable with optimization."
            )
        else:
            bio_lines.append(
                "Druggability score is <b>low</b> → more challenging scaffold, but not impossible."
            )

    if bio_lines:
        lines.append("<p><b>Pocket biophysics:</b></p>")
        lines.append("<ul>")
        for bl in bio_lines:
            lines.append(f"<li>{bl}</li>")
        lines.append("</ul>")

    if not np.isnan(bios):
        bios_pct = bios * 100.0
        lines.append(
            f"<p><b>Biosensor heuristic:</b> {bios_pct:.1f}% — combined view of volume, "
            "hydrophobicity, polarity, flexibility and fpocket druggability.</p>"
        )
        lines.append(
            "<p>This pocket could be engineered as a sensing module by introducing "
            "mutations in key lining residues and coupling conformational change "
            "to a reporting domain (fluorescent protein, transcription factor, etc.).</p>"
        )

    return "\n".join(lines)

# ---------------------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------------------
def render_best_pocket_info(best_pocket: pd.Series):
    prot = best_pocket.get("protein_id", "–")
    pocket_id = best_pocket.get("pocket_id", "–")
    bios = best_pocket.get("biosensor_score", np.nan)
    drugg = best_pocket.get("druggability_score", np.nan)
    vol = best_pocket.get("volume", np.nan)
    hyd = best_pocket.get("hydrophobicity_score", np.nan)
    org = best_pocket.get("source_organism", "unknown")
    pname = best_pocket.get("protein_name", "Unknown protein")

    compat = best_pocket.get("ligand_compat_score", np.nan)
    qshape = best_pocket.get("quickshape_score", np.nan)

    bios_str = "–" if np.isnan(bios) else f"{bios*100:.2f}".rstrip("0").rstrip(".") + "%"
    vol_str = "–" if np.isnan(vol) else f"{vol:.1f} Å³"
    hyd_str = "–" if np.isnan(hyd) else f"{hyd:.1f}"
    compat_str = "–" if np.isnan(compat) else f"{compat*100:.2f}".rstrip("0").rstrip(".") + "%"
    qshape_str = "–" if np.isnan(qshape) else f"{qshape*100:.2f}".rstrip("0").rstrip(".") + "%"

    af_url = f"https://alphafold.ebi.ac.uk/entry/AF-{prot}-F1"
    up_url = f"https://www.uniprot.org/uniprotkb/{prot}/entry"

    bg, fg = soft_color_for_compat(compat)

    html = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px 22px; font-size:0.95rem;">
      <div><b>Protein ID</b><br>{prot}<br>
        <a href="{up_url}" target="_blank">UniProt →</a><br>
        <a href="{af_url}" target="_blank">AlphaFold →</a>
      </div>
      <div><b>Ligand compatibility</b><br>
        <span style="padding:3px 10px;border-radius:999px;background:{bg};color:{fg};font-weight:600;">
          {compat_str}
        </span>
      </div>

      <div><b>Protein name</b><br>{pname}</div>
      <div><b>Biosensor score</b><br>{bios_str}</div>

      <div><b>Source organism</b><br>{org}</div>
      <div><b>QuickShape</b><br>{qshape_str}</div>

      <div><b>Pocket ID</b><br>{pocket_id}</div>
      <div><b>Hydrophobicity</b><br>{hyd_str}</div>

      <div></div>
      <div><b>Volume</b><br>{vol_str}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
if LOGO_PATH.exists():
    with open(LOGO_PATH, "r") as f:
        svg_content = f.read()

    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="width: 600px; max-width: 100%;">
        """  # logo
        + svg_content
        + """
            </div>
            <div style="
                font-size: 1.6rem;
                margin-top: 0px;
                margin-bottom: 3rem;
                color: #D3D3D3;
            ">
                Engineer Biological Sensors
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🧬 BIOLINKA")  # fallback


# ---------------------------------------------------------------------
# ANALYTE SELECTION
# ---------------------------------------------------------------------


st.markdown(
    """
    <h2 style="
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 0.25 rem;
    ">
        What do you want to sense?
    </h2>
    """,
    unsafe_allow_html=True,
)

if "analyte_query" not in st.session_state:
    st.session_state["analyte_query"] = ""

ANALYTE_PROFILES = {
    # Metabolites
    "glucose":     {"size": "medium", "polarity": "polar",       "charge": "neutral", "cat": "met"},
    "sugar":       {"size": "medium", "polarity": "polar",       "charge": "neutral", "cat": "met"},
    "lactate":     {"size": "small",  "polarity": "polar",       "charge": "negative","cat": "met"},
    "lactic acid": {"size": "small",  "polarity": "polar",       "charge": "negative","cat": "met"},
    "urea":        {"size": "small",  "polarity": "polar",       "charge": "neutral", "cat": "met"},

    # Neurotransmitters
    "dopamine":    {"size": "small", "polarity": "mixed",        "charge": "positive","cat": "nt"},
    "serotonin":   {"size": "small", "polarity": "mixed",        "charge": "positive","cat": "nt"},

    # Hormones / Steroids
    "cortisol":    {"size": "medium", "polarity": "hydrophobic", "charge": "neutral", "cat": "st"},
    "steroid":     {"size": "medium", "polarity": "hydrophobic", "charge": "neutral", "cat": "st"},

    # Drugs
    "drug":        {"size": "medium", "polarity": "mixed",       "charge": "neutral", "cat": "dr"},
    "antibiotic":  {"size": "medium", "polarity": "mixed",       "charge": "neutral", "cat": "dr"},

    # Ions
    "ion":         {"size": "small", "polarity": "polar",        "charge": "positive", "cat": "ion"},
    "na+":         {"size": "small", "polarity": "polar",        "charge": "positive", "cat": "ion"},
    "k+":          {"size": "small", "polarity": "polar",        "charge": "positive", "cat": "ion"},
}

st.markdown(
    """
    <style>
        /* Search bar: vertically centered text, left aligned */
        div[data-testid="stTextInput"] input {
            height: 3.2rem !important;
            line-height: 3.2rem !important;   /* centers text vertically */
            font-size: 1.05rem !important;
            padding: 0 1rem !important;       /* no top/bottom padding */
            box-sizing: border-box !important;
        }

        div[data-testid="stTextInput"] {
            width: 1000px !important;
            margin: -0.3rem auto 0 auto !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        /* Reduce space above the search bar */
        div[data-testid="stTextInput"] {
            margin-top: -2.25rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Free-text analyte input (still remembered in session_state)
analyte_input = st.text_input(
    "Analyte",  # non-empty label for accessibility
    value=st.session_state["analyte_query"],
    placeholder="e.g. glucose, dopamine, cortisol, ibuprofen, custom drug name…",
    label_visibility="collapsed",
)

raw_analyte = analyte_input.strip()
aq = raw_analyte.lower()

st.session_state["analyte_query"] = raw_analyte

# ------------------------------------------------------------------
# COMMON ANALYTES ROW
# ------------------------------------------------------------------

CATEGORY_COLORS = {
    "met": ("#E8F0FF", "#1E40AF"),
    "nt":  ("#FDECF3", "#9D174D"),
    "st":  ("#FEF7D1", "#B45309"),
    "dr":  ("#F3F4F6", "#374151"),
    "ion": ("#EEE6FF", "#5B21B6"),
}


def preset_label(key: str) -> str:
    k = key.lower()
    if k == "na+":
        return "Na⁺ (sodium)"
    if k == "k+":
        return "K⁺ (potassium)"
    return key.capitalize()


st.markdown(
    """
<p style="
  text-align:center;
  font-size:0.9rem;
  color:#6B7280;
  margin-top:0.75rem;
  margin-bottom:0.25rem;
">
  Common analytes
</p>
""",
    unsafe_allow_html=True,
)

keys = list(ANALYTE_PROFILES.keys())

# Build pure HTML string (no leading spaces before tags)
pills_html = ""

for i, key in enumerate(keys):
    cat = ANALYTE_PROFILES[key]["cat"]
    bg, fg = CATEGORY_COLORS.get(cat, ("#F3F4F6", "#111827"))

    pills_html += (
        f'<span style="'
        f'background:{bg};'
        f'color:{fg};'
        'padding:4px 12px;'
        'border-radius:999px;'
        'font-size:0.78rem;'
        'font-weight:500;'
        'display:inline-block;'
        'white-space:nowrap;'
        '">'
        f"{preset_label(key)}"
        "</span>"
    )

    # line break after 8th pill → rest on second row
    if i == 9:
        pills_html += '<div style="flex-basis:100%; height:0;"></div>'

pill_container = (
    '<div style="'
    'display:flex;'
    'flex-wrap:wrap;'
    'justify-content:center;'
    'gap:0.4rem 0.6rem;'
    'margin-bottom:0.5rem;'
    'margin-top:0.25rem;'
    '">'
    f"{pills_html}"
    "</div>"
)

st.markdown(pill_container, unsafe_allow_html=True)
# If no analyte yet: keep the landing page simple, show presets, and stop.
# Stop here if nothing typed yet (clean landing page)
if not aq:
    st.stop()

# ---------------------------------------------------------------------
# LIGAND INFO (PubChem) + ANALYTE PROFILE
# ---------------------------------------------------------------------

# Use free-text analyte + optional heuristic profile
profile = ANALYTE_PROFILES.get(aq, None)
ligand_info = fetch_ligand_from_pubchem(raw_analyte) if raw_analyte else None


def _to_float(val):
    try:
        return float(val)
    except Exception:
        return np.nan


mw_num = _to_float(ligand_info.get("mw")) if ligand_info else np.nan
xlogp_num = _to_float(ligand_info.get("xlogp")) if ligand_info else np.nan
tpsa_num = _to_float(ligand_info.get("tpsa")) if ligand_info else np.nan
charge_num = _to_float(ligand_info.get("charge")) if ligand_info else np.nan


# ---------------------------------------------------------------------
# DOCKING SETUP HELPER
# ---------------------------------------------------------------------

analyte_for_docking = raw_analyte
analyte_folder_name = (
    analyte_for_docking.replace(" ", "_") if analyte_for_docking else ""
)

if "dock_zip_bytes" not in st.session_state:
    st.session_state["dock_zip_bytes"] = None
if "dock_zip_log" not in st.session_state:
    st.session_state["dock_zip_log"] = ""


def build_docking_package_zip(analyte: str) -> tuple[bytes | None, str]:
    """
    For a given analyte name:
      1) run prepare_docking_setup.py
      2) collect docking_targets.csv, ligand.sdf (if any) and *_box.txt files
      3) return (zip_bytes, log_string) or (None, error_log)
    """
    analyte = analyte.strip()
    if not analyte:
        return None, "No analyte provided."

    folder_name = analyte.replace(" ", "_")
    script_path = BASE_DIR / "scripts" / "prepare_docking_setup.py"

    if not script_path.exists():
        return None, f"Docking script not found at: {script_path}"

    cmd = [
        sys.executable,
        str(script_path),
        "--analyte",
        analyte,
        "--top_n",
        "8",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )
    full_log = (result.stdout or "") + "\n" + (result.stderr or "")

    if result.returncode != 0:
        return None, f"prepare_docking_setup failed:\n{full_log}"

    docking_subdir = DOCKING_ROOT / folder_name
    targets_csv = docking_subdir / "docking_targets.csv"
    ligand_local = DOCKING_ROOT / "ligands" / f"{folder_name}.sdf"
    box_files = list(docking_subdir.glob("*_box.txt"))

    if not docking_subdir.exists() or not targets_csv.exists():
        return None, f"Docking directory or docking_targets.csv missing in {docking_subdir}"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if targets_csv.exists():
            zf.write(
                targets_csv,
                arcname=f"{folder_name}/docking_targets.csv",
            )

        if ligand_local.exists():
            zf.write(
                ligand_local,
                arcname=f"{folder_name}/ligand.sdf",
            )

        for box_file in box_files:
            zf.write(
                box_file,
                arcname=f"{folder_name}/{box_file.name}",
            )

        log_name = f"{folder_name}/prepare_docking_setup.log"
        zf.writestr(log_name, full_log)

    return zip_buffer.getvalue(), full_log


# ---------------------------------------------------------------------
# POCKET FILTERS
# ---------------------------------------------------------------------

with st.expander("Pocket filters", expanded=False):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        vol_min, vol_max = st.slider(
            "Pocket volume (Å³)", 0.0, 1500.0, (0.0, 1500.0), step=25.0
        )

    with c2:
        drug_min = st.slider(
            "Min. druggability score", 0.0, 1.0, 0.0, step=0.05
        )

    with c3:
        hyd_min, hyd_max = st.slider(
            "Hydrophobicity score", -20.0, 80.0, (-20.0, 80.0), step=2.0
        )

    with c4:
        bios_min = st.slider(
            "Min. biosensor heuristic score", 0.0, 1.0, 0.0, step=0.05
        )

df_filtered = df_all[
    (df_all["volume"].between(vol_min, vol_max))
    & (df_all["druggability_score"] >= drug_min)
    & (df_all["hydrophobicity_score"].between(hyd_min, hyd_max))
    & (df_all["biosensor_score"] >= bios_min)
].copy()

# ---------------------------------------------------------------------
# LIGAND–POCKET COMPATIBILITY & QUICKSHAPE COLUMNS
# ---------------------------------------------------------------------

if ligand_info is not None or profile is not None:
    df_filtered["ligand_compat_score"] = df_filtered.apply(
        lambda row: compute_ligand_pocket_compat(
            row,
            ligand_info=ligand_info,
            profile=profile,
            mw_num=mw_num,
            xlogp_num=xlogp_num,
            tpsa_num=tpsa_num,
            charge_num=charge_num,
        ),
        axis=1,
    )
else:
    df_filtered["ligand_compat_score"] = np.nan

if not np.isnan(mw_num):
    df_filtered["quickshape_score"] = df_filtered.apply(
        lambda row: compute_quickshape(row, mw=mw_num),
        axis=1,
    )
else:
    df_filtered["quickshape_score"] = np.nan

# ---------------------------------------------------------------------
# METRICS STRIP
# ---------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="text-align:center;">
            <p style="color:#6B7280; font-size:0.9rem; margin-bottom:0.2rem;">
                Proteins (after filters)
            </p>
            <h2 style="margin:0; font-size:2.3rem; font-weight:600;">
                {df_filtered['protein_id'].nunique()}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="text-align:center;">
            <p style="color:#6B7280; font-size:0.9rem; margin-bottom:0.2rem;">
                Candidate pockets
            </p>
            <h2 style="margin:0; font-size:2.3rem; font-weight:600;">
                {len(df_filtered)}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    # pick mean ligand compatibility or biosensor score
    label = "Mean ligand compatibility"
    mean_value = "0%"

    if not df_filtered.empty:
        if (
            "ligand_compat_score" in df_filtered.columns
            and df_filtered["ligand_compat_score"].notna().any()
        ):
            value = df_filtered["ligand_compat_score"].dropna().mean() * 100
            mean_value = f"{value:.2f}".rstrip("0").rstrip(".") + "%"
        else:
            value = df_filtered["biosensor_score"].dropna().mean() * 100
            mean_value = f"{value:.2f}".rstrip("0").rstrip(".") + "%"
            label = "Mean biosensor score"

    st.markdown(
        f"""
        <div style="text-align:center;">
            <p style="color:#6B7280; font-size:0.9rem; margin-bottom:0.2rem;">
                {label}
            </p>
            <h2 style="margin:0; font-size:2.3rem; font-weight:600;">
                {mean_value}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------
# IF NOTHING PASSED THE FILTERS
# ---------------------------------------------------------------------

if df_filtered.empty:
    st.warning("No pockets match these requirements. Relax the filters a bit.")
    st.stop()

# ---------------------------------------------------------------------
# INITIALIZE POCKET SELECTION STATE (ONCE)
# ---------------------------------------------------------------------

# rank pockets once (no UI yet)
df_ranked = (
    df_filtered.sort_values(
        ["ligand_compat_score", "biosensor_score"], ascending=False
    )
    if "ligand_compat_score" in df_filtered.columns
       and df_filtered["ligand_compat_score"].notna().any()
    else df_filtered.sort_values("biosensor_score", ascending=False)
)

df_top = df_ranked.head(20).reset_index(drop=True)
df_top["rank"] = df_top.index + 1
df_top["pocket_key"] = (
    df_top["protein_id"].astype(str) + "::" + df_top["pocket_id"].astype(str)
)

# initialize session state key ONCE
if "best_pocket_key" not in st.session_state:
    st.session_state["best_pocket_key"] = df_top.loc[0, "pocket_key"]

# resolve current best pocket
match = df_top[df_top["pocket_key"] == st.session_state["best_pocket_key"]]
best_pocket = match.iloc[0] if not match.empty else df_top.iloc[0]

# ---------------------------------------------------------------------
# Small helper: vertical divider
# ---------------------------------------------------------------------
def vertical_divider(height_px: int = 760, shrink: float = 0.8):
    effective_height = int(height_px * shrink)

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: {height_px}px;
        ">
            <div style="
                width: 1px;
                height: {effective_height}px;
                background: #E5E7EB;
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Small helper: render lining table
# ---------------------------------------------------------------------
def render_lining_table(lining_res: list[dict], charge_num: float):
    st.markdown("**Pocket lining residues (within 6 Å of pocket)**")

    lig_charge_for_hint = None
    try:
        lig_charge_for_hint = float(charge_num) if not np.isnan(charge_num) else None
    except Exception:
        lig_charge_for_hint = None

    for r in lining_res or []:
        inter, risk, notes = residue_interaction_hint(
            r.get("resn", ""),
            lig_charge=lig_charge_for_hint,
        )
        r["interaction"] = inter
        r["risk"] = risk
        r["notes"] = notes
        
    for r in lining_res:
        r["design_score"] = design_priority_score(r)

    df_lining_raw = pd.DataFrame(lining_res or [])

    if df_lining_raw.empty:
        st.info("No pocket lining residues found within cutoff.")
        return

    df_lining_raw["resi"] = pd.to_numeric(df_lining_raw["resi"], errors="coerce")
    df_lining_raw = df_lining_raw.sort_values("min_dist_A")

    # keep one row per residue (chain + resi)
    df_lining_raw = df_lining_raw.drop_duplicates(
        subset=["chain", "resi"],
        keep="first",
    )

    df_lining = df_lining_raw[
        [
            "label",
            "resn",
            "class",
            "interaction",
            "risk",
            "min_dist_A",
            "chain",
            "resi",
            "notes",
        ]
    ].rename(columns={
        "label": "Residue",
        "resn": "AA",
        "class": "Type",
        "interaction": "Likely interaction",
        "risk": "Mutation risk",
        "min_dist_A": "Min distance (Å)",
        "chain": "Chain",
        "resi": "Resi",
        "notes": "Notes",
    })

    df_lining["Min distance (Å)"] = (
        pd.to_numeric(df_lining["Min distance (Å)"], errors="coerce").round(2)
    )

    df_lining["Residue"] = df_lining["Residue"].astype(str)
    df_lining["AA"] = df_lining["AA"].astype(str)
    df_lining["Type"] = df_lining["Type"].astype(str)
    df_lining["Chain"] = df_lining["Chain"].astype(str)
    df_lining["Resi"] = pd.to_numeric(df_lining["Resi"], errors="coerce").astype("Int64")

    gb = GridOptionsBuilder.from_dataframe(df_lining)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
    )
    gb.configure_grid_options(domLayout="normal")

    AgGrid(
        df_lining,
        gridOptions=gb.build(),
        theme="streamlit",
        height=260,
        fit_columns_on_grid_load=True,
    )

# ---------------------------------------------------------------------
# Pre-compute paths + lining residues once (so we can display them in row 2)
# ---------------------------------------------------------------------
prot_id = str(best_pocket["protein_id"])
pocket_num = str(best_pocket["pocket_id"])
protein_pdb = STRUCT_DIR / f"{prot_id}.pdb"
pocket_pdb = STRUCT_DIR / f"{prot_id}_out" / "pockets" / f"pocket{pocket_num}_atm.pdb"

# Docked pose path (if exists)
interpreted_name = st.session_state.get("interpreted_name", None)
analyte_for_docking = (interpreted_name or analyte_for_docking or "").strip()
analyte_folder_name = analyte_for_docking.replace(" ", "_") if analyte_for_docking else None

docked_pose = None
if analyte_folder_name:
    docked_dir = BASE_DIR / "docking" / analyte_folder_name / "docked"
    docked_pose = docked_dir / f"{prot_id}_pocket{pocket_num}_{analyte_folder_name}_out.pdbqt"

lig_center = None
if docked_pose is not None and docked_pose.exists():
    lig_center = ligand_center_from_pdbqt(docked_pose)

lining_res, used_center = [], None
if protein_pdb.exists() and pocket_pdb.exists():
    lining_res, used_center = compute_lining_residues(
        protein_pdb_path=protein_pdb,
        pocket_pdb_path=pocket_pdb,
        center_xyz=lig_center,
        cutoff=6.0,
    )


# ---------------------------------------------------------------------
# ONE SPLIT LAYOUT (single vertical divider) + stacked content
# ---------------------------------------------------------------------
mainL, mainMid, mainR = st.columns([1, 0.02, 1])

with mainMid:
    # make it tall enough to cover both stacked blocks
    vertical_divider(height_px=760)  # adjust if needed

with mainL:
    # ---- Top-left
    st.subheader("Best Matching Pocket")
    render_best_pocket_info(best_pocket)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ---- Bottom-left
    st.subheader("Biological context & pocket interpretation")
    interp_html = interpret_pocket(best_pocket)
    st.markdown(
        f"""
        <div style="font-size: 0.92rem; line-height: 1.45;">
          {interp_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with mainR:
    # ---- Top-right
    st.subheader("3D viewer")

    vcol, lcol = st.columns([5, 1])

    with vcol:
        show_ligand_pose = False
        if docked_pose is not None and docked_pose.exists():
            show_ligand_pose = st.checkbox(
                "Show docked ligand pose (Vina)",
                value=True,
                key=f"show_docked_pose_{st.session_state['best_pocket_key']}",
            )

        if not protein_pdb.exists():
            st.warning(f"Protein structure not found: {protein_pdb}")
        else:
            try:
                view = py3Dmol.view(width=600, height=320)
                model_index = 0

                with open(protein_pdb, "r") as f:
                    view.addModel(f.read(), "pdb")
                view.setStyle(
                    {"model": model_index},
                    {"cartoon": {"opacity": 0.75, "color": "0xCCCCCC"}},
                )

                if pocket_pdb.exists():
                    model_index += 1
                    with open(pocket_pdb, "r") as f:
                        view.addModel(f.read(), "pdb")
                    view.setStyle(
                        {"model": model_index},
                        {
                            "stick": {"radius": 0.25, "color": "0xE91E63"},
                            "sphere": {"radius": 0.8, "color": "0xF06292"},
                        },
                    )

                if show_ligand_pose and docked_pose is not None and docked_pose.exists():
                    model_index += 1
                    with open(docked_pose, "r") as f:
                        view.addModel(f.read(), "pdb")
                    view.setStyle(
                        {"model": model_index},
                        {
                            "stick": {"radius": 0.22, "color": "0xFF5722"},
                            "sphere": {"radius": 0.5, "color": "0xFFC107"},
                        },
                    )
                    view.zoomTo({"model": model_index})
                else:
                    view.zoomTo({"model": 1} if pocket_pdb.exists() else {})

                view.setZoomLimits(10, 300)
                view.setBackgroundColor("0xFFFFFF")
                components.html(view._make_html(), height=340, scrolling=False)

            except Exception as e:
                st.warning(f"Viewer error: {e}")

    with lcol:
        st.markdown(
            """
            <div style="
                margin-top: 40px;
                background: rgba(255, 255, 255, 0.95);
                padding: 8px 12px;
                border-radius: 10px;
                font-size: 0.78rem;
                line-height: 1.4;
                box-shadow: 0 2px 6px rgba(0,0,0,0.10);
            ">
              <b>Legend</b><br>
              <span style="color:#4B5563;">Protein </span><br>
              <span style="color:#9D174D;">Pocket </span><br>
              <span style="color:#C2410C;">Docked ligand </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # ---- Bottom-right
    render_lining_table(lining_res=lining_res, charge_num=charge_num)

    st.markdown("---")
    st.subheader("Pocket design panel")

    if not lining_res:
        st.info("No lining residues available for design analysis.")
    else:
        df_design = pd.DataFrame(lining_res)

        # ---- Filters
        f1, f2, f3 = st.columns(3)

        with f1:
            allowed_risk = st.multiselect(
                "Mutation risk",
                ["low", "medium", "high"],
                default=["low", "medium"],
            )

        with f2:
            inter_types = sorted(df_design["interaction"].dropna().unique())
            default_inter = inter_types if inter_types else []

            allowed_inter = st.multiselect(
                "Interaction type",
                inter_types,
                default=default_inter,
            )

        with f3:
            min_score = st.slider(
                "Min design priority",
                0.0, 1.0, 0.3, step=0.05,
            )


        # ---- Apply filters
        df_design_f = df_design[
            (df_design["risk"].isin(allowed_risk))
            & (df_design["interaction"].isin(allowed_inter))
            & (df_design["design_score"] >= min_score)
        ].copy()

        df_design_f = df_design_f.sort_values(
            "design_score", ascending=False
        )

        # ---- Display
        # Compute ligand charge for suggestions (optional)
        lig_charge_for_suggestions = None
        try:
            lig_charge_for_suggestions = float(charge_num) if not np.isnan(charge_num) else None
        except Exception:
            lig_charge_for_suggestions = None

        # Build mutation suggestion columns
        def _fmt_suggestions(sug_list):
            # sug_list is list[(AA3, rationale)]
            return ", ".join([f"{aa}" for aa, _ in sug_list])

        def _fmt_rationales(sug_list):
            return " | ".join([f"{aa}: {why}" for aa, why in sug_list])

        safe_cols = []
        exp_cols = []
        safe_why = []
        exp_why = []

        for _, row in df_design_f.iterrows():
            safe_sug, exp_sug = suggest_mutations(
                resn=str(row.get("resn", "")),
                interaction=str(row.get("interaction", "other")),
                lig_charge=lig_charge_for_suggestions,
                top_n=4,
            )
            safe_cols.append(_fmt_suggestions(safe_sug))
            exp_cols.append(_fmt_suggestions(exp_sug))
            safe_why.append(_fmt_rationales(safe_sug))
            exp_why.append(_fmt_rationales(exp_sug))
        
        display_cols = [
            "label",
            "resn",
            "interaction",
            "risk",
            "min_dist_A",
            "design_score",
            "notes",
        ]

        df_show = df_design_f[display_cols].rename(columns={
            "label": "Residue",
            "resn": "AA",
            "min_dist_A": "Dist (Å)",
            "design_score": "Design priority",
        }).copy()

        df_show["SAFE mutations"] = safe_cols
        df_show["EXPLORATORY mutations"] = exp_cols

        # (Optional) keep rationales hidden but available for tooltips / future expansion
        df_show["SAFE rationale"] = safe_why
        df_show["EXPLORATORY rationale"] = exp_why

        # Make columns Arrow-safe for AgGrid + consistency
        for c in df_show.columns:
            df_show[c] = df_show[c].apply(lambda x: "" if pd.isna(x) else str(x))

        gb2 = GridOptionsBuilder.from_dataframe(df_show)
        gb2.configure_default_column(sortable=True, filter=True, resizable=True)

        # Hide rationale columns for now
        gb2.configure_column("SAFE rationale", hide=True)
        gb2.configure_column("EXPLORATORY rationale", hide=True)

        AgGrid(
            df_show,
            gridOptions=gb2.build(),
            theme="streamlit",
            height=320,
            fit_columns_on_grid_load=True,
        )

# ---------------------------------------------------------------------
# ROW 3 (FULL WIDTH): Matching pockets
# ---------------------------------------------------------------------
st.subheader("Matching pockets")

df_display = pd.DataFrame({
    "Rank": df_top["rank"].astype(int),
    "Protein ID": df_top["protein_id"].astype(str),
    "Pocket ID": df_top["pocket_id"].astype(str),
    "Biosensor (%)": (df_top["biosensor_score"] * 100).round(1),
    "Ligand compat (%)": (df_top["ligand_compat_score"] * 100).round(1),
    "QuickShape (%)": (df_top["quickshape_score"] * 100).round(1),
    "Volume (Å³)": df_top["volume"].round(1),
    "Hydrophobicity": df_top["hydrophobicity_score"].round(1),

    # hidden but critical
    "pocket_key": df_top["pocket_key"].astype(str),
})

gb = GridOptionsBuilder.from_dataframe(df_display)
gb.configure_selection("single", use_checkbox=True)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
gb.configure_default_column(sortable=True, filter=True, resizable=True)

gb.configure_column("pocket_key", hide=True)  # <- hide internal key

grid_options = gb.build()

grid_response = AgGrid(
    df_display,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    theme="streamlit",
    height=380,
    fit_columns_on_grid_load=True,
)

selected = grid_response.get("selected_rows", None)

new_key = None
if isinstance(selected, pd.DataFrame) and not selected.empty:
    new_key = str(selected.iloc[0]["pocket_key"])
elif isinstance(selected, list) and len(selected) > 0:
    new_key = str(selected[0]["pocket_key"])

if new_key and new_key != st.session_state.get("best_pocket_key"):
    st.session_state["best_pocket_key"] = new_key

    # optional: reset viewer checkbox so it doesn't "stick" across pockets
    st.session_state.pop("show_docked_pose", None)

    st.rerun()

if isinstance(selected, pd.DataFrame) and not selected.empty:
    row = selected.iloc[0]
    rank = int(row["Rank"])
    best_pocket = df_top.iloc[rank - 1]

elif isinstance(selected, list) and len(selected) > 0:
    row = selected[0]
    rank = int(row["Rank"])
    best_pocket = df_top.iloc[rank - 1]
else:
    best_pocket = st.session_state.get("best_pocket", df_top.iloc[0])


st.markdown("---")

# -----------------------------
# Docking & ligand profile (UI)
# -----------------------------

st.markdown('<div class="dock-section">', unsafe_allow_html=True)
st.subheader("Docking & ligand profile")

# Choose values (from PubChem if available, else fall back gracefully)
mw_val = mw_num if not np.isnan(mw_num) else None
logp_val = xlogp_num if not np.isnan(xlogp_num) else None
tpsa_val = tpsa_num if not np.isnan(tpsa_num) else None
chg_val = charge_num if not np.isnan(charge_num) else None

size_txt = describe_mw(mw_num) if mw_val is not None else "size: unknown"
logp_txt = describe_logp(xlogp_num, (profile or {}).get("polarity")) if logp_val is not None or profile else "polarity: unknown"
tpsa_txt = describe_tpsa(tpsa_num) if tpsa_val is not None else "H-bonding: unknown"
chg_txt  = describe_charge(charge_num, (profile or {}).get("charge")) if chg_val is not None or profile else "charge: unknown"

# Colors (you already have helpers)
mw_color   = color_for_mw(mw_num)
logp_color = color_for_logp(xlogp_num)
tpsa_color = color_for_tpsa(tpsa_num)
chg_color  = color_for_charge(charge_num if chg_val is not None else (profile or {}).get("charge", "neutral"))

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown('<div class="dock-kpi-title">Molecular weight (MW)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dock-kpi-value">{mw_val:.1f} Da</div>' if mw_val is not None else '<div class="dock-kpi-value">–</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="dock-pill" style="background:{mw_color};">{size_txt}</span>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="dock-kpi-title">Partition coefficient (logP)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dock-kpi-value">{logp_val:.2f}</div>' if logp_val is not None else '<div class="dock-kpi-value">–</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="dock-pill" style="background:{logp_color};">{logp_txt}</span>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="dock-kpi-title">Topological polar surface area (TPSA)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dock-kpi-value">{tpsa_val:.1f} Å²</div>' if tpsa_val is not None else '<div class="dock-kpi-value">–</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="dock-pill" style="background:{tpsa_color};">{tpsa_txt}</span>', unsafe_allow_html=True)

with k4:
    st.markdown('<div class="dock-kpi-title">Formal charge (pH ~7)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dock-kpi-value">{int(chg_val) if chg_val is not None else 0}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="dock-pill" style="background:{chg_color};">{chg_txt}</span>', unsafe_allow_html=True)

st.markdown("")  # small spacer

# Bottom row: ligand 3D (left) + docking package (right)
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Ligand 3D structure")

    if analyte_for_docking:
        ligand_sdf_path = DOCKING_ROOT / "ligands" / f"{analyte_folder_name}.sdf"
        sdf_text: str | None = None

        if ligand_sdf_path.exists():
            try:
                sdf_text = ligand_sdf_path.read_text()
            except Exception:
                sdf_text = None

        if sdf_text is None:
            sdf_text = fetch_pubchem_sdf_by_name(analyte_for_docking)

        if sdf_text:
            try:
                lig_view = py3Dmol.view(width=520, height=320)
                lig_view.addModel(sdf_text, "sdf")
                lig_view.setStyle({"stick": {"radius": 0.18}})
                lig_view.addStyle({"sphere": {"scale": 0.25}})
                lig_view.zoomTo()
                lig_view.setBackgroundColor("0xFFFFFF")
                components.html(lig_view._make_html(), height=340, scrolling=False)
            except Exception as e:
                st.warning(f"Could not render ligand 3D view: {e}")
        else:
            st.info("No 3D structure available for this analyte yet.")
    else:
        st.info("Type an analyte above to render its 3D structure.")

with right:
    st.subheader("Docking package")
    st.markdown(
        """
        <div class="dock-subtitle">
          Generate docking boxes, ligand 3D file and a summary CSV, packaged as a single ZIP under
          <span class="dock-code">docking/</span>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analyte_for_docking:
        st.info("Type an analyte above to enable docking export.")
    else:
        if st.button("Download docking package (.zip)", use_container_width=True):
            with st.spinner("Running docking setup and building package…"):
                zip_bytes, log_text = build_docking_package_zip(analyte_for_docking)

            if zip_bytes is None:
                st.error("Docking setup failed — see log below.")
                if log_text:
                    st.code(log_text, language="bash")
            else:
                file_name = f"{analyte_folder_name}_docking_package.zip"
                b64_zip = base64.b64encode(zip_bytes).decode("utf-8")

                html = f"""
                <html><body>
                  <a id="auto_dl" href="data:application/zip;base64,{b64_zip}" download="{file_name}"></a>
                  <script>document.getElementById('auto_dl').click();</script>
                </body></html>
                """
                st.success("Docking package ready — your download should start automatically.")
                components.html(html, height=0, width=0)

st.markdown("</div>", unsafe_allow_html=True)
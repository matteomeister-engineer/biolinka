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

from pathlib import Path
from urllib.parse import quote
from pathlib import Path



# ---------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------


def to_streamlit_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix PyArrow object-type column errors for st.dataframe().
    Convert all object columns to plain strings.
    """
    if df.empty:
        return df

    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(lambda x: "" if pd.isna(x) else str(x))

    # Force a plain NumPy object array backing and a clean RangeIndex
    arr = df2.to_numpy(dtype=object, copy=True)
    return pd.DataFrame(arr, columns=df2.columns).reset_index(drop=True)


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
STRUCT_DIR = BASE_DIR / "structures"
DOCKING_ROOT = BASE_DIR / "docking"

POCKET_CSV = RESULTS_DIR / "pockets_fpocket.csv"


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
    # add more entries here as your panel grows
}


# ---------------------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Biosensor Pocket Explorer",
    layout="wide",
)


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_pockets() -> pd.DataFrame:
    df = pd.read_csv(POCKET_CSV)

    # Basic biosensor-oriented heuristic score
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
            w_drug * drug_term +
            w_vol * vol_term +
            w_hyd * hyd_term +
            w_pol * pol_term +
            w_alpha * alpha_term +
            w_flex * flex_term
        )
        return max(0.0, min(1.0, score))

    df["biosensor_score"] = df.apply(biosensor_score, axis=1)

    # Attach local metadata
    df["protein_name"] = df["protein_id"].map(
        lambda pid: UNIPROT_INFO.get(pid, {}).get("protein_name", "Unknown protein")
    )
    df["source_organism"] = df["protein_id"].map(
        lambda pid: UNIPROT_INFO.get(pid, {}).get("organism", "unknown")
    )
    df["biological_function"] = df["protein_id"].map(
        lambda pid: UNIPROT_INFO.get(pid, {}).get(
            "function", "Function not annotated in this local panel yet."
        )
    )

    return df


df_all = load_pockets()


# ---------------------------------------------------------------------
# PUBCHEM LIGAND FETCH
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_ligand_from_pubchem(name: str):
    """
    Fetch basic physicochemical properties for a ligand name from PubChem.

    Returns dict with:
      - name, cid, mw, xlogp, tpsa, hbd, hba, charge, rotb
      - size_class, polarity, charge_class
    or None.
    """
    name = (name or "").strip()
    if not name:
        return None

    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # name -> CID
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

    # CID -> properties
    props = "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,Charge,TPSA,RotatableBondCount"
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

    # size class
    if np.isnan(mw):
        size_class = "unknown"
    elif mw < 180:
        size_class = "small"
    elif mw < 400:
        size_class = "medium"
    else:
        size_class = "large"

    # polarity from tpsa & logP
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

    # ligand size class
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

    # ligand polarity class
    if ligand_info is not None:
        lig_pol = ligand_info.get("polarity", "mixed")
    elif profile is not None:
        lig_pol = profile.get("polarity", "mixed")
    else:
        lig_pol = "mixed"

    # ligand charge class
    if ligand_info is not None:
        lig_charge_class = ligand_info.get("charge_class", "neutral")
    elif profile is not None:
        lig_charge_class = profile.get("charge", "neutral")
    else:
        lig_charge_class = "neutral"

    # ligand polarity strength from TPSA
    if not np.isnan(tpsa_num):
        if tpsa_num < 60:
            lig_pol_strength = "low"
        elif tpsa_num < 90:
            lig_pol_strength = "medium"
        else:
            lig_pol_strength = "high"
    else:
        lig_pol_strength = "unknown"

    # pocket descriptors
    vol = row.get("volume", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    pol_score = row.get("polarity_score", np.nan)
    chg_score = row.get("charge_score", 0.0)
    drug = row.get("druggability_score", 0.0)

    # pocket size class
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

    # pocket environment (hydrophobicity)
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

    # pocket charge class
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

    # polarity / H-bond matching
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

    # druggability bonus
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
    """
    Returns (bg_color, text_color) for soft pastel pills.
    Score in [0, 1].
    """
    if score is None or np.isnan(score):
        return "#E5E7EB", "#374151"

    s = max(0.0, min(1.0, float(score)))

    if s < 0.2:
        return "#FEE2E2", "#991B1B"   # very low
    elif s < 0.4:
        return "#FFEDD5", "#9A3412"   # low
    elif s < 0.6:
        return "#FEF9C3", "#854D0E"   # medium
    elif s < 0.8:
        return "#DCFCE7", "#166534"   # good
    else:
        return "#BBF7D0", "#14532D"   # excellent


# ---------------------------------------------------------------------
# POCKET INTERPRETATION
# ---------------------------------------------------------------------

def interpret_pocket(row: pd.Series) -> str:
    prot = row.get("protein_id", "–")
    pname = row.get("protein_name", "Unknown protein")
    org = row.get("source_organism", "unknown organism")
    vol = row.get("volume", np.nan)
    drug = row.get("druggability_score", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    pol = row.get("polarity_score", np.nan)
    chg = row.get("charge_score", np.nan)
    bios = row.get("biosensor_score", np.nan)
    fun = row.get("biological_function", "")

    lines = []
    lines.append(f"**Protein** `{prot}` — *{pname}* ({org}).")
    if fun:
        lines.append("")
        lines.append(f"**Biological role:** {fun}")

    lines.append("")
    lines.append("**Pocket biophysics:**")

    if not np.isnan(vol):
        if vol < 250:
            lines.append(f"• Volume ≈ {vol:.0f} Å³ — compact pocket suited for small ligands.")
        elif vol < 600:
            lines.append(f"• Volume ≈ {vol:.0f} Å³ — medium-sized pocket, compatible with many metabolites or drugs.")
        else:
            lines.append(f"• Volume ≈ {vol:.0f} Å³ — large cavity, can host bulky ligands or multi-contact binding.")

    if not np.isnan(hyd):
        if hyd < 15:
            lines.append("• Hydrophobicity is **low** → pocket is mostly polar / H-bonding.")
        elif hyd < 40:
            lines.append("• Hydrophobicity is **moderate** → mixed polar/non-polar environment.")
        else:
            lines.append("• Hydrophobicity is **high** → pocket favors hydrophobic or amphipathic ligands.")

    if not np.isnan(pol):
        if pol < 5:
            lines.append("• Polarity score is **low** → few strong polar contacts expected.")
        elif pol < 10:
            lines.append("• Polarity score is **moderate** → can support several polar interactions.")
        else:
            lines.append("• Polarity score is **high** → rich in polar groups, strong H-bond network possible.")

    if not np.isnan(chg):
        if chg > 1:
            lines.append("• Pocket is **overall positively charged** → favorable to anionic / acidic ligands.")
        elif chg < -1:
            lines.append("• Pocket is **overall negatively charged** → favorable to cationic / basic ligands.")
        else:
            lines.append("• Pocket charge is **near neutral** → not strongly biased toward charged ligands.")

    if not np.isnan(drug):
        if drug > 0.7:
            lines.append("• Druggability score is **high** → classical small-molecule drug–like site.")
        elif drug > 0.3:
            lines.append("• Druggability score is **moderate** → may still be usable with optimization.")
        else:
            lines.append("• Druggability score is **low** → more challenging scaffold, but not impossible.")

    if not np.isnan(bios):
        bios_pct = (bios * 100.0)
        lines.append("")
        lines.append(
            f"**Biosensor heuristic:** {bios_pct:.1f}% — combined view of volume, hydrophobicity, polarity, flexibility and fpocket druggability."
        )
        lines.append(
            "This pocket could be engineered as a sensing module by introducing mutations in key lining residues,"
        )
        lines.append(
            "and then coupling conformational change to a reporting domain (fluorescent protein, transcription factor, etc.)."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------------------

st.title("🧬 Biosensor Pocket Explorer")

st.write(
    "Explore fpocket-derived pockets from protein structures. Filter by ligand-like properties "
    "(size, polarity, charge, druggability) to identify pockets that could be engineered into "
    "biosensors for cell-based and organ-on-chip systems."
)


# ---------------------------------------------------------------------
# ANALYTE SELECTION + LIGAND CARD
# ---------------------------------------------------------------------

st.markdown("### What do you want to sense?")

# Ensure analyte state exists
if "analyte_query" not in st.session_state:
    st.session_state["analyte_query"] = ""

# Main search bar (actual input)
analyte_raw = st.text_input(
    "Analyte name (e.g. glucose, dopamine, cortisol, urea, Na⁺, K⁺…)",
    key="analyte_query",
)
aq = analyte_raw.lower().strip()

# ------------------------------------------------------------------
# ANALYTE PROFILES
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# COLOR THEMES (same as category pills you showed)
# ------------------------------------------------------------------
CATEGORY_COLORS = {
    "met": ("#E8F0FF", "#1E40AF"),   # metabolites (light blue / dark blue text)
    "nt":  ("#FDECF3", "#9D174D"),   # neurotransmitters (light pink / dark rose)
    "st":  ("#FEF7D1", "#B45309"),   # steroids/hormones (light yellow / brown)
    "dr":  ("#F3F4F6", "#374151"),   # drugs (neutral grey)
    "ion": ("#EEE6FF", "#5B21B6"),   # ions (light lavender / purple text)
}

# Convert label for display
def preset_label(key: str) -> str:
    k = key.lower()
    if k == "na+":
        return "Na⁺ (sodium)"
    if k == "k+":
        return "K⁺ (potassium)"
    return key.capitalize()

# ------------------------------------------------------------------
# SMALL COLORED PILLS — ONE ROW ONLY
# ------------------------------------------------------------------

st.caption("Common analytes:")

keys = list(ANALYTE_PROFILES.keys())

# Single row with max 15 pills
cols = st.columns(min(len(keys), 15))

for col, key in zip(cols, keys):
    cat = ANALYTE_PROFILES[key]["cat"]
    bg, fg = CATEGORY_COLORS.get(cat, ("#F3F4F6", "#111827"))  # fallback grey

    pill = (
        f"background:{bg};"
        f"color:{fg};"
        "padding:4px 12px;"
        "border-radius:999px;"
        "font-size:0.78rem;"
        "font-weight:500;"
        "display:inline-block;"
        "margin:3px;"
        "white-space:nowrap;"
    )

    with col:
        st.markdown(
            f"<span style='{pill}'>{preset_label(key)}</span>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------
# ONE-BUTTON DOCKING PACKAGE (prepare + download ZIP)
# ---------------------------------------------------------------------

raw_analyte = st.session_state.get("analyte_query", "").strip()
analyte_for_docking = raw_analyte
analyte_folder_name = analyte_for_docking.replace(" ", "_") if analyte_for_docking else ""

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

    # 1) run docking setup script
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

    # 2) locate output files
    docking_subdir = DOCKING_ROOT / folder_name
    targets_csv = docking_subdir / "docking_targets.csv"
    ligand_local = DOCKING_ROOT / "ligands" / f"{folder_name}.sdf"
    box_files = list(docking_subdir.glob("*_box.txt"))

    if not docking_subdir.exists() or not targets_csv.exists():
        return None, f"Docking directory or docking_targets.csv missing in {docking_subdir}"

    # 3) build ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        if targets_csv.exists():
            zf.write(
                targets_csv,
                arcname=f"{folder_name}/docking_targets.csv"
            )

        if ligand_local.exists():
            zf.write(
                ligand_local,
                arcname=f"{folder_name}/ligand.sdf"
            )

        for box_file in box_files:
            zf.write(
                box_file,
                arcname=f"{folder_name}/{box_file.name}"
            )

        log_name = f"{folder_name}/prepare_docking_setup.log"
        zf.writestr(log_name, full_log)

    return zip_buffer.getvalue(), full_log


st.markdown("")
st.subheader("Docking package")

if not analyte_for_docking:
    st.info("Type or select an analyte (e.g. glucose, dopamine, cortisol) to enable docking.")
else:
    col_explain, col_btn = st.columns([3, 1])
    with col_explain:
        st.markdown(
            "<span style='font-size:0.9rem; color:#6B7280;'>"
            "Generate docking boxes, ligand 3D file and a summary CSV, "
            "packaged as a single ZIP under <code>docking/</code>."
            "</span>",
            unsafe_allow_html=True,
        )

    with col_btn:
        if st.button("Prepare & build docking ZIP", use_container_width=True):
            with st.spinner("Running docking setup and building package…"):
                zip_bytes, log_text = build_docking_package_zip(analyte_for_docking)
            st.session_state["dock_zip_bytes"] = zip_bytes
            st.session_state["dock_zip_log"] = log_text

    zip_bytes = st.session_state["dock_zip_bytes"]
    log_text = st.session_state["dock_zip_log"]

    if zip_bytes is None:
        if log_text:
            st.error("Docking setup failed — see log below.")
            st.code(log_text, language="bash")
    else:
        st.download_button(
            label="Download docking package (.zip)",
            data=zip_bytes,
            file_name=f"{analyte_folder_name}_docking_package.zip",
            mime="application/zip",
            use_container_width=True,
        )


# ------------------------------------------------------------------
# LIGAND 3D VIEWER
# 1) Prefer local docking/ligands/<analyte>.sdf if it exists
# 2) Otherwise fetch 3D SDF from PubChem
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_pubchem_sdf_by_name(name: str) -> str | None:
    """
    Fetch a 3D (or 2D fallback) SDF for a small molecule from PubChem by name.
    Returns the SDF text, or None if it fails.
    """
    name = name.strip()
    if not name:
        return None

    try:
        cid_url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{requests.utils.quote(name)}/cids/JSON"
        )
        cid_resp = requests.get(cid_url, timeout=15)
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
        st.markdown("")
        st.subheader("Ligand 3D structure")

        try:
            lig_view = py3Dmol.view(width=450, height=260)
            lig_view.addModel(sdf_text, "sdf")
            lig_view.setStyle({"stick": {"radius": 0.18}})
            lig_view.addStyle({"sphere": {"scale": 0.25}})
            lig_view.zoomTo()
            lig_view.setBackgroundColor("0xFFFFFF")
            lig_html = lig_view._make_html()
            components.html(lig_html, height=280, scrolling=False)
        except Exception as e:
            st.warning(f"Could not render ligand 3D view: {e}")


# ------------------------------------------------------------------
# LIGAND CHEMICAL PROFILE (PubChem + heuristic fallback)
# ------------------------------------------------------------------

# Get the current analyte text from the input
analyte_raw = st.session_state.get("analyte_query", "")
aq = analyte_raw.strip().lower()

# Safe defaults so we never hit NameError
ligand_info = None
profile = None

mw_val = logp_val = tpsa_val = charge_val = None
mw_num = xlogp_num = tpsa_num = charge_num = np.nan

if aq:
    # 1) Try PubChem for full physchem profile
    ligand_info = fetch_ligand_from_pubchem(aq)

    if ligand_info:
        mw_num = ligand_info["mw"]
        xlogp_num = ligand_info["xlogp"]
        tpsa_num = ligand_info["tpsa"]
        charge_num = ligand_info["charge"]

        mw_val = f"{mw_num:.1f} Da" if not np.isnan(mw_num) else "n/a"
        logp_val = "n/a" if np.isnan(xlogp_num) else f"{xlogp_num:.2f}"
        tpsa_val = "n/a" if np.isnan(tpsa_num) else f"{tpsa_num:.1f} Å²"
        charge_val = str(charge_num)

        # Optional: remember interpreted name
        st.session_state["interpreted_name"] = ligand_info["name"]
    else:
        # 2) If PubChem fails, fall back to your ANALYTE_PROFILES heuristics
        for key, prof in ANALYTE_PROFILES.items():
            if key in aq:
                profile = prof
                break

        if profile:
            mw_val = "n/a"
            logp_val = "n/a"
            tpsa_val = "n/a"
            charge_val = profile["charge"]
else:
    ligand_info = None
    profile = None


# ------------------------------------------------------------------
# LIGAND PROPERTY HINT PILLS
# ---------------------------------------------------------------------

if ligand_info:
    mw_hint = describe_mw(mw_num)
    logp_hint = describe_logp(xlogp_num, fallback_polarity=ligand_info["polarity"])
    tpsa_hint = describe_tpsa(tpsa_num)
    charge_hint = describe_charge(charge_num)
elif profile:
    mw_hint = f"~{profile['size']} ligand"
    logp_hint = describe_logp(np.nan, fallback_polarity=profile["polarity"])
    charge_hint = describe_charge(profile["charge"])
    tpsa_hint = "polarity inferred from profile"
else:
    mw_hint = logp_hint = tpsa_hint = charge_hint = None

if mw_val is not None:
    pill_base_style = (
        "display:inline-block;"
        "margin-top:2px;"
        "padding:2px 8px;"
        "border-radius:999px;"
        "font-size:0.72rem;"
        "white-space:nowrap;"
    )

    cols = st.columns(4)

    with cols[0]:
        st.caption("Molecular weight (MW)")
        st.write(mw_val)
        if mw_hint:
            color = color_for_mw(mw_num)
            st.markdown(
                f"<span style='{pill_base_style} background:{color}; color:white;'>{mw_hint}</span>",
                unsafe_allow_html=True,
            )

    with cols[1]:
        st.caption("Partition coefficient (logP)")
        st.write(logp_val)
        if logp_hint:
            color = color_for_logp(xlogp_num)
            st.markdown(
                f"<span style='{pill_base_style} background:{color}; color:white;'>{logp_hint}</span>",
                unsafe_allow_html=True,
            )

    with cols[2]:
        st.caption("Topological polar surface area (TPSA)")
        st.write(tpsa_val)
        if tpsa_hint:
            color = color_for_tpsa(tpsa_num)
            st.markdown(
                f"<span style='{pill_base_style} background:{color}; color:white;'>{tpsa_hint}</span>",
                unsafe_allow_html=True,
            )

    with cols[3]:
        st.caption("Formal charge (pH ~7)")
        st.write(charge_val)
        if charge_hint:
            color = color_for_charge(charge_num if ligand_info else charge_val)
            st.markdown(
                f"<span style='{pill_base_style} background:{color}; color:white;'>{charge_hint}</span>",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ---------------------------------------------------------------------
# FILTERS FOR POCKETS
# ---------------------------------------------------------------------

with st.expander("Pocket filters", expanded=True):
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
    (df_all["volume"].between(vol_min, vol_max)) &
    (df_all["druggability_score"] >= drug_min) &
    (df_all["hydrophobicity_score"].between(hyd_min, hyd_max)) &
    (df_all["biosensor_score"] >= bios_min)
].copy()

# ---------------------------------------------------------------------
# LIGAND–POCKET COMPATIBILITY COLUMN
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

# ---------------------------------------------------------------------
# METRICS STRIP
# ---------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Proteins (after filters)")
    st.markdown(
        f"<h2 style='margin: 0;'>{df_filtered['protein_id'].nunique()}</h2>",
        unsafe_allow_html=True,
    )

with col2:
    st.caption("Candidate pockets")
    st.markdown(
        f"<h2 style='margin: 0;'>{len(df_filtered)}</h2>",
        unsafe_allow_html=True,
    )

with col3:
    if not df_filtered.empty:
        value = df_filtered["biosensor_score"].dropna().mean() * 100
        mean_bio = f"{value:.2f}".rstrip("0").rstrip(".") + "%"
    else:
        mean_bio = "0%"
    st.caption("Mean biosensor suitability (heuristic)")
    st.markdown(
        f"<h2 style='margin: 0;'>{mean_bio}</h2>",
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
# PICK BEST POCKET (USE LIGAND COMPAT FIRST IF AVAILABLE)
# ---------------------------------------------------------------------
if (
    "ligand_compat_score" in df_filtered.columns
    and df_filtered["ligand_compat_score"].notna().any()
):
    best_pocket = df_filtered.sort_values(
        ["ligand_compat_score", "biosensor_score"],
        ascending=False,
    ).iloc[0]
else:
    best_pocket = df_filtered.sort_values(
        "biosensor_score",
        ascending=False,
    ).iloc[0]

# ---------------------------------------------------------------------
# BEST MATCHING POCKET + 3D VIEWER
# ---------------------------------------------------------------------
best_col, viewer_col = st.columns([1, 1])

with best_col:
    st.subheader("Best Matching Pocket")

    # --- Extract values ---
    prot = best_pocket.get("protein_id", "–")
    pocket_id = best_pocket.get("pocket_id", "–")
    bios = best_pocket.get("biosensor_score", np.nan)
    drugg = best_pocket.get("druggability_score", np.nan)
    vol = best_pocket.get("volume", np.nan)
    hyd = best_pocket.get("hydrophobicity_score", np.nan)
    org = best_pocket.get("source_organism", "unknown")
    pname = best_pocket.get("protein_name", "Unknown protein")
    biofun = best_pocket.get("biological_function", "")

    compat = best_pocket.get("ligand_compat_score", np.nan)
    qshape = best_pocket.get("quickshape_score", np.nan)

    # --- Formatting numeric values ---
    bios_str = "–" if np.isnan(bios) else f"{(bios * 100):.2f}".rstrip("0").rstrip(".") + "%"
    drugg_str = "–" if np.isnan(drugg) else f"{drugg:.3f}"
    vol_str = "–" if np.isnan(vol) else f"{vol:.1f} Å³"
    hyd_str = "–" if np.isnan(hyd) else f"{hyd:.1f}"

    compat_str = (
        "–" if np.isnan(compat)
        else f"{(compat * 100):.2f}".rstrip("0").rstrip(".") + "%"
    )
    qshape_str = (
        "–" if np.isnan(qshape)
        else f"{(qshape * 100):.2f}".rstrip("0").rstrip(".") + "%"
    )

    org_display = org

    # --- External links (UniProt + AlphaFold) ---
    af_entry_url = f"https://alphafold.ebi.ac.uk/entry/AF-{prot}-F1"
    uniprot_url = f"https://www.uniprot.org/uniprotkb/{prot}/entry"

    af_link_html = (
        f"<a href='{af_entry_url}' target='_blank' "
        "style='color:#2563EB; text-decoration:none; font-weight:500;'>"
        "Open in AlphaFold →"
        "</a>"
    )
    uniprot_link_html = (
        f"<a href='{uniprot_url}' target='_blank' "
        "style='color:#2563EB; text-decoration:none; font-weight:500; margin-right:12px;'>"
        "Open in UniProt →"
        "</a>"
    )

    # --- Soft-colored pill ONLY for ligand compatibility ---
    bg, fg = soft_color_for_compat(compat)
    compat_pill_html = (
        f"<span style='"
        "display:inline-block;"
        "padding:3px 10px;"
        "border-radius:999px;"
        f"background:{bg};"
        f"color:{fg};"
        "font-size:0.82rem;"
        "font-weight:600;"
        "margin-top:3px;"
        "white-space:nowrap;"
        "'>"
        f"{compat_str}"
        "</span>"
        if compat_str != "–" else "–"
    )

    # ------------------------------------------------------------------
    # INFO GRID — left = protein info, right = pocket + scores
    # ------------------------------------------------------------------
    pocket_info_html = (
        "<div style='"
        "display:grid;"
        "grid-template-columns:1fr 1fr;"
        "gap:14px 22px;"
        "font-size:0.95rem;"
        "line-height:1.25;"
        "padding-top:6px;"
        "'>"

        # Row 1: Protein ID / Biosensor score (plain text)
        f"<div><strong>Protein ID</strong><br>{prot}<br>{uniprot_link_html}<br>{af_link_html}</div>"
        f"<div><strong>Biosensor score</strong><br>{bios_str}</div>"

        # Row 2: Protein name / Ligand compatibility (pill)
        f"<div><strong>Protein name</strong><br>{pname}</div>"
        f"<div><strong>Ligand compatibility</strong><br>{compat_pill_html}</div>"

        # Row 3: Organism / QuickShape
        f"<div><strong>Source organism</strong><br>{org_display}</div>"
        f"<div><strong>QuickShape (geom. fit)</strong><br>{qshape_str}</div>"

        # Row 4: Pocket location / Hydrophobicity
        f"<div><strong>Pocket location</strong><br>{pocket_id}</div>"
        f"<div><strong>Hydrophobicity</strong><br>{hyd_str}</div>"

        # Row 5: empty / Volume
        f"<div></div>"
        f"<div><strong>Volume</strong><br>{vol_str}</div>"

        "</div>"
    )

    st.markdown(pocket_info_html, unsafe_allow_html=True)
with viewer_col:
    st.subheader("3D viewer")

    # --- Figure out which protein/pocket we are showing ---
    prot_id = str(best_pocket["protein_id"])
    pocket_num = str(best_pocket["pocket_id"])  # keep as string for filenames

    protein_pdb = STRUCT_DIR / f"{prot_id}.pdb"
    pocket_pdb = STRUCT_DIR / f"{prot_id}_out" / "pockets" / f"pocket{pocket_num}_atm.pdb"

    # --- Figure out which analyte (folder) to use for docking results ---
    raw_analyte = st.session_state.get("analyte_query", "").strip()
    interpreted_name = st.session_state.get("interpreted_name", None)
    analyte_for_docking = (interpreted_name or raw_analyte).strip()

    docked_pose_pdbqt = None
    vina_score_for_pocket = None
    docking_info_text = None

    if analyte_for_docking:
        safe_name = analyte_for_docking.replace(" ", "_")
        analyte_dir = BASE_DIR / "docking" / safe_name
        docked_dir = analyte_dir / "docked"
        docked_pose_pdbqt = docked_dir / f"{prot_id}_pocket{pocket_num}_{safe_name}_out.pdbqt"
        results_csv = analyte_dir / "docking_results.csv"

        # Try to read Vina score from docking_results.csv (if present)
        if results_csv.exists():
            try:
                df_res = pd.read_csv(results_csv)

                # Be robust: compare as strings
                pid_str = str(prot_id)
                pocket_str = str(pocket_num)

                mask = (
                    df_res["protein_id"].astype(str).eq(pid_str)
                    & df_res["pocket_id"].astype(str).eq(pocket_str)
                )
                if mask.any() and "vina_score" in df_res.columns:
                    vina_val = df_res.loc[mask, "vina_score"].iloc[0]
                    try:
                        vina_score_for_pocket = float(vina_val)
                    except Exception:
                        vina_score_for_pocket = None
            except Exception as e:
                st.warning(f"Could not read docking_results.csv: {e}")

    # ----------------- MAIN VIEWER RENDERING -----------------
    if not protein_pdb.exists():
        st.warning(f"Protein structure not found: {protein_pdb}")
    else:
        view = py3Dmol.view(width=600, height=320)

        # --- Protein cartoon ---
        with open(protein_pdb, "r") as f:
            view.addModel(f.read(), "pdb")
        view.setStyle({"cartoon": {"opacity": 0.7, "color": "0xAAAAAA"}})

        # --- Pocket atoms (if available) ---
        if pocket_pdb.exists():
            with open(pocket_pdb, "r") as f:
                view.addModel(f.read(), "pdb")  # model index 1
            view.setStyle(
                {"model": 1},
                {"stick": {"radius": 0.25}, "sphere": {"radius": 0.75}},
            )
            view.zoomTo({"model": 1})
        else:
            view.zoomTo()

        # --- Docked ligand pose from Vina (if available) ---
        if docked_pose_pdbqt is not None and docked_pose_pdbqt.exists():
            try:
                with open(docked_pose_pdbqt, "r") as f:
                    view.addModel(f.read(), "pdbqt")  # model index 2
                view.setStyle(
                    {"model": 2},
                    {"stick": {"radius": 0.2, "colorscheme": "atom"}}
                )
                # Zoom to show ligand clearly
                view.zoomTo({"model": 2})

                if vina_score_for_pocket is not None:
                    docking_info_text = (
                        f"Showing Vina docked pose from:<br>"
                        f"<code>{docked_pose_pdbqt}</code><br>"
                        f"Best binding energy: <strong>{vina_score_for_pocket:.2f} kcal/mol</strong>"
                    )
                else:
                    docking_info_text = (
                        f"Showing Vina docked pose from:<br>"
                        f"<code>{docked_pose_pdbqt}</code>"
                    )
            except Exception as e:
                st.warning(f"Could not load docked pose: {e}")
        elif analyte_for_docking:
            docking_info_text = (
                "No docked pose found yet for this pocket.<br>"
                "Make sure you have run "
                f"<code>python scripts/run_vina_batch.py --analyte {analyte_for_docking}</code>."
            )

        # --- Zoom limits & background ---
        view.setZoomLimits(10, 300)
        view.setBackgroundColor("0xFFFFFF")

        html = view._make_html()
        components.html(html, height=340, scrolling=False)

        if docking_info_text:
            st.markdown(
                f"<p style='font-size:0.8rem; color:#6B7280; margin-top:4px;'>{docking_info_text}</p>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# INTERPRETATION SECTION
# ---------------------------------------------------------------------

st.subheader("Biological context & pocket interpretation")

interp_text = interpret_pocket(best_pocket)
interp_html = interp_text.replace("\n", "<br>")

st.markdown(
    f"""
    <div style="
        max-height: 220px;
        overflow-y: auto;
        font-size: 0.92rem;
        line-height: 1.45;
        padding-right: 0.5rem;
    ">
    {interp_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# MATCHING POCKETS TABLE (disabled for now)
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Matching pockets")

st.info(
    "The interactive pocket table will be added next.\n"
    "It will let you click any row and instantly load that pocket "
    "into the Best Matching Pocket viewer above."
)

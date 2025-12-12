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

from pathlib import Path
from urllib.parse import quote
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode



# ---------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------


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

st.set_page_config(
    page_title="BioLinka",
    layout="wide",
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
    fun = row.get("biological_function", "")

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
    <style>
    /* Hide anchor/link icons next to headers and metrics */
    a[data-testid="stHeaderLink"],
    a[data-testid="stMetricDelta"],
    a[data-testid="stMetricValue"],
    a[data-testid="stMetricLabel"],
    svg[data-testid="stHeaderLinkIcon"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
            margin-top: -3.25rem !important;
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
# RANK POCKETS ONCE (USED BY VIEWER + TABLE)
# ---------------------------------------------------------------------

if (
    "ligand_compat_score" in df_filtered.columns
    and df_filtered["ligand_compat_score"].notna().any()
):
    df_ranked = df_filtered.sort_values(
        ["ligand_compat_score", "biosensor_score"], ascending=False
    )
else:
    df_ranked = df_filtered.sort_values("biosensor_score", ascending=False)

df_top = df_ranked.head(20).reset_index(drop=True)
df_top["rank"] = df_top.index + 1

# Choose current best pocket (session_state → default to top-ranked)
best_pocket = st.session_state.get("best_pocket", df_top.iloc[0])

# ---------------------------------------------------------------------
# BEST MATCHING POCKET + 3D VIEWER (ABOVE TABLE)
# ---------------------------------------------------------------------

best_col, viewer_col = st.columns([1, 1])

with best_col:
    st.subheader("Best Matching Pocket")

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

    pocket_info_html = (
        "<div style='"
        "display:grid;"
        "grid-template-columns:1fr 1fr;"
        "gap:14px 22px;"
        "font-size:0.95rem;"
        "line-height:1.25;"
        "padding-top:6px;"
        "'>"
        f"<div><strong>Protein ID</strong><br>{prot}<br>{uniprot_link_html}<br>{af_link_html}</div>"
        f"<div><strong>Biosensor score</strong><br>{bios_str}</div>"
        f"<div><strong>Protein name</strong><br>{pname}</div>"
        f"<div><strong>Ligand compatibility</strong><br>{compat_pill_html}</div>"
        f"<div><strong>Source organism</strong><br>{org}</div>"
        f"<div><strong>QuickShape (geom. fit)</strong><br>{qshape_str}</div>"
        f"<div><strong>Pocket location</strong><br>{pocket_id}</div>"
        f"<div><strong>Hydrophobicity</strong><br>{hyd_str}</div>"
        f"<div></div>"
        f"<div><strong>Volume</strong><br>{vol_str}</div>"
        "</div>"
    )

    st.markdown(pocket_info_html, unsafe_allow_html=True)

with viewer_col:
    st.subheader("3D viewer")

    vcol, lcol = st.columns([5, 1])

    with vcol:
        prot_id = str(best_pocket["protein_id"])
        pocket_num = str(best_pocket["pocket_id"])
        protein_pdb = STRUCT_DIR / f"{prot_id}.pdb"
        pocket_pdb = STRUCT_DIR / f"{prot_id}_out" / "pockets" / f"pocket{pocket_num}_atm.pdb"

        interpreted_name = st.session_state.get("interpreted_name", None)
        analyte_for_docking = (interpreted_name or analyte_for_docking or "").strip()
        analyte_folder_name = (
            analyte_for_docking.replace(" ", "_") if analyte_for_docking else None
        )

        docked_pose = None
        if analyte_folder_name:
            docked_dir = BASE_DIR / "docking" / analyte_folder_name / "docked"
            docked_pose = (
                docked_dir
                / f"{prot_id}_pocket{pocket_num}_{analyte_folder_name}_out.pdbqt"
            )

        show_ligand_pose = False
        if docked_pose is not None and docked_pose.exists():
            show_ligand_pose = st.checkbox(
                "Show docked ligand pose (Vina)",
                value=True,
                key="show_docked_pose",
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

                if (
                    show_ligand_pose
                    and docked_pose is not None
                    and docked_pose.exists()
                ):
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
                    if pocket_pdb.exists():
                        view.zoomTo({"model": 1})
                    else:
                        view.zoomTo()

                view.setZoomLimits(10, 300)
                view.setBackgroundColor("0xFFFFFF")

                html = view._make_html()
                components.html(html, height=340, scrolling=False)

            except Exception as e:
                st.warning(f"Could not render 3D view: {e}")

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
# ---------------------------------------------------------------------
# INTERPRETATION (LEFT) + TABLE (RIGHT)
# ---------------------------------------------------------------------

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Biological context & pocket interpretation")

    interp_html = interpret_pocket(best_pocket)
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

with right_col:
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
    })

    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_selection("single", use_checkbox=True)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
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

# Persist selection for next rerun (used by viewer above)
st.session_state["best_pocket"] = best_pocket

st.markdown("---")
# ---------------------------------------------------------------------
# DOCKING PACKAGE + LIGAND 3D (below best pocket)
# ---------------------------------------------------------------------


dock_col, lig3d_col = st.columns([1.3, 1])

with dock_col:
    st.subheader("Docking package")

    if not analyte_for_docking:
        st.info(
            "Type or select an analyte (e.g. glucose, dopamine, cortisol) to enable docking."
        )
    else:
        st.markdown(
            "<span style='font-size:0.9rem; color:#6B7280;'>"
            "Generate docking boxes, ligand 3D file and a summary CSV, "
            "packaged as a single ZIP under <code>docking/</code>."
            "</span>",
            unsafe_allow_html=True,
        )

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
                <html>
                  <body>
                    <a id="auto_dl" href="data:application/zip;base64,{b64_zip}"
                       download="{file_name}"></a>
                    <script>
                      document.getElementById('auto_dl').click();
                    </script>
                  </body>
                </html>
                """

                st.success(
                    "Docking package ready — your download should start automatically."
                )
                components.html(html, height=0, width=0)

with lig3d_col:
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

st.markdown("---")
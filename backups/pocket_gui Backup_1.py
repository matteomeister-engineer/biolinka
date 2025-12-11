from pathlib import Path

import re
import requests
import html
import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol
import streamlit.components.v1 as components



@st.cache_data(show_spinner=False)
def get_uniprot_annotation(accession: str):
    """
    Fetch protein name, biological function, and organism from UniProt
    for a given accession (e.g. Q99798).

    Returns a dict with keys:
      - 'protein_name'
      - 'function'
      - 'organism'
    """
    base_url = f"https://rest.uniprot.org/uniprotkb/{accession}"
    params = {"format": "json"}

    try:
        resp = requests.get(base_url, params=params, timeout=5)
    except Exception:
        return {"protein_name": None, "function": None, "organism": None}

    if resp.status_code != 200:
        return {"protein_name": None, "function": None, "organism": None}

    try:
        data = resp.json()
    except Exception:
        return {"protein_name": None, "function": None, "organism": None}

    # Protein recommended name
    protein_name = None
    try:
        protein_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except Exception:
        protein_name = None

    # Biological function: look for FUNCTION comment
    function_text = None
    try:
        for c in data.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts") or c.get("text") or []
                if texts:
                    function_text = texts[0].get("value")
                    break
    except Exception:
        function_text = None

    # Organism: scientific + common name if available
    organism_name = None
    try:
        org_obj = data.get("organism", {})
        sci = org_obj.get("scientificName")
        common = org_obj.get("commonName")
        if sci and common:
            organism_name = f"{sci} ({common})"
        elif sci:
            organism_name = sci
        elif common:
            organism_name = common
    except Exception:
        organism_name = None

    return {
        "protein_name": protein_name,
        "function": function_text,
        "organism": organism_name,
    }
# ---------------------- Paths ---------------------- #

DATA_PATH = Path("results/pockets_fpocket.csv")
STRUCT_DIR = Path("structures")

# ---------------------- Page setup & base CSS ---------------------- #

st.set_page_config(
    page_title="Biosensor Pocket Explorer",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     system-ui, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Metrics: no dark cards, just labels + numbers */
    .metric-card {
        background: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 1rem;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #4B5563 !important;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #111827 !important;
        margin: 0;
        padding: 0;
        line-height: 1.1;
        text-align: left;
    }

    /* Preset chips */
    .preset-container {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .preset-label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 0.4rem;
    }
    .stButton>button {
        border-radius: 999px !important;
        padding: 0.35rem 1.1rem !important;
        border: 1px solid #D1D5DB !important;  /* light grey border */
        background: #fcfcfd !important;        /* light grey background */
        color: #111827 !important;             /* dark text */
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------- Title & description ---------------------- #

st.title("🧬 Biosensor Pocket Explorer")

st.caption(
    "Explore fpocket-derived pockets from protein structures. "
    "Filter by ligand-like properties (size, polarity, charge, druggability) "
    "to identify pockets that could be engineered into biosensors "
    "for cell-based and organ-on-chip systems."
)

# ---------------------- Analyte search + presets ---------------------- #

# ensure key exists before widget
if "analyte_query" not in st.session_state:
    st.session_state["analyte_query"] = ""

st.markdown("### What do you want to sense?")

analyte_query = st.text_input(
    "Type an analyte (e.g. glucose, lactate, cortisol, metal ion, drug…) and I'll bias the search toward suitable pockets.",
    key="analyte_query",
    placeholder="e.g. glucose, lactate, cortisol, dopamine",
    label_visibility="collapsed",
)

# -------------- PRESETS (one row, small pills) -------------- #

st.markdown('<div class="preset-container">', unsafe_allow_html=True)

presets = [
    ("Glucose",   "glucose"),
    ("Lactate",   "lactate"),
    ("Urea",      "urea"),
    ("Dopamine",  "dopamine"),
    ("Serotonin", "serotonin"),
    ("Cortisol",  "cortisol"),
    ("Na⁺ / K⁺",  "ion"),
]

def set_analyte(query: str):
    st.session_state["analyte_query"] = query

cols = st.columns(len(presets))
for (label, query), col in zip(presets, cols):
    with col:
        st.button(
            label,
            key=f"preset_{query}",
            on_click=set_analyte,
            args=(query,),
        )

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Define analyte profiles ---------- #

ANALYTE_PROFILES = {
    # sugars / carbohydrates
    "glucose":     {"size": "medium", "polarity": "polar",       "charge": "neutral"},
    "sugar":       {"size": "medium", "polarity": "polar",       "charge": "neutral"},

    # organic acids
    "lactate":     {"size": "small",  "polarity": "polar",       "charge": "negative"},
    "lactic acid": {"size": "small",  "polarity": "polar",       "charge": "negative"},

    # urea
    "urea":        {"size": "small",  "polarity": "polar",       "charge": "neutral"},

    # neurotransmitters
    "dopamine":    {"size": "small",  "polarity": "mixed",       "charge": "positive"},
    "serotonin":   {"size": "small",  "polarity": "mixed",       "charge": "positive"},

    # steroids / hydrophobic hormones
    "cortisol":    {"size": "medium", "polarity": "hydrophobic", "charge": "neutral"},
    "steroid":     {"size": "medium", "polarity": "hydrophobic", "charge": "neutral"},

    # generic drug-like
    "drug":        {"size": "medium", "polarity": "mixed",       "charge": "neutral"},
    "antibiotic":  {"size": "medium", "polarity": "mixed",       "charge": "neutral"},

    # ions / metals
    "ion":         {"size": "small",  "polarity": "polar",       "charge": "negative"},
    "na+":         {"size": "small",  "polarity": "polar",       "charge": "negative"},
    "k+":          {"size": "small",  "polarity": "polar",       "charge": "negative"},
}

profile = None
aq = st.session_state["analyte_query"].lower().strip()
for key, prof in ANALYTE_PROFILES.items():
    if aq and key in aq:
        profile = prof
        break


# ---------- Sync profile to sidebar dropdowns via session_state ---------- #

SIZE_LABELS = {
    "small": "small (ions / tiny metabolites)",
    "medium": "medium (drug-like)",
    "large": "large (cofactors / lipids)",
}
POLARITY_LABELS = {
    "hydrophobic": "hydrophobic",
    "mixed": "mixed",
    "polar": "polar",
}
CHARGE_LABELS = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

prev_query = st.session_state.get("_last_analyte_query", "")
analyte_changed = (aq != prev_query)
st.session_state["_last_analyte_query"] = aq

if profile and analyte_changed:
    size_label = SIZE_LABELS.get(profile["size"])
    pol_label = POLARITY_LABELS.get(profile["polarity"])
    ch_label = CHARGE_LABELS.get(profile["charge"])

    if size_label:
        st.session_state["size_choice"] = size_label
    if pol_label:
        st.session_state["polarity_choice"] = pol_label
    if ch_label:
        st.session_state["charge_choice"] = ch_label

# ---------------------- Load data ---------------------- #

if not DATA_PATH.exists():
    st.error(
        f"Could not find `{DATA_PATH}`.\n\n"
        "Run `python scripts/parse_fpocket_pockets.py` first."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)
if df.empty:
    st.warning("The pockets table is empty. Parsing likely failed or no pockets were found.")
    st.stop()

# ---------------------- Enrich data ---------------------- #

numeric_cols = [
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
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def classify_size(v: float) -> str:
    if np.isnan(v):
        return "unknown"
    if v < 250:
        return "small"
    if v < 500:
        return "medium"
    return "large"


def classify_polarity(row) -> str:
    pol = row.get("polarity_score", np.nan)
    hyd = row.get("hydrophobicity_score", np.nan)
    if np.isnan(pol) or np.isnan(hyd):
        return "unknown"
    if pol > 5 and hyd < 10:
        return "polar"
    if hyd > 40 and pol < 10:
        return "hydrophobic"
    return "mixed"


def classify_charge(charge_score: float) -> str:
    if np.isnan(charge_score):
        return "unknown"
    if charge_score > 0.5:
        return "positive"
    if charge_score < -0.5:
        return "negative"
    return "neutral"


df["size_class"] = df["volume"].apply(classify_size)
df["polarity_class"] = df.apply(classify_polarity, axis=1)
df["charge_class"] = df["charge_score"].apply(classify_charge)


def normalize(col: pd.Series) -> pd.Series:
    col = col.astype(float)
    min_val, max_val = col.min(), col.max()
    if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
        return pd.Series([0.5] * len(col), index=col.index)
    return (col - min_val) / (max_val - min_val)


df["norm_druggability"] = normalize(df["druggability_score"]) if "druggability_score" in df else 0.5
df["norm_hydrophobicity"] = normalize(df["hydrophobicity_score"]) if "hydrophobicity_score" in df else 0.5
df["norm_alpha_density"] = normalize(df["alpha_sphere_density"]) if "alpha_sphere_density" in df else 0.5
df["norm_flexibility"] = normalize(df["flexibility"]) if "flexibility" in df else 0.5

if "volume" in df:
    vol = df["volume"].astype(float)
    target_vol = 350.0  # generic drug-like size
    df["volume_match"] = np.exp(-((vol - target_vol) ** 2) / (2 * (150.0 ** 2)))
else:
    df["volume_match"] = 0.5

df["biosensor_score"] = (
    0.4 * df["norm_druggability"]
    + 0.2 * df["volume_match"]
    + 0.2 * df["norm_hydrophobicity"]
    + 0.1 * df["norm_alpha_density"]
    + 0.1 * df["norm_flexibility"]
)

# ---------------------- Sidebar filters ---------------------- #

st.sidebar.header("Advanced filters (applied after analyte profile)")

size_choice = st.sidebar.selectbox(
    "Approximate ligand size",
    options=[
        "any",
        "small (ions / tiny metabolites)",
        "medium (drug-like)",
        "large (cofactors / lipids)",
    ],
    index=2,
    key="size_choice",
)

polarity_choice = st.sidebar.selectbox(
    "Ligand polarity / hydrophobicity",
    options=["any", "hydrophobic", "mixed", "polar"],
    index=2,
    key="polarity_choice",
)

charge_choice = st.sidebar.selectbox(
    "Ligand net charge",
    options=["any", "negative", "neutral", "positive"],
    index=2,
    key="charge_choice",
)

min_druggability = st.sidebar.slider(
    "Minimum pocket druggability",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
)

st.sidebar.markdown("---")

sort_by = st.sidebar.selectbox(
    "Sort pockets by",
    options=["biosensor_score", "druggability_score", "volume", "score"],
)

sort_desc = st.sidebar.checkbox("Sort descending", value=True)

st.sidebar.markdown("---")

# ---------------------- Apply filters ---------------------- #

df_filtered = df.copy()

# 1) analyte profile-based
if profile:
    if profile["size"] == "small":
        df_filtered = df_filtered[df_filtered["size_class"] == "small"]
    elif profile["size"] == "medium":
        df_filtered = df_filtered[df_filtered["size_class"] == "medium"]
    elif profile["size"] == "large":
        df_filtered = df_filtered[df_filtered["size_class"] == "large"]

    if profile["polarity"] in {"hydrophobic", "mixed", "polar"}:
        df_filtered = df_filtered[df_filtered["polarity_class"] == profile["polarity"]]

    if profile["charge"] in {"negative", "neutral", "positive"}:
        df_filtered = df_filtered[df_filtered["charge_class"] == profile["charge"]]

# 2) sidebar advanced filters

if size_choice.startswith("small"):
    df_filtered = df_filtered[df_filtered["size_class"] == "small"]
elif size_choice.startswith("medium"):
    df_filtered = df_filtered[df_filtered["size_class"] == "medium"]
elif size_choice.startswith("large"):
    df_filtered = df_filtered[df_filtered["size_class"] == "large"]

if polarity_choice != "any":
    df_filtered = df_filtered[df_filtered["polarity_class"] == polarity_choice]

if charge_choice != "any":
    df_filtered = df_filtered[df_filtered["charge_class"] == charge_choice]

if "druggability_score" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["druggability_score"] >= min_druggability]

if sort_by in df_filtered.columns:
    df_filtered = df_filtered.sort_values(by=sort_by, ascending=not sort_desc)

if df_filtered.empty:
    st.markdown("---")
    st.warning("No pockets match these requirements. Try changing the analyte or relaxing the filters.")
    st.stop()

# Best pocket = first row after filtering & sorting
best_pocket = df_filtered.iloc[0]
others = df_filtered.iloc[1:]
others_display = others.head(200)

# ---------------------- Top metric cards ---------------------- #

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Proteins)</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{df_filtered["protein_id"].nunique()}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Candidate pockets</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{len(df_filtered)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    if not df_filtered.empty:
        value = df_filtered["biosensor_score"].dropna().mean() * 100
        mean_bio = f"{value:.2f}".rstrip("0").rstrip(".") + "%"
    else:
        mean_bio = "0%"
    st.markdown('<div class="metric-title">Mean biosensor suitability</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">{mean_bio}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------- Interpretation helper ---------------------- #

def interpret_pocket(p):
    lines = []
    vol = p.get("volume", np.nan)
    drugg = p.get("druggability_score", np.nan)
    hyd = p.get("hydrophobicity_score", np.nan)
    pol = p.get("polarity_score", np.nan)
    ch = p.get("charge_score", np.nan)

    # Druggability
    if not np.isnan(drugg):
        if drugg >= 0.7:
            lines.append("• Druggability is **very high** – strong ligand-binding scaffold with good potential for biosensor engineering.")
        elif drugg >= 0.4:
            lines.append("• Druggability is **good** – likely to bind small molecules in a well-formed pocket.")
        elif drugg >= 0.2:
            lines.append("• Druggability is **moderate** – may still be usable, but might need more engineering.")
        else:
            lines.append("• Druggability is **low** – pocket may be shallow or poorly enclosed.")

    # Volume / ligand size
    if not np.isnan(vol):
        if vol < 200:
            lines.append("• Pocket volume suggests compatibility with **small ligands** (ions, very small metabolites).")
        elif vol < 450:
            lines.append("• Pocket volume suggests **drug-sized / medium ligands** (typical small-molecule drugs, many metabolites).")
        else:
            lines.append("• Pocket volume suggests **larger ligands** (cofactors, bulky metabolites, lipids or small peptides).")

    # Hydrophobicity vs polarity
    if not np.isnan(hyd) and not np.isnan(pol):
        if hyd > 40 and pol < 3:
            lines.append("• Pocket environment is **mostly hydrophobic** – suited for hydrophobic or amphipathic ligands (e.g. many drugs, steroids, aromatic pollutants).")
        elif pol > 5 and hyd < 10:
            lines.append("• Pocket environment is **highly polar** – suited for polar or charged metabolites.")
        else:
            lines.append("• Pocket has **mixed polarity** – can accommodate amphipathic ligands (both polar head and hydrophobic moiety).")

    # Charge bias
    if not np.isnan(ch):
        if ch > 0.5:
            lines.append("• Net pocket charge is **positive** – likely better for **negatively charged** ligands.")
        elif ch < -0.5:
            lines.append("• Net pocket charge is **negative** – likely better for **positively charged** ligands.")
        else:
            lines.append("• Net pocket charge is roughly **neutral** – does not strongly bias toward charged ligands.")

    if not lines:
        return (
            "No detailed interpretation available from the current descriptors. "
            "Use the 3D view to inspect geometry and environment manually."
        )

    lines.append(
        "\nOverall: These features are compatible with using this pocket as a "
        "small-molecule sensing domain in cellular or organ-on-chip contexts, "
        "but expression, folding and signal coupling still need experimental validation."
    )
    return "\n".join(lines)

# ---------------------- Middle layout: best pocket + 3D viewer, then interpretation row ---------------------- #
best_col, viewer_col = st.columns([1, 1])

with best_col:
    st.subheader("Best Matching Pocket")

    # -------- Extract core values -------- #
    prot = str(best_pocket.get("protein_id", "–"))
    pocket_id = best_pocket.get("pocket_id", "–")
    bios = best_pocket.get("biosensor_score", np.nan)
    drugg = best_pocket.get("druggability_score", np.nan)
    vol = best_pocket.get("volume", np.nan)
    hyd = best_pocket.get("hydrophobicity_score", np.nan)

    raw_org = best_pocket.get("source_organism", None)

    # -------- Fetch UniProt metadata -------- #
    protein_name = None
    function_text = None
    organism_name = None
    try:
        meta = get_uniprot_annotation(prot)
        protein_name = meta.get("protein_name")
        function_text = meta.get("function")
        organism_name = meta.get("organism")  # e.g. "Homo sapiens (Human)"
    except Exception:
        pass

    # Final organism string
    org_display = organism_name or raw_org or "Unknown"

    # -------- Show protein name (big title) -------- #
    if protein_name:
        st.markdown(f"**{protein_name}**")
    else:
        st.markdown("**Protein name unavailable (UniProt)**")

    # -------- Formatting numbers -------- #
    bios_str = "–" if np.isnan(bios) else f"{(bios * 100):.2f}".rstrip('0').rstrip('.') + "%"
    drugg_str = "–" if np.isnan(drugg) else f"{drugg:.3f}"
    vol_str = "–" if np.isnan(vol) else f"{vol:.1f} Å³"
    hyd_str = "–" if np.isnan(hyd) else f"{hyd:.1f}"

    # -------- Smooth 6-step gradient badge for biosensor score -------- #
    if np.isnan(bios):
        bios_label = "unknown"
        bios_bg = "#E5E7EB"
        bios_fg = "#374151"
    else:
        if bios < 0.20:
            bios_label = "very low"
            bios_bg = "#FEE2E2"
            bios_fg = "#7F1D1D"
        elif bios < 0.35:
            bios_label = "low"
            bios_bg = "#FECACA"
            bios_fg = "#7F1D1D"
        elif bios < 0.50:
            bios_label = "moderate"
            bios_bg = "#FEF3C7"
            bios_fg = "#92400E"
        elif bios < 0.65:
            bios_label = "medium-high"
            bios_bg = "#D9F99D"
            bios_fg = "#3E651C"
        elif bios < 0.80:
            bios_label = "high"
            bios_bg = "#BBF7D0"
            bios_fg = "#065F46"
        else:
            bios_label = "excellent"
            bios_bg = "#86EFAC"
            bios_fg = "#065F46"

    bios_badge_html = (
        f"<span style='"
        "display:inline-block;"
        "padding:3px 10px;"
        "border-radius:999px;"
        f"background:{bios_bg};"
        f"color:{bios_fg};"
        "font-size:0.85rem;"
        "font-weight:600;"
        "'>"
        f"{bios_str} ({bios_label})"
        "</span>"
    )

    # -------- External links (UnitProt + AlphaFold) -------- #
    uniprot_url = f"https://www.uniprot.org/uniprot/{prot}"
    alphafold_url = f"https://alphafold.ebi.ac.uk/entry/AF-{prot}-F1"

    uniprot_link_html = (
        f"<a href='{uniprot_url}' target='_blank' "
        "style='color:#0066CC; text-decoration:none;'>UniProt →</a>"
    )
    alphafold_link_html = (
        f"<a href='{alphafold_url}' target='_blank' "
        "style='color:#0066CC; text-decoration:none;'>AlphaFold →</a>"
    )
    links_html = f"{uniprot_link_html}&nbsp;&nbsp;|&nbsp;&nbsp;{alphafold_link_html}"

    # -------- INFO GRID (left = protein info, right = pocket metrics) -------- #
    pocket_info_html = (
        "<div style='"
        "display:grid;"
        "grid-template-columns:1fr 1fr;"
        "gap:14px 22px;"
        "font-size:0.95rem;"
        "line-height:1.25;"
        "padding-top:6px;"
        "'>"

        # LEFT column
        f"<div><strong>Protein ID</strong><br>{prot}<br>{links_html}</div>"
        f"<div><strong>Biosensor score</strong><br>{bios_badge_html}</div>"

        f"<div><strong>Source organism</strong><br>{org_display}</div>"
        f"<div><strong>Druggability</strong><br>{drugg_str}</div>"

        f"<div><strong>Pocket location</strong><br>{pocket_id}</div>"
        f"<div><strong>Hydrophobicity</strong><br>{hyd_str}</div>"

        # Keep matrix even
        f"<div></div>"
        f"<div><strong>Volume</strong><br>{vol_str}</div>"

        "</div>"
    )

    st.markdown(pocket_info_html, unsafe_allow_html=True)

with viewer_col:
    st.subheader("3D viewer")

    prot_id = str(best_pocket["protein_id"])
    pocket_num = str(best_pocket["pocket_id"])  # keep as string for filenames
    protein_pdb = STRUCT_DIR / f"{prot_id}.pdb"
    pocket_pdb = STRUCT_DIR / f"{prot_id}_out" / "pockets" / f"pocket{pocket_num}_atm.pdb"

    if not protein_pdb.exists():
        st.warning(f"Protein structure not found: {protein_pdb}")
    else:
        view = py3Dmol.view(width=600, height=320)

        # Protein cartoon
        with open(protein_pdb, "r") as f:
            view.addModel(f.read(), "pdb")
        view.setStyle({"cartoon": {"opacity": 0.7, "color": "0xAAAAAA"}})

        # Pocket atoms (if available)
        if pocket_pdb.exists():
            with open(pocket_pdb, "r") as f:
                view.addModel(f.read(), "pdb")
            view.setStyle(
                {"model": 1},
                {"stick": {"radius": 0.25}, "sphere": {"radius": 0.75}},
            )
            # Smooth zoom to the pocket model (1 second animation)
            view.zoomTo({"model": 1}, 1000)
        else:
            # Smooth zoom to whole protein
            view.zoomTo({}, 1000)

        # --- Zoom limits (min/max zoom) ---
        view.setZoomLimits(10, 300)

        view.setBackgroundColor("0xFFFFFF")
        html_view = view._make_html()
        components.html(html_view, height=340, scrolling=False)


# -------- Combined biological function + interpretation section -------- #

st.subheader("Biological function & biosensor potential")

interp_text = interpret_pocket(best_pocket)

# ---- UniProt function block (HTML-escaped, no markdown) ----
if function_text:
    safe_func = html.escape(function_text).replace("\n", "<br>")
else:
    safe_func = "<em>No function annotation available from UniProt for this entry.</em>"

# ---- Convert **bold** markdown in interpretation to <strong> ----
# We trust interp_text content (it's from our own function)
interp_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", interp_text)
interp_html = interp_html.replace("\n", "<br>")

st.markdown(
    f"""
    <div style="
        max-height: 260px;
        overflow-y: auto;
        font-size: 0.92rem;
        line-height: 1.45;
        padding-right: 0.5rem;
    ">
      <div style="margin-bottom: 0.6rem;">
        <strong>Biological function (UniProt)</strong><br>
        {safe_func}
      </div>
      <hr style="border:none; border-top:1px solid #E5E7EB; margin: 0.6rem 0;">
      <div>
        <strong>Interpretation for biosensing</strong><br>
        {interp_html}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Matching pockets table ---------------------- #

st.markdown("---")
st.subheader("Matching pockets table")

if others_display.empty:
    # If only one pocket matched, show it in the table too
    st.caption("Only one pocket matches these filters; showing it below.")
    st.dataframe(df_filtered.head(50), use_container_width=True)
else:
    st.caption("All other pockets that match your analyte profile and filters.")
    st.dataframe(others_display, use_container_width=True)
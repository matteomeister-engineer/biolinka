<div align="center">

# 🧬 Biolinka

### Interactive Protein Pocket Analysis & Ligand Binding Site Discovery

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://biolinka.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

**[🚀 Try the Live App →](https://biolinka.streamlit.app/)**

![Biolinka Banner](assets/banner.jpg)

</div>

---

## 📖 Overview

**Biolinka** is a lightweight, fully interactive protein pocket analysis tool that identifies, ranks, and visualizes potential ligand-binding sites within any protein structure — no heavy desktop software required.

It combines automated cavity detection, geometric scoring, biochemical heuristics, and real-time 3D visualization into a single streamlined interface. Users can:

- 🔍 **Detect pockets** automatically from any PDB structure
- 📊 **Rank binding sites** using geometric and biochemical descriptors
- 🌐 **Visualize in 3D** via an embedded, rotating py3Dmol viewer
- 🧪 **Interpret results** through an AI-assisted analysis panel
- 📋 **Compare all sites** in a clean, structured results table

> *Built as an independent research project to explore structural bioinformatics and make protein ligandability analysis accessible to everyone.*

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧲 Pocket Detection | Geometric cavity detection from raw PDB coordinates |
| 📐 Geometric Scoring | Volume, surface area, and depth analysis per pocket |
| 🧫 Biochemical Heuristics | Hydrophobicity and solvent accessibility scoring |
| 🌀 3D Visualization | Real-time interactive viewer powered by py3Dmol |
| 🧠 Interpretive Analysis | Natural-language explanation of each top pocket |
| 📊 Results Table | Ranked summary of all detected sites for easy comparison |

---

## 🖼️ Screenshots

<!-- Replace with actual screenshots from your app -->
> *Add screenshots of the app interface, 3D viewer, and results table here.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/matteomeister-engineer/biolinka.git
cd biolinka

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
biolinka/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python runtime specification
├── Dockerfile.fpocket        # Docker config for fpocket integration
│
├── scripts/                 # Core analysis scripts
│   └── ...                  # Pocket detection, scoring, visualization
│
├── structures/              # Sample PDB structure files
├── results/                 # Output files from pocket analysis
├── docking/                 # Docking-related scripts and data
├── files/                   # Auxiliary data files
├── backups/                 # Backup versions
└── assets/                  # Images and UI assets
```

---

## 🔬 How It Works

1. **Input** — Provide a PDB structure (upload or fetch by PDB ID)
2. **Detection** — The engine scans the structure for geometric cavities using coordinate-space analysis
3. **Scoring** — Each pocket is scored on volume, hydrophobicity, solvent accessibility, and depth
4. **Ranking** — Sites are ranked to surface the most promising ligand-binding candidates
5. **Visualization** — Top pockets are highlighted in the embedded py3Dmol 3D viewer
6. **Interpretation** — An analysis panel explains the biological relevance of each site
7. **Comparison** — All detected pockets are summarized in a results table

---

## 🧰 Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web interface and app framework
- **[py3Dmol](https://3dmol.csb.pitt.edu/)** — Interactive 3D molecular visualization
- **[BioPython](https://biopython.org/)** — PDB parsing and structural data handling
- **[NumPy](https://numpy.org/) / [SciPy](https://scipy.org/)** — Geometric computations and spatial analysis
- **[Pandas](https://pandas.pydata.org/)** — Results table management
- **[fpocket](https://github.com/Discngine/fpocket)** *(optional)* — Enhanced pocket detection via Docker

---

## 🐳 Docker (fpocket integration)

For enhanced pocket detection using fpocket:

```bash
docker build -f Dockerfile.fpocket -t biolinka-fpocket .
docker run -v $(pwd)/structures:/structures biolinka-fpocket
```

---

## 🌐 Live Demo

The app is deployed on Streamlit Cloud and is **free to use**:

**👉 [https://biolinka.streamlit.app/](https://biolinka.streamlit.app/)**

No installation or login required — just upload a PDB file and explore.

---

## 🤝 Contributing

Contributions, suggestions, and issue reports are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 👤 Author

**Mattéo Meister**
Biomedical Engineer | Computational Biology & Structural Bioinformatics

- 🌐 [Portfolio](https://matteomeister.framer.website/mybiomedicalprojects/biolinka)
- 🐙 [GitHub](https://github.com/matteomeister-engineer)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with curiosity, caffeine, and a passion for structural biology* 🧬

</div>
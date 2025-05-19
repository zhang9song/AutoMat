<p align="center">
  <img src="https://github.com/yyt-2378/AutoMat/blob/main/AutoMat_Logo.png" alt="AutoMat Logo" width="250"/>
</p>

---
# AutoMat
AutoMat focuses on characterization to property analysis.

## ✨ Project Overview

**AutoMat** is an end‑to‑end framework that enables academic researchers to go from **scanning transmission electron microscopy (STEM)** images to **crystal‑structure reconstruction** and finally to **materials‑property prediction** in one click. The core pipeline consists of four modules:

1. **MOE‑DIVAESR** — pattern‑adaptive STEM denoising & super‑resolution
2. **Template Retriever** — physics‑constrained structure‑template search
3. **STEM2CIF** — symmetry‑aware atomic‑coordinate / CIF generation
4. **MatterSim Wrapper** — machine‑learning potential for fast relaxation & property evaluation

> For the accompanying paper and benchmark dataset, see **STEM2Mat‑Bench** (link in the paper).

---

## 🛠️ Quick Installation

A **Conda** environment is recommended. A ready‑to‑use `img2struc.yaml` is provided in the repository root:

```bash
# 1) Create an isolated environment (Python 3.10+ recommended)
conda env create -f img2struc.yaml
conda activate img2struc

# 2) (Optional) Install with requirements.txt
pip install -r requirement.txt
```
## 🚀 Usage Example

AutoMat ships with an **agent‑based entry script** that lets you reconstruct a
crystal from a STEM micrograph and predict its properties in a single command
—or interactively via chat.

```text
agent_based.py
└─ Improved Materials‑AI Agent
   ├─ Mode 1: one‑shot pipeline
   └─ Mode 2: interactive chat

### 1 · One‑shot pipeline

```bash
python agent_based.py \
  --api_key YOUR_OPENAI_KEY \
  --image_path /path/to/img.png \
  --work_root ./results \
  --user_message "Elements: Al, Sb; dose = 30k"
```

### 2 · Interactive chat

```bash
python agent_based.py --api_key YOUR_OPENAI_KEY

User> /run /path/to/img.png ./results Elements: Al, Sb
User> exit      # or quit
```

### Output folders

| Folder          | Contents                                                                 |
| --------------- | ------------------------------------------------------------------------ |
| `01_recon/`     | Denoised / super‑resolved STEM image                                     |
| `02_label/`     | Best‑matched structure template                                          |
| `03_recon_cif/` | Reconstructed primitive unit cell (`*.cif`)                              |
| `04_relaxed/`   | MatterSim‑relaxed structure + predicted properties (LLM-agent output, etc.) |

All intermediate and final results are stored under `--work_root`.



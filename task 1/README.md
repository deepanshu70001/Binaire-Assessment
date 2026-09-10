# Task 1: Model Scraping & Acquisition via Hugging Face Python API

This repository contains the solution for **Task 1: Data Scraping / Model Acquisition**, completed as part of the assessment.

The objective is to download specific target machine learning models exclusively using the **Hugging Face Hub Python API** (`huggingface_hub`), strictly adhering to the constraint of **not using any CLI commands** (such as `huggingface-cli` or `git lfs`).

---

## 📋 Assessment Objectives & Constraints

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **1. Install HF Python API** | Installed and managed via `huggingface_hub` Python package | ✅ Completed |
| **2. API-Only Implementation** | Programmatic verification and download using `HfApi` and `snapshot_download` (No CLI) | ✅ Completed |
| **3. Download Target Models** | Both requested FreznelAI models fetched programmatically to local storage | ✅ Completed |

---

## 🎯 Target Models

The following two models were requested and downloaded:

1. **Model 1 (Face Landmarker)**
   - **Repository ID:** `freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm`
   - **Hugging Face Link:** [freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm](https://huggingface.co/freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm)
   - **Artifact:** `FreznelAI_1.0_Face-Landmarker_500M_FZFP4.frzm`

2. **Model 2 (Face Detector)**
   - **Repository ID:** `freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm`
   - **Hugging Face Link:** [freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm](https://huggingface.co/freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm)
   - **Artifact:** `FreznelAI_1.0_Face-Detector_500M_FZFP4.frzm`

---

## 📁 Repository Structure

```text
task 1/
├── README.md               # Assessment documentation and instructions
├── requirements.txt        # Python dependency list
├── hf_models.py            # Main Python script for API-only verification and download
└── models/                 # Destination directory for downloaded models
    ├── FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm/
    │   ├── .gitattributes
    │   ├── README.md
    │   └── FreznelAI_1.0_Face-Detector_500M_FZFP4.frzm
    └── FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm/
        ├── .gitattributes
        ├── README.md
        └── FreznelAI_1.0_Face-Landmarker_500M_FZFP4.frzm
```

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
Python **3.8+** is required.

### 2. Install Dependencies
Install the required `huggingface_hub` Python package:

```bash
pip install -r requirements.txt
```

Alternatively, install directly via pip:

```bash
pip install --upgrade huggingface_hub
```

---

## 🚀 Execution & Usage

The script `hf_models.py` provides clean execution flags and handles model verification and downloading purely via the Python SDK.

### 1. Verification / Dry Run
To verify connectivity with Hugging Face API, confirm publisher ownership (`freznelai`), and check target repository existence without downloading large binaries:

```bash
python hf_models.py --dry-run
```

**Expected output:**
```text
INFO | Verified freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm (revision c2565839386b9ba473f8162c35b6a8cf65333223)
INFO | Verified freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm (revision 2a722cde532f5593c68b1c33dfa9667c4b62d9fa)
INFO | Dry run successful; no files downloaded.
```

### 2. Full Download
To download both model repositories to the default `./models` directory:

```bash
python hf_models.py
```

### 3. Custom Output Directory
Specify a custom destination folder using the `--output-dir` argument:

```bash
python hf_models.py --output-dir path/to/destination
```

### 4. Force Re-download
To bypass local cache and re-download:

```bash
python hf_models.py --force-download
```

---

## ⚙️ Technical Implementation Details

The implementation in `hf_models.py` adheres strictly to the assessment constraints:

1. **`huggingface_hub.HfApi` for Remote Verification:**
   - Queries model existence using `api.list_models(author=MODEL_OWNER)`.
   - Obtains precise commit hashes and metadata via `api.model_info(...)`.
   - Ensures fail-fast behavior before initiating downloads.

2. **`huggingface_hub.snapshot_download` for File Retrieval:**
   - Downloads all model assets (configuration, documentation, `.frzm` weight files) directly via Python API calls.
   - Avoids subprocess calls or CLI commands completely.
   - Preserves repo layout in individual model folders.

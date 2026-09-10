# ML Practical Assessment

This repository contains three independent machine-learning practical assessment tasks. Each task is self-contained and has its own dependencies and detailed instructions.

## Contents

| Task | Description | Documentation |
| --- | --- | --- |
| Task 1 | Downloads two specified FreznelAI face-analysis models exclusively through the Hugging Face Python API. | [task 1/README.md](task%201/README.md) |
| Task 2 | Scrapes the first 100 featured PICO-8 cartridges and produces a structured dataset with metadata, source code, and community comments. | [task 2/README.md](task%202/README.md) |
| Task 3 | Provides a RAG assistant that retrieves PICO-8 game-code context and can generate Lua code through Groq. | [task 3/README.md](task%203/README.md) |

## Quick start

Use a separate virtual environment for each task because their dependencies serve different purposes.

```powershell
# Example: run Task 2 from PowerShell
cd "task 2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data_scrapper.py --verify
```

For Task 1, run `python hf_models.py --dry-run` before downloading model artifacts. For Task 3, copy `.env.example` to `.env`, set `GROQ_API_KEY`, then run `streamlit run app.py`. Refer to the task-level READMEs for all options, requirements, and expected outputs.

## Task-by-task guide

Run every command below from the named task directory. The examples use Windows PowerShell; on macOS or Linux, replace the virtual-environment activation command with `source .venv/bin/activate`.

### Task 1 — acquire the Hugging Face models

**Goal:** Download the two requested FreznelAI face-analysis model repositories using only the Hugging Face Python SDK.

1. Open a terminal in the task folder and create an isolated environment:

   ```powershell
   cd "task 1"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Confirm that the target repositories are accessible without downloading model binaries:

   ```powershell
   python hf_models.py --dry-run
   ```

3. Download both model repositories after the dry run succeeds:

   ```powershell
   python hf_models.py
   ```

4. Check that each model has a directory under `models/`. To choose another destination, run `python hf_models.py --output-dir path/to/models`.

The script uses `HfApi` and `snapshot_download`; it does not require the Hugging Face CLI or Git LFS. See [the Task 1 README](task%201/README.md) for the exact model IDs and available flags.

### Task 2 — create and verify the PICO-8 dataset

**Goal:** Collect the first 100 featured PICO-8 cartridges and generate `data/games.csv`, including metadata, descriptions, Lua code, licenses, likes, and up to five comments per game.

1. Set up the task environment. Git must be available because one dependency (`picotool`) is installed from its source repository.

   ```powershell
   cd "task 2"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Run the scraper. It saves progress incrementally, so it can be run again if a request is interrupted:

   ```powershell
   python src/data_scrapper.py --limit 100 --delay 0.8
   ```

3. Audit the generated dataset against the assessment requirements:

   ```powershell
   python src/data_scrapper.py --verify
   ```

The final CSV is written to `task 2/data/games.csv`; downloaded cartridge and artwork files are saved in `task 2/data/raw/`. Use `--refresh` only when you intentionally want to fetch existing records again. Full column definitions and troubleshooting are in [the Task 2 README](task%202/README.md).

### Task 3 — build and use the PICO-8 RAG assistant

**Goal:** Build a FAISS index from the PICO-8 dataset, retrieve relevant game-code examples, and optionally generate Pico-8 Lua code through Groq.

1. Create the environment and install dependencies:

   ```powershell
   cd "task 3"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create a local environment file from the supplied template and set your Groq key. Never commit this file:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and replace `your_groq_api_key_here` with a valid `GROQ_API_KEY`.

3. Build or rebuild the vector index from Task 2's generated dataset:

   ```powershell
   python build_rag.py --build --csv "../task 2/data/games.csv"
   ```

4. Test retrieval without using an LLM API:

   ```powershell
   python build_rag.py --retrieve-only --query "particle explosion sparks"
   ```

5. Generate code from the command line, or launch the web application:

   ```powershell
   python build_rag.py --query "Create a player character with double jump and wall sliding"
   streamlit run app.py
   ```

6. Run the retrieval benchmark when the index is available:

   ```powershell
   python test_rag.py
   ```

The generated index is stored in `task 3/faiss_store/`. See [the Task 3 README](task%203/README.md) for application features, model selection, and benchmark details.

## Repository layout

```text
.
|-- task 1/    Hugging Face model acquisition
|-- task 2/    PICO-8 scraper and dataset
`-- task 3/    PICO-8 RAG assistant
```

## Version-control notes

The root `.gitignore` excludes local environments, credentials, Python caches, downloaded model files, raw scraped assets, and generated vector indexes. These artifacts can be recreated or are too large/sensitive for ordinary source control. Do not commit a real `.env` file or API keys.

#!/usr/bin/env python3
"""Download the two assessment models using the Hugging Face Python API only."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_OWNER = "freznelai"
MODEL_IDS = (
    "freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm",
    "freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Binaire assessment models via huggingface_hub."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify API access and list the target models without downloading files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached files and download the model snapshots again.",
    )
    return parser.parse_args()


def verify_models(api: HfApi) -> None:
    """Use the API to verify the requested publisher and repositories exist."""
    published_ids = {model.id for model in api.list_models(author=MODEL_OWNER)}
    missing = [model_id for model_id in MODEL_IDS if model_id not in published_ids]
    if missing:
        raise RuntimeError(f"Requested model(s) not found for {MODEL_OWNER}: {missing}")

    for model_id in MODEL_IDS:
        info = api.model_info(model_id)
        logging.info("Verified %s (revision %s)", info.id, info.sha or "default")


def download_models(output_dir: Path, force_download: bool = False) -> None:
    api = HfApi()
    verify_models(api)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in MODEL_IDS:
        destination = output_dir / model_id.rsplit("/", maxsplit=1)[-1]
        logging.info("Downloading %s to %s", model_id, destination)
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=destination,
            force_download=force_download,
        )
        logging.info("Saved %s", destination)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.dry_run:
        verify_models(HfApi())
        logging.info("Dry run successful; no files downloaded.")
        return 0

    download_models(args.output_dir, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

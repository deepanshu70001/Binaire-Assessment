#!/usr/bin/env python3
"""Convenience alias for src/data_scrapper.py (supporting both single and double 'p' spelling)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_scrapper import main

if __name__ == "__main__":
    raise SystemExit(main())

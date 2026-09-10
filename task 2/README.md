# Task 2: PICO-8 Data Scraping & Dataset Generation

An end-to-end web scraping, binary parsing, and dataset extraction pipeline designed to scrape, extract, and structure the **first 100 featured cartridges** from the Lexaloffle PICO-8 BBS into a clean, comprehensive CSV dataset.

---

## 🎯 Target Source

- **Target Platform**: [Lexaloffle BBS — PICO-8 Featured Cartridges](https://www.lexaloffle.com/bbs/)
- **Target URL**: `https://www.lexaloffle.com/bbs/lister.php?cat=7&carts_tab=1&sub=2&mode=carts&orderby=featured`
- **Output Dataset**: [`data/games.csv`](data/games.csv) (100 rows, 17 columns, UTF-8-SIG encoded)

---

## 📋 Deliverables & Requirements Checklist

All 8 assessment criteria are fully extracted, cleaned, and represented in the dataset:

| # | Required Entry | Dataset Column(s) | Description | Extraction Status |
|---|----------------|-------------------|-------------|:-----------------:|
| 1 | **Name of game** | `game_name` | Title of the cartridge as published | ✅ **100/100** |
| 2 | **Name of author** | `author` | Developer handle / username | ✅ **100/100** |
| 3 | **Game artwork** | `artwork_url`, `artwork_path` | Public thumbnail URL & downloaded local PNG path (`data/raw/artwork/`) | ✅ **100/100** |
| 4 | **Game code** | `game_code` | Full decompressed Lua source code extracted from cartridge | ✅ **100/100** |
| 5 | **License (if any)** | `license` | Cartridge license (e.g. `CC4-BY-NC-SA` or `No License`) | ✅ **100/100** |
| 6 | **Like count** | `like_count` | Number of stars / favorites awarded by the community | ✅ **100/100** |
| 7 | **Game description** | `description` | Author's original description & instructions (cleaned) | ✅ **100/100** |
| 8 | **Top-5 comments** | `comment_1` to `comment_5` | Up to 5 community feedback comments for each game | ✅ **100/100** |

---

## 📂 Project Structure

```text
task 2/
├── data/
│   ├── games.csv                 # Final dataset of 100 games (CSV)
│   ├── notebook.ipynb            # Jupyter notebook for quick EDA & inspection
│   ├── scrape_errors.log         # Detailed extraction log
│   └── raw/
│       ├── artwork/              # 100 downloaded game cover artwork images (.png)
│       └── carts/                # 100 downloaded cartridge files (.p8.png)
├── src/
│   ├── data_scrapper.py          # Core scraper & binary cartridge decompressor
│   └── data_scraper.py           # Convenience alias wrapper
├── requirements.txt              # Project dependencies
└── README.md                     # Comprehensive documentation & submission guide
```

---

## 📊 Dataset Schema (`data/games.csv`)

The dataset is saved with `utf-8-sig` encoding for compatibility with Python, Excel, R, and modern database loaders.

| Column | Type | Example Value | Description |
|---|---|---|---|
| `game_id` | Integer | `160388` | Unique Lexaloffle post/thread ID |
| `game_name` | String | `Petal Quest 1.2` | Cleaned title of the game |
| `author` | String | `noelcody` | Game developer / creator username |
| `artwork_url` | String | `https://www.lexaloffle.com/bbs/thumbs/pico8_petal_quest_1.2.png` | Web URL to cartridge artwork preview |
| `artwork_path` | String | `data/raw/artwork/001_Petal_Quest_1.2.png` | Local path to downloaded artwork image |
| `cart_url` | String | `https://www.lexaloffle.com/bbs/cposts/pe/petal_quest_1.2-0.p8.png` | Cartridge binary download URL |
| `cart_path` | String | `data/raw/carts/001_Petal_Quest_1.2.p8.png` | Local path to cartridge `.p8.png` file |
| `game_code` | String | `--petal quest 1.2\n--by noel cody\n...` | Decompressed Lua game code |
| `license` | String | `CC4-BY-NC-SA` | License stated by author or `No License` |
| `like_count` | Integer | `106` | Total star count on the forum |
| `description` | String | `Petal Quest is a tiny adventure...` | Post description content |
| `comment_1` | String | `this is so cute` | 1st community comment |
| `comment_2` | String | `Such a sweet little game. Puzzles...` | 2nd community comment |
| `comment_3` | String | `Wonderful little game. The birds...` | 3rd community comment |
| `comment_4` | String | `Absolutely delightful little game! Loved it!` | 4th community comment |
| `comment_5` | String | `Awesome game! Loved the little puzzles.` | 5th community comment |
| `source_url` | String | `https://www.lexaloffle.com/bbs/?pid=160388` | Direct thread URL on Lexaloffle |

---

## 🔬 Technical Engineering Highlights

### 1. Advanced Binary Cartridge Code Decompression
PICO-8 cartridges (`.p8.png`) store game code, sprites, and map data steganographically within the 2 least-significant bits of each RGBA color channel. In addition, modern cartridges compress Lua code using PICO-8's proprietary **PXA compression** algorithm:
- **Steganography Decoding**: Uses `picotool` to read raw encoded data from the PNG container.
- **PXA Bit-Stream Decompressor**: Implements custom `BitReader` and `decode_pxa` routines in pure Python to handle move-to-front byte indexing, LZ77-style backward references, and variable-length bit decoding.
- **P8SCII to Unicode Transcoding**: Translates PICO-8 custom glyphs and characters (`p8scii_to_unicode`) into standard UTF-8 text so Lua source files are cleanly formatted and human-readable.

### 2. Resilient & Polite Scraping Architecture
- **Throttling & Backoff**: Includes rate-limiting delays between requests with automatic retries and exponential backoff for HTTP 429 / 5xx errors.
- **Incremental Resumability**: Saves progress incrementally. When re-executed, the scraper skips already completed entries, preventing redundant network requests.
- **Local Asset Caching**: Both artwork thumbnails and cartridge files are stored locally in `data/raw/` and reused across subsequent executions.

### 3. Rigorous Content Sanitization
- Thread elements are filtered to isolate the author's primary post text from BBS interface chrome (e.g., *“copy and paste the snippet”*, *“mark as spam”*, and player widgets).
- HTML whitespace, tabs, non-breaking spaces (`\xa0`), and newline artifacts are normalized for clean tabular output.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **Git** (Required for installing `picotool` from GitHub)

### 1. Installation

Clone or open the repository, create a virtual environment, and install dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Scraper

To run the scraper and generate/update `data/games.csv`:

```bash
python src/data_scrapper.py
```
*(Or use the alias `python src/data_scraper.py`)*

#### Available CLI Arguments:
```bash
python src/data_scrapper.py --help
```
- `--limit LIMIT`: Number of cartridges to scrape (default: `100`, max: `100`).
- `--delay DELAY`: Delay in seconds between requests for polite crawling (default: `0.8`).
- `--output OUTPUT`: Path to output CSV file (default: `data/games.csv`).
- `--refresh`: Force re-scraping of entries even if already present in the CSV.
- `--verify`: Run automated audit verifying all 100 entries against the 8 assessment deliverables.

Examples:
```bash
# Verify the dataset integrity without scraping:
python src/data_scrapper.py --verify

# Run scraper with custom rate-limit:
python src/data_scrapper.py --limit 100 --delay 0.5 --output data/games.csv
```

---

## 📈 Dataset Verification & Quick Inspection

You can inspect the generated CSV dataset using Python:

```python
import pandas as pd

df = pd.read_csv("data/games.csv")

print(f"Total Rows: {len(df)}")
print(f"Games with valid code: {(df['game_code'].str.len() > 0).sum()}/100")
print(f"Total Community Likes: {df['like_count'].sum():,}")
print("\nTop 5 Most Liked Games:")
print(df[['game_name', 'author', 'like_count', 'license']].sort_values('like_count', ascending=False).head())
```

### Summary Statistics

| Metric | Value |
|---|---|
| **Total Games Scraped** | 100 |
| **Games with Extracted Code** | 100 (100%) |
| **Average Game Code Length** | 36,790 characters |
| **Total Community Likes** | 18,020 likes |
| **Most Popular License** | CC4-BY-NC-SA (66 games) |
| **No License Specified** | 34 games |
| **Downloaded Artwork Files** | 100 files in `data/raw/artwork/` |
| **Downloaded Cartridge Files** | 100 files in `data/raw/carts/` |

---

## 📦 Submission Artifacts

1. **Dataset**: [`data/games.csv`](data/games.csv)
2. **Scraper Script**: [`src/data_scrapper.py`](src/data_scrapper.py) (with alias [`src/data_scraper.py`](src/data_scraper.py))
3. **Exploratory Notebook**: [`data/notebook.ipynb`](data/notebook.ipynb)
4. **Dependencies**: [`requirements.txt`](requirements.txt)
5. **Raw Assets**: [`data/raw/artwork/`](data/raw/artwork/) & [`data/raw/carts/`](data/raw/carts/)
6. **Documentation**: [`README.md`](README.md)

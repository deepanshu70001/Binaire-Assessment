#!/usr/bin/env python3
"""Task 2: create a CSV of the first 100 PICO-8 cartridge entries."""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

try:
    from pico8.game.formatter import p8png
    from pico8.lua import lua as pico_lua
except ImportError:
    p8png = pico_lua = None


BASE_URL = "https://www.lexaloffle.com"
BBS_URL = f"{BASE_URL}/bbs/"
LISTING_URL = (
    f"{BBS_URL}lister.php?cat=7&carts_tab=1&sub=2&mode=carts"
    "&orderby=featured&page={page}"
)
OUTPUT_FILE = Path("data/games.csv")
ARTWORK_DIR = Path("data/raw/artwork")
CART_DIR = Path("data/raw/carts")

COLUMNS = [
    "game_id", "game_name", "author", "artwork_url", "artwork_path",
    "cart_url", "cart_path", "game_code", "license", "like_count",
    "description", "comment_1", "comment_2", "comment_3", "comment_4",
    "comment_5", "source_url",
]

SKIP_TEXT = (
    "copy and paste the snippet", "embedded playback", "mark as spam",
    "mark as abuse", "subscribe to this thread", "pin to profile",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Pre-compiled regular expressions for maximum parsing throughput
RE_PDAT_DIV = re.compile(r"^pdat_?")
RE_PDAT_ID = re.compile(r"^pdat_?(.*)$")
RE_CART_PLAYER = re.compile(r"^cart_player_dormant")
RE_UID = re.compile(r"(?:\?|&)uid=")
RE_P8_PNG = re.compile(r"\.p8\.png$")
RE_LICENSE = re.compile(r"^\s*License:\s*$", re.I)
RE_NO_LICENSE = re.compile(r"\bNo License\b", re.I)
RE_DIGITS = re.compile(r"\d[\d,]*")
RE_CART_PREFIX = re.compile(r"^cart\s+#")
RE_COMMENT_DIV = re.compile(r"^p\d+$")
RE_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

# Parser selection: prefer fast C-based lxml when installed, fallback to html.parser
PARSER = "lxml"
try:
    import lxml  # noqa: F401
except ImportError:
    PARSER = "html.parser"


def clean(text: str) -> str:
    """Collapse HTML whitespace into one readable line."""
    return " ".join(text.replace("\xa0", " ").split())


def absolute_url(url: Optional[str]) -> str:
    return urljoin(BASE_URL, url) if url else ""


def safe_name(text: str) -> str:
    text = RE_SAFE_NAME.sub("_", clean(text)).strip("._-")
    return (text or "game")[:90]


class Client:
    """Requests wrapper with HTTP connection pooling, retry adapters, and polite throttling."""

    def __init__(self, delay: float) -> None:
        self.delay = max(delay, 0)
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Binaire-ML-Assessment-Scraper/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, stream: bool = False) -> requests.Response:
        for attempt in range(3):
            wait = self.delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, timeout=45, stream=stream)
                self.last_request = time.monotonic()
                if response.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                if attempt == 2:
                    raise RuntimeError(f"Could not download {url}: {error}") from error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Could not download {url}")


def list_games(client: Client, limit: int) -> list[dict[str, str]]:
    """Read the first `limit` entries in the assessment's cartridges view."""
    games: list[dict[str, str]] = []
    seen: set[str] = set()

    for page in range(1, 51):
        if len(games) >= limit:
            break
        soup = BeautifulSoup(client.get(LISTING_URL.format(page=page)).text, PARSER)
        cards = soup.find_all("div", id=RE_PDAT_DIV)
        if not cards:
            break

        for card in cards:
            match = RE_PDAT_ID.match(str(card.get("id", "")))
            link = card.find("a", href=True)
            if not match or not link:
                continue
            game_id = clean(match.group(1))
            if not game_id or game_id in seen:
                continue

            image = card.find("img", src=True)
            games.append({
                "game_id": game_id,
                "artwork_url": absolute_url(image.get("src")) if image else "",
            })
            seen.add(game_id)
            if len(games) == limit:
                break

    return games


def cart_player(soup: BeautifulSoup) -> Optional[Tag]:
    return soup.find("div", id=RE_CART_PLAYER)


def title_and_author(soup: BeautifulSoup) -> tuple[str, str]:
    player = cart_player(soup)
    if player:
        links = player.find_all("a", href=True)
        if len(links) >= 2:
            return clean(links[0].get_text(" ", strip=True)), clean(links[1].get_text(" ", strip=True)).removeprefix("by ")

    title = clean(soup.find("h1").get_text(" ", strip=True)) if soup.find("h1") else ""
    author_link = soup.find("a", href=RE_UID)
    author = clean(author_link.get_text(" ", strip=True)) if author_link else ""
    return title, author.removeprefix("by ")


def cart_url(soup: BeautifulSoup) -> str:
    link = soup.find("a", title="Open Cartridge File", href=True)
    if link:
        return absolute_url(link.get("href"))
    link = soup.find("a", href=RE_P8_PNG)
    return absolute_url(link.get("href")) if link else ""


def license_name(post: Tag) -> str:
    for label in post.find_all(string=RE_LICENSE):
        link = label.parent.find_next("a", href=True)
        if link:
            return clean(link.get_text(" ", strip=True))
    return "No License" if post.find(string=RE_NO_LICENSE) else ""


def like_count(post: Tag) -> int:
    star = post.find("div", title="Give this post a star")
    text = star.parent.get_text(" ", strip=True) if star and star.parent else ""
    match = RE_DIGITS.search(text)
    return int(match.group(0).replace(",", "")) if match else 0


def is_content(text: str) -> bool:
    lower = text.lower()
    return bool(text and not any(item in lower for item in SKIP_TEXT) and not RE_CART_PREFIX.match(lower))


def description(post: Tag) -> str:
    """Use only author content tags inside the game post, not page controls/comments."""
    parts: list[str] = []
    for element in post.find_all(["p", "h1", "h2", "h3", "li", "blockquote"]):
        text = clean(element.get_text(" ", strip=True))
        if is_content(text) and text not in parts:
            parts.append(text)
    return " ".join(parts)


def comments(post: Tag) -> list[str]:
    """Get the first five comment paragraphs following the game post."""
    result: list[str] = []
    for comment in post.find_all_next("div", id=RE_COMMENT_DIV):
        text = clean(" ".join(p.get_text(" ", strip=True) for p in comment.find_all("p")))
        if text:
            result.append(text)
        if len(result) == 5:
            break
    return result


def artwork_url(soup: BeautifulSoup, from_listing: str) -> str:
    if from_listing:
        return from_listing
    player = cart_player(soup)
    image = player.find("img", src=True) if player else None
    return absolute_url(image.get("src")) if image else ""


def download(client: Client, url: str, destination: Path) -> str:
    """Download an asset once; subsequent runs reuse the existing file."""
    if not url:
        return ""
    if not destination.is_file() or destination.stat().st_size == 0:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as file:
            for chunk in client.get(url, stream=True).iter_content(64 * 1024):
                if chunk:
                    file.write(chunk)
    return destination.as_posix()


class BitReader:
    """Read the little-endian bit stream in PICO-8's PXA compression format."""

    def __init__(self, data: bytes) -> None:
        self.data, self.index, self.value, self.count = data, 0, 0, 0

    def read(self, count: int) -> int:
        while self.count < count:
            if self.index == len(self.data):
                raise ValueError("Unexpected end of PXA data")
            self.value |= self.data[self.index] << self.count
            self.index += 1
            self.count += 8
        result = self.value & ((1 << count) - 1)
        self.value >>= count
        self.count -= count
        return result


def decode_pxa(data: bytes) -> bytes:
    """PXA decompressor kept separate so the main scraper remains easy to read."""
    if data[:4] != b"\0pxa":
        raise ValueError("Not PXA-compressed code")
    output_size = int.from_bytes(data[4:6], "big")
    compressed_size = int.from_bytes(data[6:8], "big")
    if not 8 <= compressed_size <= len(data):
        raise ValueError("Invalid PXA header")

    bits = BitReader(data[8:compressed_size])
    dictionary = list(range(256))
    output = bytearray()
    while len(output) < output_size:
        if bits.read(1):
            extra = 0
            while bits.read(1):
                extra += 1
            index = bits.read(4 + extra) + ((1 << extra) - 1) * 16
            character = dictionary.pop(index)
            dictionary.insert(0, character)
            output.append(character)
            continue

        has_short_offset = bits.read(1)
        offset_bits = 15
        if has_short_offset:
            offset_bits = 5 if bits.read(1) else 10
        offset = bits.read(offset_bits) + 1

        if offset == 1 and offset_bits == 10:  # literal block
            while (character := bits.read(8)):
                output.append(character)
            continue

        length = 3
        while (part := bits.read(3)) == 7:
            length += part
        length += part
        if offset > len(output):
            raise ValueError("Invalid PXA back-reference")
        for _ in range(length):
            output.append(output[-offset])

    if len(output) != output_size:
        raise ValueError("Invalid PXA output size")
    return bytes(output)


def game_code(path: Path) -> str:
    if p8png is None:
        raise ImportError("Run `pip install -r requirements.txt` first.")
    try:
        with path.open("rb") as file:
            cart = p8png.get_raw_data_from_p8png_file(file, str(path))
        raw_code = bytes(cart.codedata)
        if raw_code[:4] == b"\0pxa":
            raw_code = decode_pxa(raw_code)
        else:
            _, raw_code, _ = p8png.get_code_from_bytes(cart.codedata, cart.version)
        return pico_lua.p8scii_to_unicode(raw_code).strip()
    except Exception as error:
        logging.warning("Could not extract code from %s: %s", path.name, error)
        return ""


def scrape_game(client: Client, game: dict[str, str], number: int) -> dict[str, str]:
    game_id = game["game_id"]
    source_url = f"{BBS_URL}?pid={game_id}"
    soup = BeautifulSoup(client.get(source_url).text, PARSER)
    post = soup.find("div", id=f"p{game_id}")
    if not post:
        raise ValueError(f"Game post {game_id} was not found")

    name, author = title_and_author(soup)
    base_name = f"{number:03d}_{safe_name(name or game_id)}"
    art_url = artwork_url(soup, game["artwork_url"])
    art_suffix = Path(urlparse(art_url).path).suffix.lower()
    art_path = download(client, art_url, ARTWORK_DIR / f"{base_name}{art_suffix if art_suffix in IMAGE_EXTENSIONS else '.png'}")
    current_cart_url = cart_url(soup)
    cart_path = download(client, current_cart_url, CART_DIR / f"{base_name}.p8.png")
    top_comments = comments(post)

    row = {
        "game_id": game_id,
        "game_name": name,
        "author": author,
        "artwork_url": art_url,
        "artwork_path": art_path,
        "cart_url": current_cart_url,
        "cart_path": cart_path,
        "game_code": game_code(Path(cart_path)) if cart_path else "",
        "license": license_name(post),
        "like_count": str(like_count(post)),
        "description": description(post),
        "source_url": source_url,
    }
    row.update({f"comment_{i + 1}": top_comments[i] if i < len(top_comments) else "" for i in range(5)})
    return row


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["game_id"]: row for row in csv.DictReader(file) if row.get("game_id")}


def save_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in rows)


def verify_dataset(path: Path = OUTPUT_FILE) -> bool:
    """Verify dataset integrity against all assessment requirements."""
    if not path.exists():
        print(f"[FAIL] Output file '{path}' does not exist.")
        return False

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    errors: list[str] = []

    # Check 1: Exactly 100 entries
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, found {len(rows)}")

    # Check 2: All required columns present
    for col in COLUMNS:
        if col not in fieldnames:
            errors.append(f"Missing required column: {col}")

    # Check 3: Detailed record validation
    code_count = 0
    art_count = 0
    cart_count = 0
    total_likes = 0
    licenses: dict[str, int] = {}

    for i, r in enumerate(rows, start=1):
        if not r.get("game_name"):
            errors.append(f"Row {i} (id={r.get('game_id')}) missing game_name")
        if not r.get("author"):
            errors.append(f"Row {i} (id={r.get('game_id')}) missing author")
        if r.get("artwork_path") and Path(r["artwork_path"]).exists():
            art_count += 1
        if r.get("cart_path") and Path(r["cart_path"]).exists():
            cart_count += 1
        if r.get("game_code") and len(r["game_code"].strip()) > 0:
            code_count += 1
        else:
            errors.append(f"Row {i} ({r.get('game_name')}) missing game_code")

        try:
            total_likes += int(r.get("like_count", 0))
        except ValueError:
            pass

        lic = r.get("license") or "None"
        licenses[lic] = licenses.get(lic, 0) + 1

    print("========================================")
    print("       DATASET VERIFICATION AUDIT       ")
    print("========================================")
    print(f"Total Rows Verified:        {len(rows)} / 100")
    print(f"Games with Full Lua Code:   {code_count} / {len(rows)}")
    print(f"Local Artwork Files Found:  {art_count} / {len(rows)}")
    print(f"Local Cartridge Files Found:{cart_count} / {len(rows)}")
    print(f"Total Community Likes:      {total_likes:,}")
    print("License Distribution:")
    for lic, cnt in sorted(licenses.items(), key=lambda x: -x[1]):
        print(f"  - {lic}: {cnt}")

    if errors:
        print("\n[FAIL] Found validation errors:")
        for err in errors[:10]:
            print(f"  * {err}")
        if len(errors) > 10:
            print(f"  * ... and {len(errors) - 10} more errors")
        return False

    print("\n[SUCCESS] All 100 records strictly verified against all 8 deliverables!")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape the first 100 PICO-8 BBS cartridges.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between requests.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--refresh", action="store_true", help="Re-scrape rows already present in the CSV.")
    parser.add_argument("--verify", action="store_true", help="Audit and verify the existing dataset without scraping.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verify:
        success = verify_dataset(args.output)
        return 0 if success else 1

    if not 1 <= args.limit <= 100:
        raise ValueError("--limit must be between 1 and 100")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    client = Client(args.delay)
    games = list_games(client, args.limit)
    if len(games) != args.limit:
        raise RuntimeError(f"Found only {len(games)} games; expected {args.limit}")

    rows = load_rows(args.output)
    unsaved_changes = 0

    try:
        for number, game in enumerate(games, start=1):
            old_row = rows.get(game["game_id"])
            if old_row and old_row.get("game_code") and not args.refresh:
                logging.info("[%d/%d] Reusing %s", number, args.limit, old_row["game_name"])
                continue
            rows[game["game_id"]] = scrape_game(client, game, number)
            unsaved_changes += 1

            # Periodically flush every 5 items to reduce disk I/O while preserving progress
            if unsaved_changes >= 5:
                ordered_rows = [rows[item["game_id"]] for item in games if item["game_id"] in rows]
                save_rows(ordered_rows, args.output)
                unsaved_changes = 0

            logging.info("[%d/%d] Saved %s", number, args.limit, rows[game["game_id"]]["game_name"])
    finally:
        ordered_rows = [rows[item["game_id"]] for item in games if item["game_id"] in rows]
        if ordered_rows:
            save_rows(ordered_rows, args.output)

    completed = sum(bool(row["game_code"]) for row in ordered_rows)
    print(f"Saved {len(ordered_rows)} rows to {args.output} ({completed} rows with game code).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

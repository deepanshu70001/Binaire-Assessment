"""
test_rag.py — Performance & Retrieval Benchmark for Pico-8 RAG System.

Tests retrieval accuracy, latency, and optionally full LLM code generation.
Usage:
  python test_rag.py             # Runs fast retrieval benchmark (22 test cases)
  python test_rag.py --test-llm  # Also runs 1 end-to-end LLM code generation test
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from src.vectorstore import FaissVectorStore

parser = argparse.ArgumentParser(description="Test RAG performance on Pico-8 dataset")
parser.add_argument("--test-llm", action="store_true", help="Also test LLM code generation")
parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks (default: 5)")
args = parser.parse_args()

# ─── Setup ───
print("Loading FAISS vector store...")
t_load_start = time.perf_counter()
store = FaissVectorStore("faiss_store")
store.load()
t_load = (time.perf_counter() - t_load_start) * 1000
print(f"Loaded {store.index.ntotal} vectors in {t_load:.1f} ms\n")

passed = 0
failed = 0
latencies = []


def run_test(category, name, query, expected_games, top_k=args.top_k):
    """Check if expected games appear in top_k retrieved results and benchmark latency."""
    global passed, failed, latencies
    t0 = time.perf_counter()
    results = store.query(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    latencies.append(elapsed_ms)

    found_names = set()
    for r in results:
        meta = r.get("metadata", {})
        game = meta.get("game_name", "")
        if game:
            found_names.add(game)

    # Check if at least one expected game is in results
    matches = [g for g in expected_games if any(g.lower() in fn.lower() for fn in found_names)]
    status = "[PASS]" if matches else "[FAIL]"
    if matches:
        passed += 1
    else:
        failed += 1

    print(f"  {status} | [{category}] {name} ({elapsed_ms:.1f} ms)")
    print(f"         Query: \"{query}\"")
    print(f"         Expected: {expected_games}")
    print(f"         Found:    {list(found_names)}")
    if matches:
        print(f"         Matched:  {matches}")
    print()


# ═══════════════════════════════════════════════════════════
# TEST CASES — Retrieval Accuracy & Latency Benchmark
# ═══════════════════════════════════════════════════════════

print("=" * 65)
print("  TEST SUITE: Pico-8 RAG Retrieval Benchmark (100 Games Dataset)")
print("=" * 65 + "\n")

# --- 1. Search by exact game name ---
run_test("Exact Name", "PICO-BALL", "PICO-BALL", ["PICO-BALL"])
run_test("Exact Name", "Moss Moss", "Moss Moss", ["Moss Moss"])
run_test("Exact Name", "Air Delivery", "Air Delivery", ["Air Delivery"])
run_test("Exact Name", "Petal Quest", "Petal Quest", ["Petal Quest"])

# --- 2. Search by game genre/type ---
run_test(
    "Genre", "Platformer",
    "platformer game with jumping and obstacles",
    ["carrot_kingdom", "super hat girl", "Moss Moss", "LootSlime", "Feathered Escape"]
)
run_test(
    "Genre", "Puzzle",
    "puzzle game with logic and levels",
    ["Baba Is You", "Puzzles of the Paladin", "sokoblox", "Solitomb", "nemonemo", "Dino Sort", "Dust Bunny"]
)
run_test(
    "Genre", "Racing/Driving",
    "racing driving car speed game",
    ["driftmania", "top_speed", "Whiplash Taxi"]
)
run_test(
    "Genre", "Shooter (shmup)",
    "shooting enemies bullets lasers",
    ["ghostwave", "Steel Surge", "S.P.A.C.E.", "PRAXIS FIGHTER", "Touhou"]
)
run_test(
    "Genre", "Flight Sim",
    "flying airplane flight aircraft",
    ["Air Pico", "tinyhawk", "picowings"]
)

# --- 3. Search by game mechanic ---
run_test(
    "Mechanic", "Mouse Controls",
    "mouse click controls cursor point and click",
    ["Mouse Required"]
)
run_test(
    "Mechanic", "Roguelike / Auto-Battler",
    "roguelike items inventory dungeon auto battler",
    ["From Rust To Ash", "Libryinth", "clockwise_knight"]
)
run_test(
    "Mechanic", "Sports / Bowling",
    "bowling pins ball alley strike",
    ["freds72_bowling"]
)
run_test(
    "Mechanic", "Sports / Volleyball",
    "volleyball sports ball net serve",
    ["Vacay Volley"]
)
run_test(
    "Mechanic", "Suika / Merge Dropper",
    "merge fruits drop physics watermelon suika",
    ["Suika Game", "marble_merger"]
)

# --- 4. Search by theme/setting ---
run_test(
    "Theme", "Underwater / Ocean",
    "swimming underwater ocean fish diving",
    ["tad", "Subsurface"]
)
run_test(
    "Theme", "Space Exploration",
    "space stars planets rocket spaceship cosmic",
    ["S.P.A.C.E.", "The Heavens", "To take root Among the Stars"]
)
run_test(
    "Theme", "Food / Cooking",
    "pizza food cooking kitchen delivery chef",
    ["PIZZA PANDA", "Time For Lunch"]
)
run_test(
    "Theme", "Spooky / Dungeon",
    "skeleton bones dungeon undead ghost",
    ["Ruby Eyes", "Skeleton Gelatin", "Don't Dig Up the Dead", "blood_of_vladula"]
)
run_test(
    "Theme", "Cute Pets",
    "cat kitten pet animal cute cuddle",
    ["Pet the cat", "Lab Cat"]
)

# --- 5. Search by Pico-8 code patterns & API ---
run_test(
    "Code Pattern", "btn() / btnp() Player Input",
    "button input btn btnp arrow keys controller",
    ["carrot_kingdom", "Moss Moss", "super hat girl", "Vacay Volley"]
)
run_test(
    "Code Pattern", "Explosions & Particle Effects",
    "particle effects explosion sparks blast smoke",
    ["hotwax", "From Rust To Ash", "manbomber", "ghostwave"]
)
run_test(
    "Code Pattern", "Hitbox / Collision Detection",
    "collision detection hitbox bounding box overlap",
    ["carrot_kingdom", "super hat girl", "PICO-BALL"]
)

# --- 6. End-to-End LLM Generation (Optional) ---
if args.test_llm:
    print("=" * 65)
    print("  LLM CODE GENERATION BENCHMARK")
    print("=" * 65 + "\n")
    try:
        from src.search import RAGSearch
        rag = RAGSearch("faiss_store")
        sample_query = "Create a simple bouncing ball game with paddle"
        print(f"Query: \"{sample_query}\"")
        print("Generating Pico-8 code via LLM...")
        t_llm_start = time.perf_counter()
        code_resp = rag.query(sample_query, top_k=3)
        t_llm = (time.perf_counter() - t_llm_start)
        print(f"Generated response in {t_llm:.2f}s:\n")
        print(code_resp[:300] + ("..." if len(code_resp) > 300 else ""))
        print("\n[PASS] LLM Generation Test Succeeded!")
    except Exception as e:
        print(f"\n[FAIL] LLM Generation Test Failed: {e}")

# ═══════════════════════════════════════════════════════════
# RESULTS & PERFORMANCE BENCHMARK SUMMARY
# ═══════════════════════════════════════════════════════════

total = passed + failed
avg_lat = sum(latencies) / len(latencies) if latencies else 0
min_lat = min(latencies) if latencies else 0
max_lat = max(latencies) if latencies else 0

print("=" * 65)
print("  PERFORMANCE BENCHMARK RESULTS")
print("=" * 65)
print(f"  Retrieval Accuracy:  {passed}/{total} passed ({100*passed//total}%)")
print(f"  [PASS] Passed:       {passed}")
print(f"  [FAIL] Failed:       {failed}")
print("-" * 65)
print(f"  Vector Store Size:   {store.index.ntotal} vectors")
print(f"  Index Load Time:     {t_load:.1f} ms")
print(f"  Average Query Time:  {avg_lat:.1f} ms")
print(f"  Fastest Query Time:  {min_lat:.1f} ms")
print(f"  Slowest Query Time:  {max_lat:.1f} ms")
print("=" * 65)

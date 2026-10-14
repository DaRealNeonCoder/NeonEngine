"""
analyze_matches.py -- turn a games JSONL into per-matchup results with proper uncertainty.

Input: one JSON object per line, one line per game:
  {"matchup": "acp_thrifty__vs__ocp_prodigal",   # any id; games with the same id are pooled
   "a": "acp_thrifty", "b": "ocp_prodigal",
   "opening_id": 17,        # SAME id for the two games of a colour-swapped pair
   "black": "a",            # which side moved first in this game ("a" or "b")
   "outcome": 1,            # +1 black won, 0 draw, -1 white won  (same convention as the Python runner)
   "plies": 31,
   "budget": {"type": "time_ms", "value": 1000}}   # optional, copied into the report

Usage: python analyze_matches.py games.jsonl [--boot 20000]
"""
import argparse
import collections
import json
import math

import numpy as np


def elo(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def score_of(rec, who):
    s_black = 0.5 * (rec["outcome"] + 1.0)  # 0 / 0.5 / 1 for black
    return s_black if rec["black"] == who else 1.0 - s_black


def analyse(recs, n_boot, rng):
    a, b = recs[0]["a"], recs[0]["b"]
    n = len(recs)
    sa = np.array([score_of(r, "a") for r in recs])
    wins = int((sa == 1).sum()); draws = int((sa == 0.5).sum()); losses = int((sa == 0).sum())

    # pair games by opening: the two colour-swapped games share an opening, so they are
    # NOT independent -> resample whole openings, not single games.
    by_open = collections.defaultdict(list)
    for r, s in zip(recs, sa):
        by_open[r["opening_id"]].append(s)
    units = np.array([np.mean(v) for v in by_open.values()])
    both_colours = all(len(v) == 2 for v in by_open.values())

    mean = float(sa.mean())
    boots = rng.choice(units, size=(n_boot, len(units)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    black_scores = [0.5 * (r["outcome"] + 1.0) for r in recs]
    as_black = [score_of(r, "a") for r in recs if r["black"] == "a"]
    as_white = [score_of(r, "a") for r in recs if r["black"] == "b"]

    print(f"\n=== {a}  vs  {b}   ({n} games, {len(units)} openings) ===")
    if not both_colours:
        print("  WARNING: some openings were not played with both colours -> colour bias not cancelled")
    print(f"  {a}: W {wins}  D {draws}  L {losses}   draw rate {draws / n:.3f}")
    print(f"  score({a}) = {mean:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   (opening-level bootstrap)")
    print(f"  Elo({a} - {b}) = {elo(mean):+.0f}   95% CI [{elo(lo):+.0f}, {elo(hi):+.0f}]")
    verdict = "SIGNIFICANT" if (lo > 0.5 or hi < 0.5) else "not distinguishable from equal at 95%"
    print(f"  -> {verdict}")
    if as_black and as_white:
        print(f"  {a} as black: {np.mean(as_black):.3f} (n={len(as_black)})   as white: {np.mean(as_white):.3f} (n={len(as_white)})")
    print(f"  first-player score (all games): {np.mean(black_scores):.3f}")
    plies = [r["plies"] for r in recs if "plies" in r]
    if plies:
        print(f"  mean game length: {np.mean(plies):.1f} plies")
    bud = recs[0].get("budget")
    if bud:
        print(f"  budget: {bud}")
    half = 1.96 * units.std(ddof=1) / math.sqrt(len(units)) if len(units) > 1 else float("nan")
    print(f"  half-width now ~{half:.3f};  openings needed for +-0.03: ~{int(math.ceil((1.96 * units.std(ddof=1) / 0.03) ** 2)) if len(units) > 1 else '?'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    groups = collections.defaultdict(list)
    with open(args.games) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[r.get("matchup", f"{r['a']}__vs__{r['b']}")].append(r)
    for _, recs in groups.items():
        analyse(recs, args.boot, rng)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
baseline_ngram.py — a dumb-but-honest baseline. NO neural net, NO GPU.

Why this exists (read this before trusting any transformer number):
  A trigram model just counts how often step C follows the pair (A, B). It is
  pure local memorization: it CANNOT look back 12 steps to check "was there a
  clean before this deposit", and it has near-zero ability to generalize to an
  unseen family. So:
    * It tells you the FLOOR. If the transformer gets 70% exact match and this
      gets 65%, the transformer barely earned its GPU hours. If this gets 30%,
      the transformer is doing real work.
    * The GAP between this and the transformer IS our evidence that the
      transformer learned process logic, not surface statistics — especially on
      the hidden 4th family, where this baseline should collapse.

Back-off: trigram -> bigram -> unigram, so unseen contexts still predict
something (important for OOD / Task 4 fairness).

Usage:
    # train the counts and self-evaluate next-step + completion on a held-out
    python baseline_ngram.py --train data/mosfet.csv data/igbt.csv data/ic.csv \
        --heldout data/heldout_ic.csv

    # or produce Task 1 / Task 2 submission files from the official eval input
    python baseline_ngram.py --train data/*.csv \
        --valid eval_input_valid.csv --out submission_baseline
"""

from __future__ import annotations
import argparse, csv
from collections import defaultdict, Counter
from pathlib import Path

from generate_sequences import read_csv_sequences

BOS, EOS = "<BOS>", "<EOS>"


def family_of(path: str) -> str:
    n = Path(path).name.lower()
    for fam in ("mosfet", "igbt", "ic"):
        if fam in n:
            return fam
    return "unk"


def load(paths) -> list[tuple[str, list[str]]]:
    out = []
    for p in paths:
        fam = family_of(p)
        for steps in read_csv_sequences(Path(p)).values():
            out.append((fam, steps))
    return out


class NGramModel:
    """Trigram with back-off. We key contexts by family too, because the same
    pair can have different continuations across families. Back-off drops the
    family last, so an unseen family still falls back to family-agnostic stats."""

    def __init__(self):
        self.tri = defaultdict(Counter)   # (fam, a, b)   -> Counter(next)
        self.bi = defaultdict(Counter)    # (fam, b)      -> Counter(next)
        self.uni = Counter()              # next          -> count (global)

    def fit(self, samples):
        for fam, steps in samples:
            seq = [BOS, BOS] + steps + [EOS]
            for i in range(2, len(seq)):
                a, b, c = seq[i - 2], seq[i - 1], seq[i]
                self.tri[(fam, a, b)][c] += 1
                self.bi[(fam, b)][c] += 1
                self.uni[c] += 1
        return self

    def _ranked(self, fam, a, b) -> list[str]:
        """Return candidate next steps, most likely first, via back-off."""
        if (fam, a, b) in self.tri:
            return [s for s, _ in self.tri[(fam, a, b)].most_common()]
        if (fam, b) in self.bi:
            return [s for s, _ in self.bi[(fam, b)].most_common()]
        # family-agnostic bigram back-off (helps OOD)
        agg = Counter()
        for (f, bb), cnt in self.bi.items():
            if bb == b:
                agg.update(cnt)
        if agg:
            return [s for s, _ in agg.most_common()]
        return [s for s, _ in self.uni.most_common()]

    def topk(self, fam, a, b, k=5) -> list[str]:
        return self._ranked(fam, a, b)[:k]

    def next_step(self, fam, a, b) -> str:
        r = self._ranked(fam, a, b)
        return r[0] if r else EOS

    def complete(self, fam, prefix: list[str], max_new=80) -> list[str]:
        seq = [BOS, BOS] + list(prefix)
        out = []
        for _ in range(max_new):
            nxt = self.next_step(fam, seq[-2], seq[-1])
            if nxt == EOS:
                break
            out.append(nxt)
            seq.append(nxt)
        return out


# --------------------------------------------------------------------------- #
# Self-evaluation (mirrors the transformer's metrics so they're comparable)
# --------------------------------------------------------------------------- #

def evaluate(model, samples, cut_fracs=(0.6, 0.8)):
    top1 = top5 = total = 0
    exact = comp_total = 0
    for fam, steps in samples:
        # next-step: score every internal position
        seq = [BOS, BOS] + steps
        for i in range(2, len(seq)):
            gold = seq[i]
            ranked = model.topk(fam, seq[i - 2], seq[i - 1], k=5)
            total += 1
            if ranked and ranked[0] == gold:
                top1 += 1
            if gold in ranked:
                top5 += 1
        # completion at each cut fraction
        for frac in cut_fracs:
            cut = max(1, int(len(steps) * frac))
            pred = model.complete(fam, steps[:cut])
            gold_tail = steps[cut:]
            comp_total += 1
            if pred == gold_tail:
                exact += 1
    return {
        "next_step_top1": top1 / max(total, 1),
        "next_step_top5": top5 / max(total, 1),
        "completion_exact_match": exact / max(comp_total, 1),
    }


# --------------------------------------------------------------------------- #
# Submission writers (same format as predict.py, so scoring is apples-to-apples)
# --------------------------------------------------------------------------- #

def _split_pipe(s): return [x for x in s.split("|") if x != ""]


def read_valid(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            seq_col = next(c for c in d if "PARTIAL" in c.upper() or "SEQUENCE" in c.upper())
            fam_col = next(c for c in d if c.upper() == "FAMILY")
            id_col = next(c for c in d if "ID" in c.upper())
            rows.append({"id": d[id_col], "fam": d[fam_col].strip().lower(),
                         "steps": _split_pipe(d[seq_col])})
    return rows


def write_submission(model, rows, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "task1_nextstep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for r in rows:
            s = r["steps"]
            a, b = (s[-2] if len(s) >= 2 else BOS), (s[-1] if s else BOS)
            top = model.topk(r["fam"], a, b, k=5)
            while len(top) < 5:
                top.append("")
            w.writerow([r["id"], *top[:5]])
    with open(outdir / "task2_completion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for r in rows:
            pred = model.complete(r["fam"], r["steps"])
            w.writerow([r["id"], "|".join(pred)])
    print("wrote baseline submission ->", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--valid", help="official eval_input_valid.csv")
    ap.add_argument("--out", default="submission_baseline")
    args = ap.parse_args()

    model = NGramModel().fit(load(args.train))
    print(f"trained on {len(args.train)} file(s); "
          f"{len(model.tri)} trigram contexts, {len(model.uni)} unique steps")

    # self-eval on training distribution (upper bound for this baseline)
    train_metrics = evaluate(model, load(args.train))
    print("train:", {k: round(v, 4) for k, v in train_metrics.items()})

    if args.heldout:
        held_metrics = evaluate(model, load(args.heldout))
        print("heldout:", {k: round(v, 4) for k, v in held_metrics.items()})

    if args.valid:
        write_submission(model, read_valid(args.valid), args.out)


if __name__ == "__main__":
    main()

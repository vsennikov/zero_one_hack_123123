#!/usr/bin/env python3
"""
vocab.py — word-level tokenizer for process sequences.

1 step string = 1 token. No subword splitting. The whole vocabulary is ~120
step strings plus a handful of special tokens. Build it once from your data,
save vocab.json, reuse everywhere (training + eval + submission).

Special tokens:
  <PAD>  padding to equal length within a batch
  <BOS>  beginning of sequence (we put the FAMILY token right after it)
  <EOS>  end of sequence
  <UNK>  any step string unseen at build time (should be ~never if built well)
  <MOSFET> <IGBT> <IC>  family context tokens (extend if more families appear)
"""

from __future__ import annotations
import json
from pathlib import Path

SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
FAMILY_TOKENS = {"mosfet": "<MOSFET>", "igbt": "<IGBT>", "ic": "<IC>"}


class Vocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    # --- ids of special tokens (handy shortcuts) ---
    @property
    def pad(self): return self.stoi["<PAD>"]
    @property
    def bos(self): return self.stoi["<BOS>"]
    @property
    def eos(self): return self.stoi["<EOS>"]
    @property
    def unk(self): return self.stoi["<UNK>"]

    def __len__(self): return len(self.stoi)

    def encode(self, steps: list[str], family: str | None = None,
               add_bos=True, add_eos=True) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos)
        if family is not None:
            ids.append(self.stoi[FAMILY_TOKENS[family.lower()]])
        for s in steps:
            ids.append(self.stoi.get(s, self.unk))
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids: list[int], strip_special=True) -> list[str]:
        out = []
        specials = set(SPECIAL) | set(FAMILY_TOKENS.values())
        for i in ids:
            tok = self.itos.get(int(i), "<UNK>")
            if strip_special and tok in specials:
                continue
            out.append(tok)
        return out

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def build_vocab(step_iter) -> Vocab:
    """step_iter yields step strings (any order). Returns a Vocab.
    Sorted for determinism so the same data always gives the same ids."""
    unique = sorted(set(step_iter))
    tokens = list(SPECIAL) + list(FAMILY_TOKENS.values()) + unique
    stoi = {t: i for i, t in enumerate(tokens)}
    return Vocab(stoi)


if __name__ == "__main__":
    # Build from all *_variants.csv / reference csvs passed on the command line.
    import sys, csv
    steps = []
    for p in sys.argv[1:]:
        with open(p, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            col = "STEP" if "STEP" in r.fieldnames else r.fieldnames[-1]
            for row in r:
                steps.append(row[col])
    v = build_vocab(steps)
    v.save("vocab.json")
    print(f"vocab size = {len(v)}  (saved to vocab.json)")

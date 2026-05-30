#!/usr/bin/env python3
"""
predict.py — turn a trained model into the THREE submission files.

Reads the organizers' eval inputs and writes outputs in the format
eval_metrics.py expects:

  Task 1 (next-step):   reads  eval_input_valid.csv
                        writes task1_nextstep.csv   (EXAMPLE_ID, RANK_1..RANK_5)
  Task 2 (completion):  reads  eval_input_valid.csv
                        writes task2_completion.csv (EXAMPLE_ID, PREDICTED_SEQUENCE)
  Task 3 (anomaly):     reads  eval_input_anomaly.csv
                        writes task3_anomaly.csv    (EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE)

Run on a Leonardo compute node (inside a SLURM job), after training:
    python predict.py --ckpt $SCRATCH/proc-logic/ckpt_baseline/model.pt \
        --vocab vocab.json \
        --valid eval_input_valid.csv \
        --anomaly eval_input_anomaly.csv \
        --out $SCRATCH/proc-logic/submission

IMPORTANT — verify against the REAL eval_metrics.py when you get it.
Column names / order here follow the spec in generation_rules.md, but the
actual scoring script is handed out at the event. If a column name or the
empty-rule encoding differs, fix it here (it's a 5-minute change, all in the
write_* functions below). Budget time for this on the day.
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path

import torch

from vocab import Vocab, FAMILY_TOKENS
from model import make_model
import validator_tools as vt


# --------------------------------------------------------------------------- #
# Reading the official eval files (PIPE format, NOT the long training format)
# --------------------------------------------------------------------------- #

def _split_pipe(s: str) -> list[str]:
    return [x for x in s.split("|") if x != ""]


def read_valid(path: str) -> list[dict]:
    """eval_input_valid.csv columns:
       EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
    Returns dicts with parsed step list under 'steps'."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            d = dict(d)
            # tolerate small header naming differences
            seq_col = next(c for c in d if "SEQUENCE" in c.upper() or "PARTIAL" in c.upper())
            fam_col = next(c for c in d if c.upper() == "FAMILY")
            id_col = next(c for c in d if "ID" in c.upper())
            d["steps"] = _split_pipe(d[seq_col])
            d["family"] = d[fam_col].strip().lower()
            d["example_id"] = d[id_col]
            rows.append(d)
    return rows


def read_anomaly(path: str) -> list[dict]:
    """eval_input_anomaly.csv columns: EXAMPLE_ID, FAMILY, SEQUENCE"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            d = dict(d)
            seq_col = next(c for c in d if c.upper() == "SEQUENCE")
            fam_col = next(c for c in d if c.upper() == "FAMILY")
            id_col = next(c for c in d if "ID" in c.upper())
            d["steps"] = _split_pipe(d[seq_col])
            d["family"] = d[fam_col].strip().lower()
            d["example_id"] = d[id_col]
            rows.append(d)
    return rows


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device)
    model = make_model(ck["vocab_size"], preset=ck["config"],
                       block_size=ck["block_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["block_size"]


def _prompt_ids(vocab: Vocab, steps: list[str], family: str, device):
    ids = vocab.encode(steps, family=family, add_bos=True, add_eos=False)
    return torch.tensor(ids, device=device).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Task 1 — next-step prediction (top-5)
# --------------------------------------------------------------------------- #

def write_task1(model, vocab, rows, device, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for d in rows:
            idx = _prompt_ids(vocab, d["steps"], d["family"], device)
            top_ids = model.next_step_topk(idx, k=5)
            top_steps = vocab.decode(top_ids, strip_special=True)
            # pad to 5 in case some were special tokens and got stripped
            while len(top_steps) < 5:
                top_steps.append("")
            w.writerow([d["example_id"], *top_steps[:5]])
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# Task 2 — sequence completion (predict only the steps AFTER the cut)
# --------------------------------------------------------------------------- #

def write_task2(model, vocab, rows, device, out_path, max_new=80):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for d in rows:
            idx = _prompt_ids(vocab, d["steps"], d["family"], device)
            gen_ids = model.generate(idx, max_new_tokens=max_new, eos_id=vocab.eos)
            gen_steps = vocab.decode(gen_ids, strip_special=True)
            w.writerow([d["example_id"], "|".join(gen_steps)])
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# Task 3 — anomaly detection
# --------------------------------------------------------------------------- #
# We expose TWO scoring strategies. Pick per the rules / what you trained.
#
#   "validator"  : ground-truth rule check. Most accurate, but submitting the
#                  organizers' own checker may go against the spirit of the task
#                  (it tests whether the MODEL learned the logic). Use as an
#                  upper-bound baseline / sanity check, and clarify with mentors.
#   "model"      : score by the model's own likelihood of the sequence
#                  (perplexity). Lower likelihood => more anomalous. This is the
#                  honest "did the model learn it" signal. Needs a threshold,
#                  tuned on your labeled negatives.

@torch.no_grad()
def seq_nll(model, vocab, steps, family, device):
    """Average negative log-likelihood the model assigns to a full sequence.
    High = surprising = likely anomalous."""
    ids = vocab.encode(steps, family=family, add_bos=True, add_eos=True)
    idx = torch.tensor(ids, device=device).unsqueeze(0)
    x, y = idx[:, :-1], idx[:, 1:]
    logits, _ = model(x)
    logp = torch.log_softmax(logits, dim=-1)
    tok_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float(-tok_lp.mean().item())


def tune_threshold(model, vocab, labeled, device):
    """Pick the NLL threshold that maximizes F1 on a LABELED set.

    labeled: list of (family, steps, is_valid)  where is_valid in {0,1}.
    Returns (threshold, f1_at_threshold).

    This is the ONLY correct way to set the threshold. We never look at the
    eval set's score distribution to choose it -- doing so silently assumes a
    fixed anomaly rate (the median bug). We sweep candidate thresholds on data
    where we know the truth, freeze the best, and apply it blindly to eval.
    """
    scored = [(seq_nll(model, vocab, s, fam, device), v) for fam, s, v in labeled]
    # candidate thresholds = the observed NLL values (sweep all split points)
    cands = sorted({nll for nll, _ in scored})
    best_t, best_f1 = cands[0] if cands else 0.0, -1.0
    P = sum(1 for _, v in scored if v == 0)   # positives = anomalies (is_valid==0)
    for t in cands:
        # predict anomaly (is_valid=0) when nll >= t
        tp = sum(1 for nll, v in scored if nll >= t and v == 0)
        fp = sum(1 for nll, v in scored if nll >= t and v == 1)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def load_labeled(valid_paths, neg_per_rule=None, seed=0):
    """Build a labeled valid/invalid set for threshold tuning, from YOUR OWN
    data (never from the eval set). Uses generate_sequences + step-A negatives.
    valid_paths: csv files of known-valid sequences (long format)."""
    import random
    from train import load_all  # reuse family-aware loader
    samples = load_all(list(valid_paths))
    rng = random.Random(seed)
    labeled = [(fam, steps, 1) for fam, steps in samples]          # valid
    negs = vt.make_labeled_negatives([s for _, s in samples], rng=rng,
                                     per_rule=neg_per_rule)
    # negatives don't carry family; infer from the source sample order is messy,
    # so re-pair: each negative was built from some valid seq -> tag by majority
    # family of its steps is unreliable, so we just reuse the family distribution
    # by sampling. Simplest correct option: attach the family we know per source.
    # We rebuild negatives per (family, seq) to keep the family label exact:
    labeled_neg = []
    for fam, steps in samples:
        for rule in vt.ALL_RULES:
            case = vt.make_negative(steps, rule, rng)
            if case is not None:
                labeled_neg.append((fam, case.steps, 0))
                if neg_per_rule is None:
                    break  # one negative per sequence is enough for tuning
    labeled.extend(labeled_neg)
    return labeled


def write_task3(model, vocab, rows, device, out_path,
                strategy="model", threshold=None):
    if strategy == "model" and threshold is None:
        raise ValueError(
            "Task 3 'model' strategy needs a threshold tuned on LABELED data. "
            "Pass --task3_threshold, or --task3_labeled <valid.csv ...> so it "
            "can be tuned via tune_threshold(). Refusing to guess from the eval "
            "set (that silently assumes a fixed anomaly rate).")
    scores = {}
    if strategy == "model":
        for d in rows:
            scores[d["example_id"]] = seq_nll(model, vocab, d["steps"],
                                               d["family"], device)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for d in rows:
            if strategy == "validator":
                viols = vt.validate_sequence(d["steps"]) if hasattr(vt, "validate_sequence") \
                        else __import__("generate_sequences").validate_sequence(d["steps"])
                is_valid = 1 if not viols else 0
                score = 0.0 if is_valid else 1.0
                rule = "" if is_valid else viols[0].rule
            else:  # "model"
                s = scores[d["example_id"]]
                is_valid = 1 if s < threshold else 0
                score = s
                rule = ""  # model strategy doesn't attribute a specific rule
            w.writerow([d["example_id"], is_valid, f"{score:.6f}", rule])
    print("wrote", out_path)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--valid", help="eval_input_valid.csv (Tasks 1 & 2)")
    ap.add_argument("--anomaly", help="eval_input_anomaly.csv (Task 3)")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--task3_strategy", default="model",
                    choices=["model", "validator"])
    ap.add_argument("--task3_threshold", type=float, default=None)
    ap.add_argument("--task3_labeled", nargs="*", default=[],
                    help="valid-sequence csv(s) to tune the Task-3 threshold on "
                         "(labeled negatives are generated automatically)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(args.vocab)
    model, _ = load_model(args.ckpt, device)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  vocab={len(vocab)}")

    if args.valid:
        rows = read_valid(args.valid)
        print(f"valid rows: {len(rows)}")
        write_task1(model, vocab, rows, device, outdir / "task1_nextstep.csv")
        write_task2(model, vocab, rows, device, outdir / "task2_completion.csv")

    if args.anomaly:
        rows = read_anomaly(args.anomaly)
        print(f"anomaly rows: {len(rows)}")
        threshold = args.task3_threshold
        if args.task3_strategy == "model" and threshold is None and args.task3_labeled:
            labeled = load_labeled(args.task3_labeled)
            threshold, f1 = tune_threshold(model, vocab, labeled, device)
            print(f"[task3] tuned threshold = {threshold:.4f}  (labeled F1={f1:.3f})")
        write_task3(model, vocab, rows, device, outdir / "task3_anomaly.csv",
                    strategy=args.task3_strategy, threshold=threshold)


if __name__ == "__main__":
    main()

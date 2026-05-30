#!/usr/bin/env python3
"""
train.py — train the process-logic transformer + measure REAL process logic
(not just loss) during training via the validator from step A.

Run (on a Leonardo compute node, inside the SLURM job):
    python train.py --config baseline --epochs 30 \
        --data data/mosfet.csv data/igbt.csv data/ic.csv \
        --heldout data/heldout.csv --out $SCRATCH/proc-logic/ckpt

Key design choices (see chat for the why):
  * word-level vocab, 1 step = 1 token
  * FAMILY injected as a special token after <BOS>
  * two eval signals logged side by side:
      - val_loss / token accuracy        (standard)
      - pct_valid_continuations          (the process-logic curve, via validator)
  * held-out set generated with a DIFFERENT seed -> memorization detector
"""

from __future__ import annotations
import argparse, json, time, random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from vocab import Vocab, build_vocab, FAMILY_TOKENS
from model import make_model, PRESETS
from generate_sequences import read_csv_sequences
import validator_tools as vt


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def family_of(path: str) -> str:
    name = Path(path).name.lower()
    for fam in FAMILY_TOKENS:
        if fam in name:
            return fam
    raise ValueError(f"cannot infer family from filename {name!r}")


def load_all(paths: list[str]) -> list[tuple[str, list[str]]]:
    """Returns list of (family, steps)."""
    out = []
    for p in paths:
        fam = family_of(p)
        for steps in read_csv_sequences(Path(p)).values():
            out.append((fam, steps))
    return out


class SeqDataset(Dataset):
    """Each item: a full encoded sequence (BOS, FAMILY, steps..., EOS),
    padded/truncated to block_size. Targets are inputs shifted by one."""
    def __init__(self, samples, vocab: Vocab, block_size: int):
        self.vocab = vocab
        self.block_size = block_size
        self.data = []
        for fam, steps in samples:
            ids = vocab.encode(steps, family=fam, add_bos=True, add_eos=True)
            ids = ids[:block_size + 1]          # +1 because we shift for targets
            self.data.append(ids)

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
        x = ids[:-1]
        y = ids[1:]
        pad = self.vocab.pad
        # pad to block_size
        x = x + [pad] * (self.block_size - len(x))
        y = y + [pad] * (self.block_size - len(y))
        return torch.tensor(x[:self.block_size]), torch.tensor(y[:self.block_size])


# --------------------------------------------------------------------------- #
# Eval: the process-logic signal (validator in the loop)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def process_logic_eval(model, vocab, samples, device, n=100,
                       cut_frac=0.6, max_new=60):
    """Take n sequences, cut at cut_frac, let the model complete them,
    and measure the fraction of completions that introduce NO new violation
    (validated correctly via prefix+continuation gluing)."""
    model.eval()
    rng = random.Random(1234)
    picks = rng.sample(samples, min(n, len(samples)))
    valid = 0
    exact = 0
    for fam, steps in picks:
        cut = max(1, int(len(steps) * cut_frac))
        prefix_steps = steps[:cut]
        gold_tail = steps[cut:]
        # build prompt ids: BOS, FAMILY, prefix steps (no EOS yet)
        prompt = vocab.encode(prefix_steps, family=fam, add_bos=True, add_eos=False)
        idx = torch.tensor(prompt, device=device).unsqueeze(0)
        gen_ids = model.generate(idx, max_new_tokens=max_new, eos_id=vocab.eos)
        gen_steps = vocab.decode(gen_ids)            # strips specials/EOS
        rep = vt.validate_continuation(prefix_steps, gen_steps)
        valid += int(rep.valid)
        exact += int(gen_steps == gold_tail)
    return {"pct_valid_continuations": valid / len(picks),
            "exact_match": exact / len(picks)}


@torch.no_grad()
def loss_eval(model, loader, device, pad_id):
    model.eval()
    tot, n, correct, count = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y, pad_id=pad_id)
        tot += loss.item(); n += 1
        pred = logits.argmax(-1)
        mask = y != pad_id
        correct += (pred[mask] == y[mask]).sum().item()
        count += mask.sum().item()
    return {"val_loss": tot / max(n, 1), "token_acc": correct / max(count, 1)}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="baseline", choices=list(PRESETS))
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=1)   # epochs
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  config={args.config}")

    # vocab: load if exists else build from data
    if Path(args.vocab).exists():
        vocab = Vocab.load(args.vocab)
    else:
        steps = []
        for fam, s in load_all(args.data):
            steps += s
        vocab = build_vocab(steps)
        vocab.save(args.vocab)
    print(f"vocab size = {len(vocab)}")

    train_samples = load_all(args.data)
    held_samples = load_all(args.heldout) if args.heldout else []
    print(f"train sequences = {len(train_samples)}  heldout = {len(held_samples)}")

    train_ds = SeqDataset(train_samples, vocab, args.block_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, drop_last=True)
    held_dl = None
    if held_samples:
        held_ds = SeqDataset(held_samples, vocab, args.block_size)
        held_dl = DataLoader(held_ds, batch_size=args.batch_size)

    model = make_model(len(vocab), preset=args.config,
                       block_size=args.block_size, dropout=args.dropout).to(device)
    print(f"model params = {model.num_params()/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "metrics.jsonl"
    logf = log_path.open("w")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y, pad_id=vocab.pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        if (epoch + 1) % args.eval_every == 0:
            rec = {"epoch": epoch + 1, "step": step,
                   "train_loss": float(loss.item()),
                   "elapsed_s": round(time.time() - t0, 1)}
            # process-logic on TRAIN-distribution sequences
            rec.update({f"train_{k}": v for k, v in
                        process_logic_eval(model, vocab, train_samples, device).items()})
            # the memorization detector: same metric on UNSEEN heldout
            if held_samples:
                rec.update({f"held_{k}": v for k, v in
                            process_logic_eval(model, vocab, held_samples, device).items()})
                rec.update({f"held_{k}": v for k, v in
                            loss_eval(model, held_dl, device, vocab.pad).items()})
            print(json.dumps(rec))
            logf.write(json.dumps(rec) + "\n"); logf.flush()

    torch.save({"model": model.state_dict(),
                "config": args.config,
                "vocab_size": len(vocab),
                "block_size": args.block_size},
               outdir / "model.pt")
    logf.close()
    print(f"saved -> {outdir/'model.pt'}")


if __name__ == "__main__":
    main()

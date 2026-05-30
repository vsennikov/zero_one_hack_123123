#!/bin/bash
#SBATCH --job-name=proc-logic
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc      # hackathon reservation (1 node per team)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1            # 1 GPU baseline; fair-share scales w/ this
#SBATCH --mem=120GB                  # 120GB * gpus-per-task
#SBATCH --cpus-per-task=8            # 8 * gpus-per-task
#SBATCH --time=01:00:00              # plenty for a ~7M model; raise for xl
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# IMPORTANT (from the hackathon intro slides):
#  * Compute nodes have NO internet. Install everything on a LOGIN node first
#    (pixi / venv) and only RUN here. Do not pip-install inside this job.
#  * Use $SCRATCH for data + checkpoints. $WORK is off-limits for the hackathon.
#  * vocab.json must already exist (build it once on the login node).
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

PROJ=$SCRATCH/proc-logic
cd "$PROJ"

# --- activate the environment you prepared on the login node ---
# Option A: pixi  (recommended per slides)
#   export RUN="/path/to/pixi run --as-is --manifest-path $PROJ/pixi.toml"
# Option B: a prebuilt venv
#   source $PROJ/venv/bin/activate ; RUN=""
RUN="${RUN:-}"

$RUN python3 train.py \
    --config baseline \
    --data data/mosfet.csv data/igbt.csv data/ic.csv \
    --heldout data/heldout_mosfet.csv data/heldout_igbt.csv data/heldout_ic.csv \
    --vocab vocab.json \
    --out "$PROJ/ckpt_baseline" \
    --epochs 30 \
    --batch_size 64

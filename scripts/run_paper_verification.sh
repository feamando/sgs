#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-shot paper verification for the VSP negative-result paper (JMLR target).
#
# Runs the two blocking controls a main-conference reviewer will demand:
#   C1  SCRAMBLED-BUNDLE control  (grounded init vs a permuted-bundle init)
#   C2  40k REPRODUCTION           (>=3 seeds of grounded vs baseline at full compute)
#
# Everything is idempotent: a checkpoint or result that already exists is
# skipped, so you can re-run after an interruption and it resumes. When all
# results are present it runs scripts/verify_paper.py and, unless --no-push,
# commits + pushes the new result JSONs.
#
# Usage (from repo root, on the 4090 box):
#   bash scripts/run_paper_verification.sh                 # full run + push
#   bash scripts/run_paper_verification.sh --no-push       # compute only
#   bash scripts/run_paper_verification.sh --quick         # 2k-only smoke (no 40k)
#
# GPU budget: 3x 40k runs (~10h each) dominate. The 2k control runs are ~1.3h.
# Expect ~30h wall on a single 4090 for the full grid; --quick is ~4h.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root
ROOT="$(pwd)"

TOKENS="data/wiki_vsps"
VOCAB="data/vsps/vocab.json"
GLOVE="data/glove.6B.300d.txt"
SUBWORD="data/hertz12_data/tokenizer.model"
PAIRS="scripts/assets/disambig_pairs.json"

PUSH=1
QUICK=0
SEEDS_40K="0 1 2"                # C2: 3 seeds at full compute (s0 already exists)
for arg in "$@"; do
  case "$arg" in
    --no-push) PUSH=0 ;;
    --quick)   QUICK=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { echo -e "\n[verify $(date +%H:%M:%S)] $*"; }

# train <arm-flag> <regime-steps> <freeze-flag> <seed> <ckpt-dir>
# arm-flag: "" (grounded) | "--random-init" | "--shuffle-bundles"
train_one() {
  local armflag="$1" steps="$2" freeze="$3" seed="$4" dir="$5"
  if [[ -f "$dir/final.pt" ]]; then
    log "SKIP train (exists): $dir/final.pt"
    return
  fi
  log "TRAIN $dir  (steps=$steps seed=$seed flags='$armflag $freeze')"
  python scripts/train_planck2.py --tokens "$TOKENS" --vocab "$VOCAB" \
    --opt-steps "$steps" --seed "$seed" $armflag $freeze --save-dir "$dir"
}

# eval <ckpt-dir> <out-json>
eval_one() {
  local dir="$1" out="$2"
  if [[ -f "$out" ]]; then
    log "SKIP eval (exists): $out"
    return
  fi
  if [[ ! -f "$dir/final.pt" ]]; then
    log "WARN: no checkpoint at $dir, cannot eval -> $out"
    return
  fi
  log "EVAL $dir -> $out"
  python scripts/eval_disambiguation.py --checkpoint "$dir/final.pt" \
    --vocab "$VOCAB" --glove "$GLOVE" --subword-model "$SUBWORD" \
    --pairs "$PAIRS" --out "$out"
}

# ── C1: scrambled-bundle control (grounded is s0, already done; add scrambled) ──
# freeze-forever so the init actually persists (matches the 2k grounded/baseline arm)
log "=== C1: scrambled-bundle control ==="
train_one "--shuffle-bundles" 2000 "--freeze-vp-forever" 0 "checkpoints/planck2_scrambled_2k_s0"
eval_one  "checkpoints/planck2_scrambled_2k_s0" "results/disambig_scrambled_2k_s0.json"

if [[ "$QUICK" -eq 0 ]]; then
  train_one "--shuffle-bundles" 40000 "" 0 "checkpoints/planck2_scrambled_40k_s0"
  eval_one  "checkpoints/planck2_scrambled_40k_s0" "results/disambig_scrambled_40k_s0.json"

  # ── C2: 40k reproduction. s0 exists as legacy disambig_{vsp,baseline}.json ──
  log "=== C2: 40k reproduction (seeds $SEEDS_40K) ==="
  for s in $SEEDS_40K; do
    if [[ "$s" -eq 0 ]]; then
      # seed 0 already trained+evaled under legacy names; skip retrain, alias eval
      [[ -f "results/disambig_vsp_40k_s0.json" ]]      || cp results/disambig_vsp.json      results/disambig_vsp_40k_s0.json 2>/dev/null || true
      [[ -f "results/disambig_baseline_40k_s0.json" ]] || cp results/disambig_baseline.json results/disambig_baseline_40k_s0.json 2>/dev/null || true
      continue
    fi
    train_one "" 40000 "" "$s" "checkpoints/planck2_vsp_40k_s$s"
    eval_one  "checkpoints/planck2_vsp_40k_s$s" "results/disambig_vsp_40k_s$s.json"
    train_one "--random-init" 40000 "" "$s" "checkpoints/planck2_baseline_40k_s$s"
    eval_one  "checkpoints/planck2_baseline_40k_s$s" "results/disambig_baseline_40k_s$s.json"
  done
else
  log "QUICK mode: skipping all 40k runs (C2 and scrambled-40k)"
fi

# ── verdict ──
log "=== verification summary ==="
python scripts/verify_paper.py | tee results/paper_verification_summary.txt

# ── push ──
if [[ "$PUSH" -eq 1 ]]; then
  log "=== git push ==="
  git add results/disambig_scrambled_*.json results/disambig_vsp_40k_*.json \
          results/disambig_baseline_40k_*.json results/paper_verification_summary.txt 2>/dev/null || true
  if git diff --cached --quiet; then
    log "nothing new to commit"
  else
    git commit -q -m "paper verification: scrambled-bundle control + 40k reproduction

Automated run of scripts/run_paper_verification.sh. Adds the two blocking
controls for the JMLR submission: scrambled-bundle init (C1) and multi-seed 40k
reproduction of the -3.8 (C2). See results/paper_verification_summary.txt."
    git push
    log "pushed."
  fi
else
  log "--no-push: leaving results uncommitted."
fi

log "DONE."

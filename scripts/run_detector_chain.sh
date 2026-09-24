#!/usr/bin/env bash
# Serial driver for the three detector sources.
#
# Two reasons this exists rather than typing the commands each time:
#
# 1. SERIAL, not parallel. Running local_fold1 alongside external_pretrain put
#    both at 100% GPU and ~127 s/epoch, against 32 s/epoch for fold0 on its own.
#    Two concurrent trainings on one 8.5 GB card buy nothing and double the
#    chance of the host-RAM exhaustion that already killed one fold.
# 2. DETACHED. The previous chain died mid-epoch when the session's shell tasks
#    were torn down. Launched with nohup, the chain outlives the session.
#
# Each stage appends to one log so the ordering is auditable afterwards.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/c/Users/liujunqing/anaconda3/envs/pytorch_gpu/python.exe
export PYTHONIOENCODING=utf-8
LOG=artifacts/experiments/rescue_external_20260922/detectors/chain.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

stage() {
  local name="$1"; shift
  say "BEGIN $name"
  "$PY" scripts/train_morph_detector_folds.py "$@" >>"$LOG" 2>&1
  local rc=$?
  say "END   $name rc=$rc"
  return $rc
}

say "chain start"
stage "local folds 1-4"        --source local              --folds 1,2,3,4
stage "external pretrain"      --source external
stage "external_then_local"    --source external_then_local --folds 0,1,2,3,4
say "chain done"

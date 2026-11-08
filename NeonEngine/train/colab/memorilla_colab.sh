#!/usr/bin/env bash
# Memorilla on a Colab T4, end to end (docs: shaders/ML/docs/memorilla_plan.md, P5):
#   1. setup_colab.sh (Vulkan SDK + NVIDIA Vulkan ICD + NeonML build)
#   2. TinyStories with an 8k BPE tokenizer (prepare_data.py)
#   3. the t4 gpt-oss decoder, from scratch, with MXFP4 QAT
#   4. Memorilla data + self encoder, module training, and the four evaluations
#      (closed book, untrained module, noise documents, trained module)
# Every stage is skipped when its output exists, so a rerun continues.
#
#   MODE=smoke bash memorilla_colab.sh <src dir>    # minutes: tiny sizes, checks the pipeline
#   MODE=full  bash memorilla_colab.sh <src dir>    # the experiment (set DEC_STEPS / MEM_STEPS)
#
# Writes everything under $WORK (default /content/mem_$MODE) and a summary to $WORK/results.txt.
set -euo pipefail

SRC="$(cd "${1:?usage: memorilla_colab.sh <NeonEngine/NeonEngine dir>}" && pwd)"
MODE="${MODE:-smoke}"
WORK="${WORK:-/content/mem_$MODE}"
mkdir -p "$WORK"
say() { printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ "$MODE" = smoke ]; then
  TRAIN_TOKENS=3000000; TOK_MB=10; DEC_STEPS="${DEC_STEPS:-30}"; DEC_BATCH=8; DEC_SEQ=256
  MEM_TRAIN=2000; MEM_VAL=256; MEM_STEPS="${MEM_STEPS:-40}"; EVAL_BATCHES=4
else
  TRAIN_TOKENS="${TRAIN_TOKENS:-120000000}"; TOK_MB=50; DEC_STEPS="${DEC_STEPS:-6000}"; DEC_BATCH=16; DEC_SEQ=512
  MEM_TRAIN=200000; MEM_VAL=2000; MEM_STEPS="${MEM_STEPS:-4000}"; EVAL_BATCHES=62
fi

# ---- 1. build ---------------------------------------------------------------
if [ ! -f /content/neon_env.sh ]; then
  say "setup (Vulkan SDK, NVIDIA ICD, NeonML)"
  bash "$SRC/train/colab/setup_colab.sh" "$SRC"
fi
# shellcheck disable=SC1091
source /content/neon_env.sh
SH="$NEON_SRC/shaders/ML"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# ---- 2. data ----------------------------------------------------------------
DATA="$WORK/ts_bpe8k"
if [ ! -f "$DATA/meta.json" ]; then
  say "TinyStories, 8k BPE ($TRAIN_TOKENS training tokens)"
  python -c "import tiktoken, datasets" 2>/dev/null || pip -q install tiktoken datasets
  python "$NEON_SRC/train/prepare_data.py" --out "$DATA" --vocab-size 8192 --tokenizer-sample-mb "$TOK_MB" \
    --max-train-tokens "$TRAIN_TOKENS" --max-val-tokens 1000000 --streaming
fi

# ---- 3. decoder -------------------------------------------------------------
DEC="$WORK/decoder"
if ! grep -q "checkpoint: step $DEC_STEPS " "$WORK/decoder.log" 2>/dev/null; then
  say "t4 decoder: $DEC_STEPS steps of $DEC_BATCH x $DEC_SEQ tokens (MXFP4 QAT from 80%)"
  $NEON_ML --gpt-oss-train train --config "$NEON_SRC/train/configs/t4.json" --data "$DATA" --out "$DEC" \
    --shaders "$SH" --batch "$DEC_BATCH" --seq "$DEC_SEQ" --steps "$DEC_STEPS" --warmup $((DEC_STEPS / 20 + 1)) \
    --lr 1e-3 --min-lr 1e-4 --schedule wsd --eval-every $((DEC_STEPS / 5 + 1)) --eval-batches 20 --log-every 50 \
    --ckpt-minutes 20 --no-export 2>&1 | tee -a "$WORK/decoder.log"
fi

# ---- 4. Memorilla -----------------------------------------------------------
MDATA="$WORK/mem_data"
if [ ! -f "$MDATA/val_queries.nmem" ]; then
  say "Memorilla data + self encoder"
  python "$NEON_SRC/train/prepare_memorilla_data.py" --data "$DATA" --out "$MDATA" --decoder "$DEC" \
    --exe "$NEON_ML" --train-examples "$MEM_TRAIN" --val-examples "$MEM_VAL" 2>&1 | tee "$WORK/mem_data.log"
fi
MRUN="$WORK/mem_run"
if [ ! -f "$MRUN/module/memorilla.safetensors" ]; then
  say "Memorilla module: $MEM_STEPS steps"
  $NEON_ML --memorilla train --decoder "$DEC" --data "$MDATA" --out "$MRUN" \
    --mem-config "$NEON_SRC/train/configs/memorilla.json" --shaders "$SH" --batch 32 --seq 128 \
    --steps "$MEM_STEPS" --lr 3e-4 --eval-every $((MEM_STEPS / 8 + 1)) --eval-batches "$EVAL_BATCHES" \
    --log-every 50 --ckpt-minutes 20 2>&1 | tee -a "$WORK/mem_train.log"
fi
say "evaluations"
: > "$WORK/results.txt"
{
  echo "mode $MODE, decoder $DEC_STEPS steps, module $MEM_STEPS steps, $(date)"
  grep -E "^step $DEC_STEPS " "$WORK/decoder.log" | tail -1 || true
} >> "$WORK/results.txt"
for a in closed untrained noise none; do
  $NEON_ML --memorilla eval --decoder "$DEC" --data "$MDATA" --module "$MRUN/module" --shaders "$SH" \
    --batch 32 --seq 128 --eval-batches "$EVAL_BATCHES" --ablation "$a" 2>&1 | grep "eval (" | tee -a "$WORK/results.txt"
done
say "done"
cat "$WORK/results.txt"

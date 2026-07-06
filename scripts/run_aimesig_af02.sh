#!/bin/bash
cd ~/decodeatt
M=/root/data1/cyb/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
rm -f results/aimesig_af02_done.flag
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  [ $((t-u)) -gt 25000 ] && break
  sleep 60
done
echo "START $(date +%s)" > results/aimesig_af02.log
PYTHONPATH=. .venv/bin/python scripts/step3_eval.py --model $M --dataset aime --n 10 \
  --anchor --backend rkv --budget 1024 --recent 64 --max-new 16384 \
  --anchor-frac 0.2 \
  --out results/step3_aime_sig_af02.json >> results/aimesig_af02.log 2>&1
echo "END $(date +%s)" >> results/aimesig_af02.log
touch results/aimesig_af02_done.flag

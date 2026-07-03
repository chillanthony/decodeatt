#!/bin/bash
cd ~/decodeatt
M=/root/data1/cyb/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
rm -f results/nll_af02_done.flag
while [ ! -f results/aimesig_af02_done.flag ]; do
  sleep 120
done
while :; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  [ $((t-u)) -gt 25000 ] && break
  sleep 60
done
PYTHONPATH=. .venv/bin/python scripts/step3_nll.py --model $M --budget 1024 \
  --anchor-frac 0.2 --max-traces 8 \
  --out results/step3_nll_af02.json > results/step3_nll_af02.log 2>&1
touch results/nll_af02_done.flag

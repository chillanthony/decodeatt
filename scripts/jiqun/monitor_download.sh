#!/usr/bin/env bash
set -euo pipefail

HF_HOME="${HF_HOME:-/home/ma-user/work/bucket-wulan-green/chenyanbo/hf_cache}"
INTERVAL="${INTERVAL:-10}"
SAMPLES="${SAMPLES:-60}"

size_bytes() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo 0
    return
  fi
  if du -sb "$path" >/dev/null 2>&1; then
    du -sb "$path" | awk '{print $1}'
  else
    awk -v kb="$(du -sk "$path" | awk '{print $1}')" 'BEGIN {print kb * 1024}'
  fi
}

file_count() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo 0
    return
  fi
  find "$path" -type f 2>/dev/null | wc -l | awk '{print $1}'
}

format_bytes() {
  awk -v bytes="$1" 'BEGIN {
    split("B KB MB GB TB", unit, " ");
    value = bytes;
    i = 1;
    while (value >= 1024 && i < 5) {
      value /= 1024;
      i++;
    }
    printf "%.2f%s", value, unit[i];
  }'
}

echo "[monitor] hf_home=$HF_HOME"
echo "[monitor] interval=${INTERVAL}s samples=$SAMPLES"
echo "[monitor] columns: time total_size delta_size avg_speed file_count"

prev_size="$(size_bytes "$HF_HOME")"
prev_time="$(date +%s)"

for ((i = 1; i <= SAMPLES; i++)); do
  sleep "$INTERVAL"
  now_time="$(date +%s)"
  now_size="$(size_bytes "$HF_HOME")"
  files="$(file_count "$HF_HOME")"
  elapsed=$((now_time - prev_time))
  delta=$((now_size - prev_size))
  if (( delta < 0 )); then
    delta=0
  fi
  speed="$(awk -v bytes="$delta" -v sec="$elapsed" 'BEGIN {
    if (sec <= 0) {
      printf "0.00MB/s";
    } else {
      printf "%.2fMB/s", bytes / sec / 1024 / 1024;
    }
  }')"
  printf "[monitor] %s total=%s delta=%s speed=%s files=%s\n" \
    "$(date '+%F %T')" \
    "$(format_bytes "$now_size")" \
    "$(format_bytes "$delta")" \
    "$speed" \
    "$files"
  prev_size="$now_size"
  prev_time="$now_time"
done

echo "[monitor] meaning:"
echo "[monitor] - speed > 0 means the cache is still growing."
echo "[monitor] - speed near 0 for many samples means the download may be stalled or writing elsewhere."
echo "[monitor] - files increasing but speed low usually means many small metadata/shard files."

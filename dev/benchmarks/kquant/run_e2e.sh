#!/bin/bash
# ABAB end-to-end comparison: A = splash packed-q4 package, B = GGUF K-quant package. Same binary, same server, 60 s idle gap between runs.
set -u
cd ~/dev/splash
LOG=~/dev/q4k-m5/logs/e2e_results_v2.jsonl; : > $LOG
A=~/dev/splash/install/models/incoai/Qwen3.8-27B-Splash; B=~/dev/q4k-m5/pkg-kq
stop_server() { pkill -f "server/server.py" 2>/dev/null; pkill -x splash 2>/dev/null; sleep 3; }
run_one() {  # $1 = tag, $2 = package root, $3 = run index
  local tag=$1 pkg=$2 idx=$3
  stop_server
  nohup .venv/bin/python -u server/server.py $pkg/target $pkg/draft --tokenizer $pkg/tokenizer --model incoai/Qwen3.8-27B-Splash --binary build/splash --host 127.0.0.1 --port 8011 --max-memory auto --max-context auto --no-webui > ~/dev/q4k-m5/logs/server_${tag}_${idx}.log 2>&1 &
  for i in $(seq 1 200); do sleep 2; curl -s -m 2 http://127.0.0.1:8011/ready >/dev/null 2>&1 && break; pgrep -f "server/server.py" >/dev/null || { echo "server died ($tag)"; return 1; }; done
  echo "$(date +%T) $tag run $idx ready"
  ~/dev/q4k-m5/venv/bin/python ~/dev/q4k-m5/bench_e2e.py --tag "${tag}#${idx}" >> $LOG
  stop_server
}
run_one splash "$A" 1; sleep 60
run_one kquant "$B" 1; sleep 60
run_one splash "$A" 2; sleep 60
run_one kquant "$B" 2
echo "$(date +%T) done"

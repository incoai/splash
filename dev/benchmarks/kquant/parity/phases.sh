#!/bin/bash
# Sequential end-to-end phases on the M5 (one engine on the GPU at a time, 60 s gap before every perf run).
# usage: phases.sh <phase>   phases: llama-bench llama-tasks llama-stop score splash-gguf splash-uniform
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
P=/Users/liang2kl/dev/q4k-m5/parity; L=/Users/liang2kl/dev/q4k-m5/logs; S=/Users/liang2kl/dev/splash
PY=$S/.venv/bin/python
TOK="$S/install/models/incoai-internal/Qwen3.8-27B-Splash-GGUF:UD-Q4_K_M/tokenizer/tokenizer.json"
gap() { echo "-- 60 s gap $(date)"; sleep 60; }
wait_ready() { for i in $(seq 1 600); do curl -sf -m 2 "$1" >/dev/null 2>&1 && return 0; sleep 2; done; echo "server not ready: $1"; return 1; }
case "$1" in
  llama-bench)
    gap; $PY $P/bench_driver.py bench --url http://127.0.0.1:8080 --model gguf --engine llama --label llama.cpp-UD-Q4_K_M --prompts $P/perf_prompts.json --out $P/bench.jsonl --max-tokens 512 --tokenizer "$TOK" ;;
  llama-tasks)
    $PY $P/bench_driver.py tasks --url http://127.0.0.1:8080 --model gguf --engine llama --label llama.cpp-UD-Q4_K_M --tasks $P/tasks.json --out $P/tasks.jsonl --max-tokens 768 --tokenizer "$TOK" --concurrency ${2:-4} ;;
  llama-start)
    cd $P && (nohup ./llama_server.sh ${2:-1} > $L/llama_server_np${2:-1}.log 2>&1 < /dev/null &)
    T0=$(date +%s); wait_ready http://127.0.0.1:8080/health || exit 1; echo "llama-server (${2:-1} slots) ready after $(( $(date +%s) - T0 )) s" ;;
  llama-stop)
    pkill -f "llama-server -m" ; sleep 3; pgrep -f "llama-server -m" >/dev/null && echo "llama-server still running" || echo "llama-server stopped" ;;
  score)
    cd $S && $S/build/engine-tests/score-prefix $S/build/splash.metallib "$S/install/models/incoai-internal/Qwen3.8-27B-Splash-GGUF:UD-Q4_K_M" < $P/jobs.tsv > $P/splash_logits.txt 2> $L/score_prefix.log; echo "score exit $?"; tail -2 $L/score_prefix.log
    python3 $P/parity_driver.py compare --reference $P/reference.json --logits $P/splash_logits.txt --out $P/parity_metrics.json | tee $P/parity_report.md ;;
  splash-gguf|splash-uniform)
    if [ "$1" = splash-gguf ]; then MODEL="incoai-internal/Qwen3.8-27B-Splash-GGUF:UD-Q4_K_M"; LABEL=splash-UD-Q4_K_M; else MODEL="incoai/Qwen3.8-27B-Splash"; LABEL=splash-uniform-q4; fi
    ROOT="$S/install/models/$MODEL"
    [ -e "$ROOT/manifest.json" ] || (cd $S && $PY install/models.py --models install/models --model "$MODEL" prepare 2>&1 | tail -2)
    cd $S && (nohup $PY -u server/server.py "$ROOT/target" "$ROOT/draft" --tokenizer "$ROOT/tokenizer" --model "$MODEL" --binary build/splash --host 127.0.0.1 --port 8011 --max-memory auto --max-context auto --no-webui > $L/splash_$LABEL.log 2>&1 < /dev/null &)
    T0=$(date +%s); wait_ready http://127.0.0.1:8011/ready || exit 1; echo "splash $LABEL ready after $(( $(date +%s) - T0 )) s"
    grep -i -E "load|ready|weights|seconds" $L/splash_$LABEL.log | head -5 | cut -c1-140
    gap; $PY $P/bench_driver.py bench --url http://127.0.0.1:8011 --model "$MODEL" --engine splash --label $LABEL --prompts $P/perf_prompts.json --out $P/bench.jsonl --max-tokens 512 --tokenizer "$TOK"
    $PY $P/bench_driver.py tasks --url http://127.0.0.1:8011 --model "$MODEL" --engine splash --label $LABEL --tasks $P/tasks.json --out $P/tasks.jsonl --max-tokens 768 --tokenizer "$TOK" --concurrency ${2:-4}
    pkill -f "server/server.py"; sleep 5; pgrep -f "server/server.py" >/dev/null && echo "splash still running" || echo "splash stopped" ;;
  *) echo "unknown phase $1"; exit 2 ;;
esac
echo "== PHASE $1 DONE $(date)"

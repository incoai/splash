#!/bin/bash
# Full sequential campaign: waits for the llama.cpp reference extraction, then runs every GPU-exclusive phase in order.
P=/Users/liang2kl/dev/q4k-m5/parity; L=/Users/liang2kl/dev/q4k-m5/logs
until grep -q "^wrote" $L/parity_reference.log 2>/dev/null; do if ! pgrep -f "parity_driver.py reference" >/dev/null; then echo "reference extraction is not running and did not finish"; exit 1; fi; sleep 10; done
echo "== reference done $(date)"; tail -1 $L/parity_reference.log
$P/phases.sh llama-bench   > $L/phase_llama_bench.log 2>&1
$P/phases.sh llama-stop    >> $L/phase_llama_bench.log 2>&1
$P/phases.sh llama-start 4 > $L/phase_llama_tasks.log 2>&1
$P/phases.sh llama-tasks 4 >> $L/phase_llama_tasks.log 2>&1
$P/phases.sh llama-stop    >> $L/phase_llama_tasks.log 2>&1
$P/phases.sh score         > $L/phase_score.log 2>&1
$P/phases.sh splash-gguf 4 > $L/phase_splash_gguf.log 2>&1
$P/phases.sh splash-uniform 4 > $L/phase_splash_uniform.log 2>&1
echo "== CHAIN DONE $(date)"

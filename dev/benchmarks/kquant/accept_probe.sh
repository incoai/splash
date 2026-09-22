#!/bin/bash
# Acceptance-rate probe: same greedy prompts on each package, read the engine's drafted/accepted counters from /status.
set -u; cd ~/dev/splash
A=~/dev/splash/install/models/incoai/Qwen3.8-27B-Splash; B=~/dev/q4k-m5/pkg-kq; OUT=~/dev/q4k-m5/logs/accept_probe.jsonl; : > $OUT
stop_server() { pkill -f "server/server.py" 2>/dev/null; pkill -x splash 2>/dev/null; sleep 3; }
probe() { local tag=$1 pkg=$2; stop_server
  nohup .venv/bin/python -u server/server.py $pkg/target $pkg/draft --tokenizer $pkg/tokenizer --model incoai/Qwen3.8-27B-Splash --binary build/splash --host 127.0.0.1 --port 8011 --max-memory auto --max-context auto --no-webui > ~/dev/q4k-m5/logs/server_accept_$tag.log 2>&1 &
  for i in $(seq 1 200); do sleep 2; curl -s -m 2 http://127.0.0.1:8011/ready >/dev/null 2>&1 && break; done
  for prompt in "Write a long, detailed essay about the history of the Roman Empire, covering its founding, expansion, culture, and fall. Do not stop early." "Explain how a transformer language model works, in detail, for a graduate student. Cover attention, feed-forward layers, training and inference." "Write a detailed story about a lighthouse keeper who discovers something unexpected. Make it long."; do
    curl -s -m 600 http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"incoai/Qwen3.8-27B-Splash\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":400,\"temperature\":0}" > /dev/null
  done
  curl -s http://127.0.0.1:8011/status | ~/dev/q4k-m5/venv/bin/python -c "import json,sys; d=json.load(sys.stdin); m=d.get('metrics',d); keys=['decode_output_tokens','drafted_tokens','accepted_draft_tokens','draft_acceptance_rate','decode_tokens_per_second','decode_wall_ms']; print(json.dumps({'tag':'$tag', **{k:m.get(k) for k in keys}}))" >> $OUT
  stop_server; }
probe splash "$A"; sleep 60; probe kquant "$B"; echo done >> $OUT

#!/bin/zsh
# Serve the DSpark drafter with the context in the paged pool and with it in the private
# arena, on the same build, and require both to draft and to emit the same greedy tokens.
# Same output is the point: the rework moves where the context lives, not what it computes.
set -u
WT=$HOME/DSpark-data/wt/dspark-paged
R=$HOME/DSpark-data/results/m5max-dspark-paged-smoke
PORT=8137
T=$HOME/DSpark-data/models/target/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
D=$HOME/DSpark-data/models/draft/3457dff1417cb84927f6098a5fcb7cee85c934b7
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/.local/dev/bin:$HOME/DSpark-data/bin:$PATH"
export VLLM_METAL_BUILD_FROM_SOURCE=1 HF_HUB_OFFLINE=1 UV_NO_MODIFY_PATH=1 PYTHONPATH=$WT
mkdir -p $R; S=$R/summary.txt
source $HOME/DSpark-data/wt/dspark-pr/.venv-vllm-metal/bin/activate
cd $WT || exit 2
echo "=== $(date '+%F %T') paged smoke on $(git rev-parse --short HEAD)" | tee -a $S

serve() {  # serve LABEL PAGED
  local label=$1 paged=$2
  # Refuse to start on an occupied port rather than silently measuring a survivor.
  if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "=== $label ABORT: port $PORT already in use" | tee -a $S
    return 1
  fi
  env VLLM_METAL_DSPARK_PAGED_CONTEXT=$paged PYTHONPATH=$WT \
    vllm serve "$T" --port $PORT --host 127.0.0.1 --generation-config vllm \
    --gpu-memory-utilization 0.4 --max-model-len 2048 --max-num-seqs 8 \
    --no-enable-prefix-caching \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$D\",\"num_speculative_tokens\":4}" \
    > $R/$label.server.log 2>&1 &
  SERVER_PID=$!
  local i=0
  # Health alone is not proof THIS server is up: a survivor from an earlier run
  # still holding the port answers /health, and the arm then measures the wrong
  # process. Wait for this server's own log to name the backend it built.
  while ! { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
            && grep -q "DSpark draft context:" $R/$label.server.log; }; do
    sleep 5; i=$((i+5))
    kill -0 $SERVER_PID 2>/dev/null || {
      echo "=== $label DID NOT START: $(grep -hiE 'error|Traceback' $R/$label.server.log | tail -2 | cut -c1-200)" | tee -a $S
      return 1; }
    if [ $i -ge 900 ]; then
      kill $SERVER_PID
      echo "=== $label timeout" | tee -a $S
      return 1
    fi
  done
  # A while loop returns the status of the last command its body ran. The bounds
  # check above is false on every normal iteration, so without this the function
  # reported failure for a server that had come up fine -- and an arm whose health
  # check passed on the first try (because a previous server still held the port)
  # returned success, so the labels came out inverted.
  return 0
}
stop() {
  # The API server forks the engine; killing only the parent leaves the engine
  # holding the GPU and a pgrep wait then spins forever. Kill the tree, and bound
  # the wait so a survivor is reported rather than hanging the run.
  pkill -P $SERVER_PID 2>/dev/null
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  local waited=0
  while pgrep -f 'VLLM::EngineCore' >/dev/null; do
    sleep 2; waited=$((waited+2))
    [ $waited -ge 30 ] && { pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; break; }
  done
  sleep 4
}
drafted() {
  curl -s "http://127.0.0.1:$PORT/metrics" \
    | awk '/^vllm:spec_decode_num_draft_tokens_total/{d=$2} /^vllm:spec_decode_num_accepted_tokens_total/{a=$2} END {printf "%d/%d", a+0, d+0}'
}
ask() {  # three prompts, greedy, token ids so the comparison is exact
  for p in "The capital of France is" "def fibonacci(n):" "Explain why the sky is blue:"; do
    curl -sS "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$T\",\"prompt\":\"$p\",\"temperature\":0,\"max_tokens\":48,\"return_token_ids\":true}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0].get("token_ids"))'
  done
}

for arm in "arena 0" "paged 1"; do
  label=${arm% *}; paged=${arm#* }
  echo "=== $(date '+%T') arm start: label=$label paged=$paged" | tee -a $S
  if serve $label $paged; then
    before=$(drafted)
    ask > $R/$label.tokens
    backend=$(grep -h 'DSpark draft context:' $R/$label.server.log | tail -1 | sed -E 's/.*draft context: //')
    echo "=== $label healthy | accepted/drafted $before -> $(drafted) | $backend" | tee -a $S
    stop
  else
    echo "=== $(date '+%T') arm $label did not serve" | tee -a $S
  fi
done

if [ -s $R/arena.tokens ] && [ -s $R/paged.tokens ]; then
  if diff -q $R/arena.tokens $R/paged.tokens >/dev/null; then
    echo "=== TOKENS IDENTICAL across both context backends" | tee -a $S
  else
    echo "=== TOKENS DIFFER -- the rework changed what the drafter computes" | tee -a $S
    diff $R/arena.tokens $R/paged.tokens | head -8 | tee -a $S
  fi
fi
echo "=== $(date '+%F %T') PAGED_SMOKE_DONE" | tee -a $S

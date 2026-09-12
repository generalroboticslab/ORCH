
#!/usr/bin/env bash
# Generate pregenerated ORCH team configs using the project's LLM utilities.
# Style and options mirror other run_ORCH scripts.

set -euo pipefail

PRESETS=(
  # "Cut_Trees_Sparse_small"
  "Cut_Trees_Sparse_large"
  "Cut_Trees_Lines_small"
  "Cut_Trees_Lines_large"
  "Scout_Fire_small"
  "Scout_Fire_large"
  "Transport_Firefighters_small"
  "Transport_Firefighters_large"
  "Rescue_Civilians_Known_Location_small"
  "Rescue_Civilians_Known_Location_large"
  "Suppress_Fire_Contain"
  "Suppress_Fire_Extinguish"
  "Rescue_Civilians_Search_and_Rescue"
  "Suppress_Fire_Locate_and_Suppress"
  "Suppress_Fire_Locate_Deploy_Suppress"
  "Rescue_Civilians_Search_Rescue_Transport"
  "Full_Game"
  "Scout_Fire_Drone_Lost"
  "Transport_Helicopter_Down"
  "Rescue_Civilians_Surprise"
  "Suppress_Fire_Extinguish_Second_Fire"
  "Suppress_Fire_Contain_Water_Source"
  "Suppress_Fire_Extinguish_Rapid_Growth"
  "Scale_Level_Simple"
  # "Scale_Level_Complex"
)

# ----- Configuration (edit as needed) -----
MODEL="${MODEL:-gpt}"
GPT_MODEL="${GPT_MODEL:-gpt-5.5}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
LLM_URL="${LLM_URL:-https://api.openai.com/v1}"
API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
# Generated types:
#   no_critic/both
#   critic/both

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="$SCRIPT_DIR/crew_algorithms/wildfire_alg/team_configs/LLM_generated/$MODEL"

echo "Output root: $OUT_ROOT"
echo "Model: $GPT_MODEL (reasoning effort: $REASONING_EFFORT)"
echo "Generated JSON variants:"
echo "  $OUT_ROOT/no_critic/both"
echo "  $OUT_ROOT/critic/both"

run_job() {
  local preset="$1"
  local critic_mode="$2"
  local critic_flag=()
  local variants_flag=()
  local out_dir="$OUT_ROOT/$critic_mode"
  local log_dir="$out_dir/logs"

  if [ "$critic_mode" = "critic" ]; then
    critic_flag=(--critic)
    variants_flag=(--variants both)
  fi

  echo "Generating preset: $preset ($critic_mode)"

  python "$SCRIPT_DIR/crew_algorithms/wildfire_alg/data/generate_pregenerated_team_configs.py" \
    --model "$MODEL" \
    --model-name "$GPT_MODEL" \
    --reasoning-effort "$REASONING_EFFORT" \
    --task "$preset" \
    --api-key "$API_KEY" \
    --llm-url "$LLM_URL" \
    --outdir "$out_dir" \
    "${variants_flag[@]}" \
    "${critic_flag[@]}"

  status=$?
  if [ "$status" -eq 0 ]; then
    echo "Finished: $preset ($critic_mode)"
  else
    echo "Failed: $preset ($critic_mode, exit=$status)"
  fi

  return ${status:-1}
}

# Main loop: iterate PRESETS
for preset in "${PRESETS[@]}"; do
  run_job "$preset" "no_critic" || echo "Job failed for $preset no_critic (see $OUT_ROOT/no_critic/logs/${preset}.log)"
  run_job "$preset" "critic" || echo "Job failed for $preset critic (see $OUT_ROOT/critic/logs/${preset}.log)"
done

echo "Done. Generated configs are under:"
echo "  $OUT_ROOT/no_critic/both"
echo "  $OUT_ROOT/critic/both"

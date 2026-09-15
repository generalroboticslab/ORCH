PRESETS=(
  "Cut_Trees_Sparse_small"
  # "Cut_Trees_Sparse_large"
  # "Cut_Trees_Lines_small"
  # "Cut_Trees_Lines_large"
  # "Scout_Fire_small"
  # "Scout_Fire_large"
  # "Transport_Firefighters_small"
  # "Transport_Firefighters_large"
  # "Rescue_Civilians_Known_Location_small"
  # "Rescue_Civilians_Known_Location_large"
  # "Suppress_Fire_Contain"
  # "Suppress_Fire_Extinguish"
  # "Rescue_Civilians_Search_and_Rescue"
  # "Suppress_Fire_Locate_and_Suppress"
  # "Suppress_Fire_Locate_Deploy_Suppress"
  # "Rescue_Civilians_Search_Rescue_Transport"
  # "Full_Game"
  # "Scout_Fire_Drone_Lost"
  # "Transport_Helicopter_Down"
  # "Rescue_Civilians_Surprise"
  # "Suppress_Fire_Extinguish_Second_Fire"
  # "Suppress_Fire_Contain_Water_Source"
  # "Suppress_Fire_Extinguish_Rapid_Growth"
  # "Scale_Level_Simple"
  # "Scale_Level_Complex"
)

declare -A SEEDS
SEEDS["Cut_Trees_Sparse_small"]="375 483 43 6370 9964"
SEEDS["Cut_Trees_Sparse_large"]="212 981 1530 5382 9405"
SEEDS["Cut_Trees_Lines_small"]="9259 4881 8456 59497 66768"
SEEDS["Cut_Trees_Lines_large"]="820 5406 6503 7328 2747"
SEEDS["Scout_Fire_small"]="4651 6841 7593 1012 8528"
SEEDS["Scout_Fire_large"]="5324 3603 8592 43126 70576"
SEEDS["Transport_Firefighters_small"]="283 2461 2478 7622 7647"
SEEDS["Transport_Firefighters_large"]="741 7305 9528 8079 6232"
SEEDS["Rescue_Civilians_Known_Location_small"]="9502 3972 6545 5884 8491"
SEEDS["Rescue_Civilians_Known_Location_large"]="7979 1539 2269 7152 5226"
SEEDS["Suppress_Fire_Contain"]="3761 3207 1841 2943 425"
SEEDS["Suppress_Fire_Extinguish"]="1975 4936 6216 2817 6628"
SEEDS["Rescue_Civilians_Search_and_Rescue"]="966 7377 7285 6505 1286"
SEEDS["Suppress_Fire_Locate_and_Suppress"]="5280 2142 2628 2276 524"
SEEDS["Suppress_Fire_Locate_Deploy_Suppress"]="6309 3821 6117 8747 2397"
SEEDS["Rescue_Civilians_Search_Rescue_Transport"]="8208 150 2577 7419 2318"
SEEDS["Full_Game"]="6434 9424 9500 8378 6543"
SEEDS["Scout_Fire_Drone_Lost"]="42 137 256 8362 8503"
SEEDS["Transport_Helicopter_Down"]="42 137 256 6272 8510"
SEEDS["Rescue_Civilians_Surprise"]="42 137 256 5478 4792"
SEEDS["Suppress_Fire_Extinguish_Second_Fire"]="42 8778 4977 6979 1167"
SEEDS["Suppress_Fire_Contain_Water_Source"]="6124 1393 5217 8643 4977"
SEEDS["Suppress_Fire_Extinguish_Rapid_Growth"]="42 137 256 1960 3429"
SEEDS["Scale_Level_Simple"]="42 137 256 503 819"
SEEDS["Scale_Level_Complex"]="42 137 256 503 819"


MODEL="google/gemma-4-31B-it"
URL="http://localhost:8000/v1"
# Leave blank for a local/unauthenticated endpoint.
API_KEY=""

export BASELINE_API_KEY="$API_KEY"
MODEL_FAMILY="custom"
[[ "$URL" == https://api.openai.com/* ]] && MODEL_FAMILY="gpt"
MODEL_LOG_NAME="${MODEL##*/}"


ALGOS=("CAMON" "COELA" "HMAS_2" "Embodied")
MAX_JOBS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/experiment_logs/${MODEL_LOG_NAME}"

mkdir -p "$LOG_DIR"
echo "Logging experiments to: $LOG_DIR"

run_job () {
  local preset=$1
  local seed=$2
  local algo=$3
  local log_file="$LOG_DIR/${algo}/${preset}/seed${seed}.log"
  local status=0

  mkdir -p "$(dirname "$log_file")"

  echo "Starting: $algo | $preset | seed=$seed | log=$log_file"

  {
    echo "Starting: $algo | $preset | seed=$seed"
    echo "Log file: $log_file"
    echo

    python -m crew_algorithms.wildfire_alg.algorithms.${algo} \
      envs.level=$preset \
      envs.seed=$seed \
      envs.collaboration_mode=ai_control \
      envs.llm_model="$MODEL_FAMILY" \
      envs.model_name="$MODEL" \
      envs.llm_url="$URL" \
      envs.no_graphics=true

    status=$?
    echo

    if (( status == 0 )); then
      echo "Finished: $algo | $preset | seed=$seed"
    else
      echo "Failed: $algo | $preset | seed=$seed | exit=$status"
    fi
  } > "$log_file" 2>&1

  return "$status"
}

job_count=0

for algo in "${ALGOS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    for seed in ${SEEDS[$preset]}; do

      run_job "$preset" "$seed" "$algo" &

      ((job_count++))

      sleep 5

      if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
      fi

    done
  done
done

wait

echo "All experiments completed."

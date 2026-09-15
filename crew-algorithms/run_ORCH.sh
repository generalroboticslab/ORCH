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
  # # "Full_Game"
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




declare -A TEAM_CONFIGS
TEAM_CONFIGS["Cut_Trees_Sparse_small"]='++envs.team_config={humans: [], managers: {4: {children: [1, 2, 3], type: "horizontal", team_name: "TREE_CUTTING_TEAM"}}}'
TEAM_CONFIGS["Cut_Trees_Sparse_large"]='++envs.team_config={humans: [], managers: {11: {children: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type: "horizontal", team_name: "TREE_CLEARANCE_TEAM"}}}'
TEAM_CONFIGS["Cut_Trees_Lines_small"]='++envs.team_config={humans: [], managers: {4: {children: [1, 2, 3], type: "horizontal", team_name: "LINE_TREE_CLEARANCE_TEAM"}}}'
TEAM_CONFIGS["Cut_Trees_Lines_large"]='++envs.team_config={humans: [], managers: {8: {children: [1, 2, 3, 4, 5, 6, 7], type: "horizontal", team_name: "TREE_CLEARING_LINE_TEAMS"}}}'
TEAM_CONFIGS["Scout_Fire_small"]='++envs.team_config={humans: [], managers: {4: {children: [1, 2, 3], type: "horizontal", team_name: "MAP_WIDE_FIRE_SEARCH_COORDINATION"}}}'
TEAM_CONFIGS["Scout_Fire_large"]='++envs.team_config={humans: [], managers: {6: {children: [1, 2, 3, 4, 5], type: "horizontal", team_name: "MAP_WIDE_FIRE_SEARCH_AND_CONFIRM"}}}'
TEAM_CONFIGS["Transport_Firefighters_small"]='++envs.team_config={humans: [], managers: {8: {children: [1, 2, 3, 4, 5, 6, 7], type: "vertical", team_name: "FIREFIGHTER_DEPLOYMENT_TRANSPORT"}}}'
TEAM_CONFIGS["Transport_Firefighters_large"]='++envs.team_config={humans: [], managers: {15: {children: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], type: "vertical", team_name: "FIREFIGHTER DEPLOYMENT AIRLIFT COMMAND"}}}'
TEAM_CONFIGS["Rescue_Civilians_Known_Location_small"]='++envs.team_config={humans: [], managers: {4: {children: [1, 2, 3], type: "vertical", team_name: "CIVILIAN_EXTRACTION_TEAM"}}}'
TEAM_CONFIGS["Rescue_Civilians_Known_Location_large"]='++envs.team_config={humans: [], managers: {6: {children: [1, 2, 3, 4, 5], type: "horizontal", team_name: "CIVILIAN RESCUE COORDINATION"}}}'
TEAM_CONFIGS["Suppress_Fire_Contain"]='++envs.team_config={humans: [], managers: {8: {children: [6, 1, 2, 3, 4, 5, 7], type: "vertical", team_name: "FIREBREAK CONTAINMENT TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Extinguish"]='++envs.team_config={humans: [], managers: {9: {children: [1, 2, 3, 4, 5, 6, 7, 8], type: "vertical", team_name: "FIRE_SUPPRESSION_TEAM"}}}'
TEAM_CONFIGS["Rescue_Civilians_Search_and_Rescue"]='++envs.team_config={humans: [], managers: {8: {children: [6, 7, 1, 2, 3, 4, 5], type: "vertical", team_name: "CIVILIAN_SEARCH_AND_RESCUE_TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Locate_and_Suppress"]='++envs.team_config={humans: [], managers: {9: {children: [1, 2, 3, 4, 5, 6, 7, 8], type: "vertical", team_name: "FIRE_SUPPRESSION_TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Locate_Deploy_Suppress"]='++envs.team_config={humans: [], managers: {15: {children: [11, 12], type: "horizontal", team_name: "FIRE_RECON_TEAM"}, 16: {children: [13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type: "horizontal", team_name: "FIRE_SUPPRESSION_TEAM"}, 17: {children: [15, 16], type: "vertical", team_name: "WILDFIRE_SEARCH_AND_SUPPRESSION_COMMAND"}}}'
TEAM_CONFIGS["Rescue_Civilians_Search_Rescue_Transport"]='++envs.team_config={humans: [], managers: {15: {children: [11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type: "vertical", team_name: "SEARCH AND RESCUE COMMAND"}}}'
TEAM_CONFIGS["Full_Game"]='++envs.team_config={humans: [], managers: {16: {children: [1, 2, 3, 4, 5, 11, 14], type: "horizontal", team_name: "Team A"}, 17: {children: [6, 7, 8, 9, 10, 15], type: "horizontal", team_name: "Team B"}, 18: {children: [12, 13], type: "horizontal", team_name: "Recon Team"}, 19: {children: [16, 17, 18], type: "vertical", team_name: "Main Team"}}}'
TEAM_CONFIGS["Scout_Fire_Drone_Lost"]='++envs.team_config={humans: [], managers: {4: {children: [1, 2, 3], type: "horizontal", team_name: "MAP_WIDE_FIRE_SEARCH_COORDINATION"}}}'
TEAM_CONFIGS["Transport_Helicopter_Down"]='++envs.team_config={humans: [], managers: {13: {children: [1, 2, 3, 4, 5, 11], type: "horizontal", team_name: "Transport Team A"}, 14: {children: [6, 7, 8, 9, 10, 12], type: "horizontal", team_name: "Transport Team B"}, 15: {children: [13, 14], type: "horizontal", team_name: "Transport Command"}}}'
TEAM_CONFIGS["Rescue_Civilians_Surprise"]='++envs.team_config={humans: [], managers: {6: {children: [1, 2, 3, 4, 5], type: "vertical", team_name: "FIREBREAK CONTAINMENT TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Extinguish_Second_Fire"]='++envs.team_config={humans: [], managers: {9: {children: [1, 2, 3, 4, 5, 6, 7, 8], type: "vertical", team_name: "FIRE_SUPPRESSION_TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Contain_Water_Source"]='++envs.team_config={humans: [], managers: {6: {children: [1, 2, 3, 4, 5], type: "vertical", team_name: "FIREBREAK CONTAINMENT TEAM"}}}'
TEAM_CONFIGS["Suppress_Fire_Extinguish_Rapid_Growth"]='++envs.team_config={humans: [], managers: {8: {children: [1, 2, 3, 4, 5, 6, 7], type: "vertical", team_name: "FIRE_SUPPRESSION_TASK_FORCE"}}}'
TEAM_CONFIGS["Scale_Level_Simple"]='++envs.team_config={humans: [], managers: {51: {children: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type: "horizontal", team_name: "Squad A"}, 52: {children: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], type: "horizontal", team_name: "Squad B"}, 53: {children: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], type: "horizontal", team_name: "Squad C"}, 54: {children: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], type: "horizontal", team_name: "Squad D"}, 55: {children: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50], type: "horizontal", team_name: "Squad E"}, 56: {children: [51, 52, 53, 54, 55], type: "horizontal", team_name: "Command"}}}'
TEAM_CONFIGS["Scale_Level_Complex"]='++envs.team_config={humans: [], managers: {51: {children: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type: "horizontal", team_name: "Squad A"}, 52: {children: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], type: "horizontal", team_name: "Squad B"}, 53: {children: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], type: "horizontal", team_name: "Squad C"}, 54: {children: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], type: "horizontal", team_name: "Squad D"}, 55: {children: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50], type: "horizontal", team_name: "Squad E"}, 56: {children: [51, 52, 53, 54, 55], type: "vertical", team_name: "Command"}}}'

# Set these values to the exact model ID exposed by your API and its URL.
MODEL="google/gemma-4-31B-it"
URL="http://localhost:8000/v1"
# Leave blank for a local/unauthenticated endpoint.
API_KEY=""
export ORCH_API_KEY="$API_KEY"
MODEL_FAMILY="custom"
[[ "$URL" == https://api.openai.com/* ]] && MODEL_FAMILY="gpt"
MODEL_LOG_NAME="${MODEL##*/}"
GPU_IDS=(0 1 2 3 4 6 7)  # GPU 5 is disabled here.
NUM_GPUS=${#GPU_IDS[@]}
ALGOS=("ORCH")
MAX_JOBS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/experiment_logs/${MODEL_LOG_NAME}"

mkdir -p "$LOG_DIR"
echo "Logging experiments to: $LOG_DIR"

run_job () {
  local preset=$1
  local seed=$2
  local algo=$3
  local gpu_id=$4
  local log_file="$LOG_DIR/${algo}/${preset}/seed${seed}.log"
  local status

  mkdir -p "$(dirname "$log_file")"

  echo "Starting: $algo | $preset | seed=$seed | GPU=$gpu_id | log=$log_file"

  {
    echo "Starting: $algo | $preset | seed=$seed | GPU=$gpu_id"
    echo "Log file: $log_file"
    echo "Team config: $team_config"
    echo


    python -m crew_algorithms.wildfire_alg.algorithms.${algo} \
      envs.level=$preset \
      envs.seed=$seed \
      envs.collaboration_mode=ai_control \
      envs.llm_model="$MODEL_FAMILY" \
      envs.model_name="$MODEL" \
      envs.llm_url="$URL" \
      envs.no_graphics=true \
      "${TEAM_CONFIGS[$preset]}"

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
job_id=0

for preset in "${PRESETS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for seed in ${SEEDS[$preset]}; do

      gpu_id=${GPU_IDS[$(( job_id % NUM_GPUS ))]}
      ((job_id++))

      run_job "$preset" "$seed" "$algo" "$gpu_id" &

      ((job_count++))

      sleep 5

      # Wait when MAX_JOBS are running
      if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
      fi

    done
  done
done

# Wait for all remaining jobs
wait

echo "All experiments completed."

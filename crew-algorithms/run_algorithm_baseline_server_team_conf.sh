PRESETS=(
  # "Cut_Trees_Sparse_small"
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
  "Suppress_Fire_Extinguish_Rapid_Growth"
  "Scale_Level_Simple"
  "Scale_Level_Complex"
)

declare -A SEEDS
SEEDS["Cut_Trees_Sparse_small"]="375"
SEEDS["Cut_Trees_Sparse_large"]="212 981 1530 5382 9405"
SEEDS["Cut_Trees_Lines_small"]="9259 4881 8456 59497 66768"
SEEDS["Cut_Trees_Lines_large"]="820 5406 6503 7328 2747"
SEEDS["Scout_Fire_small"]="4651 6841 7593 1012 8528"
SEEDS["Scout_Fire_large"]="3603 8592 43126 70576"
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

ALGOS=("WILDFIRE")
MAX_JOBS=5

run_job () {
  IFS="|" read -r preset seed algo <<< "$1"

  echo "[START] $preset | seed $seed | algo $algo"

    SDL_VIDEODRIVER=dummy CUDA_VISIBLE_DEVICES=2 python crew_algorithms/wildfire_alg/run_experiments.py \
    --presets "$preset" \
    --parallel 1 \
    --seed "$seed" \
    --algorithm "$algo" \
    --llm_model "qwen" \
    --no_graphics "true" \

  status=$?

  if [ $status -ne 0 ]; then
    echo "[FAILED] $preset | seed $seed | algo $algo (exit $status)" >> failed_jobs.log
  else
    echo "[DONE] $preset | seed $seed | algo $algo"
  fi
}

# Build job list
JOBS=()



for algo in "${ALGOS[@]}"; do
  for preset in "${PRESETS[@]}"; do
      JOBS+=("$preset|1|$algo")
  done
done

running=0
i=0

while [ $i -lt ${#JOBS[@]} ]; do

  while [ $running -lt $MAX_JOBS ] && [ $i -lt ${#JOBS[@]} ]; do
    (
      run_job "${JOBS[$i]}"
    ) &
    ((i++))
    ((running++))
  done

  wait -n
  ((running--))

done

wait
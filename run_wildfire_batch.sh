#!/usr/bin/env bash
set -euo pipefail

# 1. Suppress Fire Contain - all 10
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Contain" --seeds 3761 3207 1841 2943 425 5149 7832 1478 6293 8517 --parallel 5

# 2. Suppress Fire Extinguish - 5 new seeds only
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Extinguish" --seeds 3184 7956 1623 4587 9241 --parallel 5

# 3. Suppress Fire Locate and Suppress - redo 524 + 5 new
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Locate_and_Suppress" --seeds 524 4813 7291 1536 6847 9354 --parallel 5

# 4. Suppress Fire Locate Deploy Suppress - 5 new seeds only
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Locate_Deploy_Suppress" --seeds 2783 5641 8129 3467 7094 --parallel 5

# 5. Suppress Fire Contain Water Source - all 5
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Contain_Water_Source" --seeds 6124 1393 5217 8643 4977 --parallel 5

# 6. Suppress Fire Rapid Growth - redo 137 + replaced 8564
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Suppress_Fire_Extinguish_Rapid_Growth" --seeds 137 3429 --parallel 5

# 7. Full Game - default preset seeds
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Full_Game" --parallel 5

# 8. Cut Trees Sparse Small - seed 43
docker exec -w /app/crew-algorithms/crew_algorithms/wildfire_alg -e PYTHONPATH=/app/crew-algorithms wildfire-algorithm python -m crew_algorithms.wildfire_alg.run_experiments --presets "Cut_Trees_Sparse_small" --seeds 43 --parallel 5

#!/bin/bash

# ORCH Algorithm LLM Test Suite
# Comprehensive testing of team structure generation methods
#
# Test Matrix:
# - 3 Levels: Cut_Trees_Sparse_large, Transport_Firefighters_large, Rescue_Civilians_Known_Location_large
# - 4 Methods per level:
#   2. Human-generated (default predefined structure)
#   2. LLM-generated temp=0 WITH critic
#   3. LLM-generated temp=0 WITHOUT critic
#   4. Simple flat structure
# - 3 seeds per level
# Total: 36 tests (runs 4 at a time in parallel with staggered starts)

echo "=========================================="
echo "ORCH LLM Test Suite"
echo "Starting comprehensive test matrix..."
echo "=========================================="

# ==========================================
# LEVEL 1: Cut_Trees_Sparse_large
# 10 firefighters, 25 trees, 60 map size, 50 max time
# Seeds: 212, 981, 1530
# ==========================================

echo ""
echo ">>> Starting Cut_Trees_Sparse_large tests..."

 #Batch 1: Seed 212 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=212 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=212 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=212 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=212 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &



echo "Cut_Trees_Sparse_large seed 212 completed."

# Batch 2: Seed 981 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=981 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=981 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=981 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=981 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &


wait
echo "Cut_Trees_Sparse_large seed 981 completed."

# Batch 3: Seed 1530 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=1530 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=1530 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=1530 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Cut_Trees_Sparse_large \
    envs.seed=1530 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &



echo "Cut_Trees_Sparse_large all seeds completed."

# ==========================================
# LEVEL 2: Transport_Firefighters_large
# 12 firefighters, 2 helicopters, 100 map size, 20 max time
# Seeds: 741, 7305, 9528
# ==========================================

echo ""
echo ">>> Starting Transport_Firefighters_large tests..."

# Batch 4: Seed 741 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=741 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=741 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &

sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=741 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=741 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &

wait
echo "Transport_Firefighters_large seed 741 completed."

# Batch 5: Seed 7305 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=7305 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=7305 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=7305 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=7305 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &



echo "Transport_Firefighters_large seed 7305 completed."

# Batch 6: Seed 9528 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=9528 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=9528 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=9528 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Transport_Firefighters_large \
    envs.seed=9528 \
    envs.max_steps=20 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &


wait
echo "Transport_Firefighters_large all seeds completed."

# ==========================================
# LEVEL 3: Rescue_Civilians_Known_Location_large
# 3 firefighters, 3x3 civilians, 80 map size, 50 max time
# Seeds: 7979, 1539, 2269, 
# ==========================================

echo ""
echo ">>> Starting Rescue_Civilians_Known_Location_large tests..."

# Batch 7: Seed 7979 - All 4 methods
# python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
#     envs.level=Rescue_Civilians_Known_Location_large \
#     envs.seed=7979 \
#     envs.max_steps=50 \
#     envs.collaboration_mode=ai_control &
# sleep 20

# python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
#     envs.level=Rescue_Civilians_Known_Location_large \
#     envs.seed=7979 \
#     envs.max_steps=50 \
#     envs.collaboration_mode=ai_control \
#     envs.graph=llm_generated \
#     llms.structure_generator_temperature=0.7 \
#     llms.use_structure_critic=true &
# sleep 20

# python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
#     envs.level=Rescue_Civilians_Known_Location_large \
#     envs.seed=7979 \
#     envs.max_steps=50 \
#     envs.collaboration_mode=ai_control \
#     envs.graph=llm_generated \
#     llms.structure_generator_temperature=0.0 \
#     llms.use_structure_critic=false &
# sleep 20

# python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
#     envs.level=Rescue_Civilians_Known_Location_large \
#     envs.seed=7979 \
#     envs.max_steps=50 \
#     envs.collaboration_mode=ai_control \
#     envs.graph=simple &


echo "Rescue_Civilians_Known_Location_large seed 7979 completed."

# Batch 8: Seed 1539 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=1539 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=1539 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=1539 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=1539 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &



echo "Rescue_Civilians_Known_Location_large seed 1539 completed."

# Batch 9: Seed 2269 - All 4 methods
python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=2269 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=2269 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=simple &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=2269 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.7 \
    llms.use_structure_critic=true &
sleep 20

python -m crew_algorithms.wildfire_alg.algorithms.ORCH \
    envs.level=Rescue_Civilians_Known_Location_large \
    envs.seed=2269 \
    envs.max_steps=50 \
    envs.collaboration_mode=ai_control \
    envs.graph=llm_generated \
    llms.structure_generator_temperature=0.0 \
    llms.use_structure_critic=false &


wait
echo "Rescue_Civilians_Known_Location_large all seeds completed."

echo ""
echo "=========================================="
echo "All 36 tests completed!"
echo "=========================================="

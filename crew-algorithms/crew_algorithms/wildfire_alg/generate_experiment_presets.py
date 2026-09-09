"""
Generate experiment_presets.json by running the LLM team structure generator
(with critic) for each experiment-relevant level.

Usage:
    cd crew-algorithms
    python crew_algorithms/wildfire_alg/generate_experiment_presets.py

Requires OPENAI_API_KEY environment variable.
"""

import json
import os
import sys

# Add crew-algorithms to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crew_algorithms.wildfire_alg.config.build_config import create_level_presets
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
    generate_mission_description_from_config,
    generate_team_structure_with_llm,
)


# Levels to skip (utility/demo/scale levels)
SKIP_LEVELS = {
    "VLM_Collection",
    "Scale_Level_Simple",
    "Scale_Level_Complex",
    "Demo_Level",
}

# Seeds per level — each level gets its own set for reproducible sweeps.
# The experiment runner reads these directly; no need to pass --seeds.
LEVEL_SEEDS = {
    # Standard levels
    "Cut_Trees_Sparse_small": [42, 137, 256, 503, 819],
    "Cut_Trees_Sparse_large": [42, 137, 256, 503, 819],
    "Cut_Trees_Lines_small": [42, 137, 256, 503, 819],
    "Cut_Trees_Lines_large": [42, 137, 256, 503, 819],
    "Scout_Fire_small": [42, 137, 256, 503, 819],
    "Scout_Fire_large": [42, 137, 256, 503, 819],
    "Transport_Firefighters_small": [42, 137, 256, 503, 819],
    "Transport_Firefighters_large": [42, 137, 256, 503, 819],
    "Rescue_Civilians_Known_Location_small": [42, 137, 256, 503, 819],
    "Rescue_Civilians_Known_Location_large": [42, 137, 256, 503, 819],
    "Suppress_Fire_Contain": [42, 137, 256, 503, 819],
    "Suppress_Fire_Extinguish": [42, 137, 256, 503, 819],
    "Rescue_Civilians_Search_and_Rescue": [42, 137, 256, 503, 819],
    "Suppress_Fire_Locate_and_Suppress": [42, 137, 256, 503, 819],
    "Suppress_Fire_Locate_Deploy_Suppress": [42, 137, 256, 503, 819],
    "Rescue_Civilians_Search_Rescue_Transport": [42, 137, 256, 503, 819],
    "Full_Game": [42, 137, 256, 503, 819],
    # Scale levels
    "Scale_Level_Simple": [42, 137, 256, 503, 819],
    "Scale_Level_Complex": [42, 137, 256, 503, 819],
    # Special/dynamic levels
    "Scout_Fire_Drone_Lost": [42, 137, 256, 503, 819],
    "Transport_Helicopter_Down": [42, 137, 256, 503, 819],
    "Rescue_Civilians_Surprise": [42, 137, 256, 503, 819],
    "Suppress_Fire_Extinguish_Second_Fire": [42, 137, 256, 503, 819],
    "Suppress_Fire_Contain_Water_Source": [42, 137, 256, 503, 819],
    "Suppress_Fire_Extinguish_Rapid_Growth": [42, 137, 256, 503, 819],
}


class MockEnvs:
    """Mock config.envs object with attributes from a level preset."""

    def __init__(self, preset: dict):
        # Set all preset keys as attributes
        for k, v in preset.items():
            setattr(self, k, v)
        # Ensure defaults for fields the mission generator expects
        for attr, default in [
            ("map_size", 100),
            ("game_type", 0),
            ("known", True),
            ("lines", False),
            ("tree_count", 0),
            ("trees_per_line", 1),
            ("fire_spread_frequency", 0),
            ("water", False),
            ("civilian_count", 0),
            ("civilian_clusters", 1),
            ("civilian_move_frequency", 300),
            ("starting_firefighter_agents", 0),
            ("starting_bulldozer_agents", 0),
            ("starting_drone_agents", 0),
            ("starting_helicopter_agents", 0),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)


class MockLLMs:
    """Mock config.llms object."""

    structure_generator_temperature = 1.0
    use_structure_critic = True


class MockCfg:
    """Mock config object that has .envs and .llms."""

    def __init__(self, preset: dict):
        self.envs = MockEnvs(preset)
        self.llms = MockLLMs()


def get_worker_counts(preset: dict) -> dict:
    return {
        "firefighter": preset.get("starting_firefighter_agents", 0),
        "bulldozer": preset.get("starting_bulldozer_agents", 0),
        "drone": preset.get("starting_drone_agents", 0),
        "helicopter": preset.get("starting_helicopter_agents", 0),
    }


def llm_structure_to_preset_hierarchy(
    structure: dict, worker_counts: dict
) -> dict:
    """
    Convert LLM-generated tree structure to the flat preset hierarchy format.

    LLM output:
        {"root_manager": {"manager_type": "horizontal", "team_name": "...",
         "children": [{"type": "worker", "worker_type": "firefighter", "count": 3}, ...]}}

    Preset format:
        {"managers": ["AGENT_N"], "config": {"AGENT_N": {"children": [...], "type": "...", "team_name": "..."}}}
    """
    # Assign worker IDs in algorithm order: firefighters, bulldozers, drones, helicopters
    worker_id_ranges = {}
    next_id = 1
    for agent_type in ["firefighter", "bulldozer", "drone", "helicopter"]:
        count = worker_counts.get(agent_type, 0)
        worker_id_ranges[agent_type] = list(range(next_id, next_id + count))
        next_id += count

    total_workers = sum(worker_counts.values())

    # Track which workers have been assigned (use a pointer per type)
    worker_usage = {t: 0 for t in worker_id_ranges}

    # First pass: count total managers to assign IDs
    manager_count = [0]

    def count_managers(node):
        if "root_manager" in node:
            node = node["root_manager"]
        if node.get("type") == "manager" or "manager_type" in node:
            manager_count[0] += 1
            for child in node.get("children", []):
                count_managers(child)

    count_managers(structure)

    # Assign manager IDs starting after workers
    next_manager_id = [total_workers + 1]
    managers_list = []
    managers_config = {}

    def process_node(node) -> list:
        """Process a node and return a list of AGENT_X strings (the children IDs)."""
        if "root_manager" in node:
            node = node["root_manager"]

        if node.get("type") == "worker":
            # Assign worker IDs
            wtype = node["worker_type"]
            count = node.get("count", 1)
            ids = []
            for _ in range(count):
                idx = worker_usage[wtype]
                worker_usage[wtype] += 1
                ids.append(f"AGENT_{worker_id_ranges[wtype][idx]}")
            return ids

        elif node.get("type") == "manager" or "manager_type" in node:
            # This is a manager node
            mgr_id = next_manager_id[0]
            next_manager_id[0] += 1

            mgr_name = f"AGENT_{mgr_id}"
            mgr_type = "horizontal" if node.get("manager_type") == "horizontal" else "vertical"
            team_name = node.get("team_name", f"Team_{mgr_id}")

            # Process all children and collect their IDs
            children_ids = []
            for child in node.get("children", []):
                children_ids.extend(process_node(child))

            managers_list.append(mgr_name)
            managers_config[mgr_name] = {
                "children": children_ids,
                "type": mgr_type,
                "team_name": team_name,
            }

            return [mgr_name]

        return []

    process_node(structure)

    return {"managers": managers_list, "config": managers_config}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment presets via LLM team structure generation")
    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help="Only generate for these specific levels (merges into existing file). E.g. Scale_Level_Simple Scale_Level_Complex",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    presets = create_level_presets()

    if args.levels:
        # Only generate for specified levels
        experiment_levels = {}
        for level in args.levels:
            if level in presets:
                experiment_levels[level] = presets[level]
            else:
                print(f"WARNING: Level '{level}' not found in build_config.py. Available: {list(presets.keys())}")
    else:
        experiment_levels = {k: v for k, v in presets.items() if k not in SKIP_LEVELS}

    print(f"Generating hierarchies for {len(experiment_levels)} levels...")
    print(f"Levels: {list(experiment_levels.keys())}")
    print()

    log_dir = os.path.join(os.path.dirname(__file__), "data", "hierarchy_generation_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Load existing presets to merge into (when using --levels)
    output_path = os.path.join(os.path.dirname(__file__), "experiment_presets.json")

    experiment_presets = {}
    if args.levels and os.path.exists(output_path):
        with open(output_path) as f:
            experiment_presets = json.load(f)
        print(f"Loaded {len(experiment_presets)} existing presets (will merge)")

    for level_key, preset in experiment_levels.items():
        print(f"\n{'='*60}")
        print(f"Level: {level_key}")
        print(f"{'='*60}")

        cfg = MockCfg(preset)
        mission_desc = generate_mission_description_from_config(cfg)
        worker_counts = get_worker_counts(preset)

        total_workers = sum(worker_counts.values())
        worker_summary = ", ".join(
            f"{c} {t}{'s' if c > 1 else ''}"
            for t, c in worker_counts.items()
            if c > 0
        )
        print(f"  Mission: {mission_desc}")
        print(f"  Workers: {worker_summary} ({total_workers} total)")

        # Generate team structure with LLM + critic
        level_log_dir = os.path.join(log_dir, level_key)
        os.makedirs(level_log_dir, exist_ok=True)

        try:
            structure = generate_team_structure_with_llm(
                worker_counts=worker_counts,
                mission_description=mission_desc,
                api_key=api_key,
                log_path=level_log_dir,
                cfg=cfg,
            )

            # Convert to preset hierarchy format
            hierarchy = llm_structure_to_preset_hierarchy(structure, worker_counts)

            # Build experiment preset entry
            experiment_presets[level_key] = {
                "name": preset.get("name", level_key),
                "description": preset.get("description", ""),
                "level": level_key,
                "seeds": LEVEL_SEEDS.get(level_key, [42]),
                "collaboration_mode": "ai_control",
                "hierarchy": hierarchy,
            }

            print(f"  Hierarchy: {len(hierarchy['managers'])} manager(s)")
            for mgr, cfg_data in hierarchy["config"].items():
                print(f"    {mgr} ({cfg_data['type']}): {cfg_data['team_name']} -> {cfg_data['children']}")

        except Exception as e:
            print(f"\n  ERROR generating hierarchy for {level_key}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Skipping {level_key} — fix the issue and re-run.")
            continue

    # Write output
    with open(output_path, "w") as f:
        json.dump(experiment_presets, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Written {len(experiment_presets)} presets to {output_path}")
    print(f"Generation logs saved to {log_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate pregenerated team configs using WILDFIRE LLM utilities.

This script uses the same LLM helper functions from
`crew_algorithms/wildfire_alg/algorithms/WILDFIRE/utils.py` to generate team
hierarchies and convert them into preset-style team configs.

Generated files are written under:
  data/pregenerated_team_configs/<model>/<critic|no_critic>/<variant>/<task>.json

Variants:
  both       - allow both full and horizontal managers
  full       - force vertical managers only
  horizontal - force horizontal managers only

By default, critic is disabled unless `--critic` is passed.

Usage:
  python data/generate_pregenerated_team_configs.py --model qwen
  python data/generate_pregenerated_team_configs.py --model qwen --task Full_Game --critic
  python data/generate_pregenerated_team_configs.py --model qwen --critic --variants horizontal
"""

import argparse
import json
import os
import sys

repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from crew_algorithms.wildfire_alg.config.build_config import create_level_presets

VARIANTS = ["both", "full", "horizontal"]
VARIANT_TO_MANAGER_TYPE = {
    "both": "both",
    "full": "vertical",
    "horizontal": "horizontal",
}

generate_mission_description_from_config = None
generate_team_structure_with_llm = None


class TeamStructureConversionError(ValueError):
    """Raised when an LLM team structure cannot map onto the preset workers."""


def load_wildfire_utils():
    try:
        from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
            generate_mission_description_from_config,
            generate_team_structure_with_llm,
        )
    except ImportError as e:
        raise RuntimeError(
            "Unable to import WILDFIRE utils. Make sure the package dependencies are installed, "
            "including the `openai` package for LLM generation."
        ) from e
    return generate_mission_description_from_config, generate_team_structure_with_llm


class MockEnvs:
    def __init__(self, preset: dict):
        for k, v in preset.items():
            setattr(self, k, v)
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
    structure_generator_temperature = 0.7
    use_structure_critic = True


class MockCfg:
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


def llm_structure_to_preset_hierarchy(structure: dict, worker_counts: dict) -> dict:
    worker_id_ranges = {}
    next_id = 1
    for agent_type in ["firefighter", "bulldozer", "drone", "helicopter"]:
        count = worker_counts.get(agent_type, 0)
        worker_id_ranges[agent_type] = list(range(next_id, next_id + count))
        next_id += count

    total_workers = sum(worker_counts.values())
    worker_usage = {t: 0 for t in worker_id_ranges}
    manager_count = [0]

    def count_managers(node):
        if "root_manager" in node:
            node = node["root_manager"]
        if node.get("type") == "manager" or "manager_type" in node:
            manager_count[0] += 1
            for child in node.get("children", []):
                count_managers(child)

    count_managers(structure)

    next_manager_id = [total_workers + 1]
    managers_list = []
    managers_config = {}

    def process_node(node) -> list:
        if "root_manager" in node:
            node = node["root_manager"]

        if node.get("type") == "worker":
            wtype = node["worker_type"]
            count = node.get("count", 1)
            if wtype not in worker_id_ranges:
                raise TeamStructureConversionError(f"Unknown worker type in LLM structure: {wtype}")
            ids = []
            for _ in range(count):
                idx = worker_usage[wtype]
                if idx >= len(worker_id_ranges[wtype]):
                    raise TeamStructureConversionError(
                        f"LLM structure requested too many {wtype} workers "
                        f"({idx + 1} requested, {len(worker_id_ranges[wtype])} available)"
                    )
                worker_usage[wtype] += 1
                ids.append(f"AGENT_{worker_id_ranges[wtype][idx]}")
            return ids

        if node.get("type") == "manager" or "manager_type" in node:
            mgr_id = next_manager_id[0]
            next_manager_id[0] += 1
            mgr_name = f"AGENT_{mgr_id}"
            mgr_type = "horizontal" if node.get("manager_type") == "horizontal" else "vertical"
            team_name = node.get("team_name", f"Team_{mgr_id}")
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


def normalize_tasks(raw_tasks):
    if raw_tasks is None:
        return None
    tasks = []
    for item in raw_tasks:
        for part in item.split(","):
            name = part.strip()
            if name:
                tasks.append(name)
    return tasks or None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def build_team_config(hierarchy: dict) -> dict:
    return {
        "humans": [],
        "managers": hierarchy,
    }


def save_output(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def generate_for_level(
    level_name: str,
    preset: dict,
    model: str,
    critic: bool,
    variants: list[str] | None,
    llm_url: str | None,
    temperature: float,
    out_root: str,
    api_key: str,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
    max_structure_attempts: int = 10,
):
    worker_counts = get_worker_counts(preset)
    if sum(worker_counts.values()) == 0:
        raise ValueError(f"Task {level_name} has no workers defined in preset")

    # When no_critic, only generate "both" variant. Otherwise generate requested variants.
    variants_to_generate = variants if critic and variants else (VARIANTS if critic else ["both"])

    for variant in variants_to_generate:
        variant_manager_type = VARIANT_TO_MANAGER_TYPE[variant]
        cfg = MockCfg(preset)
        cfg.envs.llm_model = model
        cfg.envs.manager_type = variant_manager_type
        cfg.envs.llm_url = llm_url
        cfg.envs.model_name = model_name
        cfg.envs.reasoning_effort = reasoning_effort
        cfg.llms.use_structure_critic = critic
        cfg.llms.structure_generator_temperature = temperature

        mission_desc = generate_mission_description_from_config(cfg)
        log_dir = os.path.join(out_root, "logs", variant)
        ensure_dir(log_dir)
        log_file = os.path.join(log_dir, f"{level_name}.log")

        last_conversion_error = None
        for attempt in range(1, max_structure_attempts + 1):
            if attempt > 1:
                print(
                    f"Regenerating {level_name} ({'critic' if critic else 'no_critic'}/{variant}) "
                    f"after invalid structure: {last_conversion_error}"
                )

            structure = generate_team_structure_with_llm(
                worker_counts=worker_counts,
                mission_description=mission_desc,
                api_key=api_key,
                log_path=log_file,
                cfg=cfg,
            )

            try:
                hierarchy = llm_structure_to_preset_hierarchy(structure, worker_counts)
                break
            except (IndexError, TeamStructureConversionError) as e:
                last_conversion_error = e
                if attempt == max_structure_attempts:
                    raise TeamStructureConversionError(
                        f"Failed to convert generated structure for {level_name} "
                        f"({'critic' if critic else 'no_critic'}/{variant}) after "
                        f"{max_structure_attempts} attempt(s): {e}"
                    ) from e

        team_config = build_team_config(hierarchy["config"])

        output = {
            "level": level_name,
            "model": model,
            "variant": variant,
            "critic": critic,
            "mission_description": mission_desc,
            "worker_counts": worker_counts,
            "team_config": team_config,
            "raw_llm_structure": structure,
        }

        out_dir = os.path.join(out_root, variant)
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{level_name}.json")
        save_output(out_path, output)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pregenerated WILDFIRE team configs via LLM utilities")
    parser.add_argument("--model", "-m", required=True, help="LLM model name used by WILDFIRE utils (gpt, qwen, gemma, glm, llama, deepseek)")
    parser.add_argument("--model-name", help="Exact model name to use for team generation and critique")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="none",
        help="Reasoning effort for GPT requests",
    )
    parser.add_argument("--task", "-t", nargs="*", help="Task level name(s) to generate. Comma-separated values are also accepted.")
    parser.add_argument("--critic", action="store_true", help="Enable critic during generation")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=None,
        help="Critic variants to generate: both, full, horizontal. Defaults to all critic variants.",
    )
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--llm-url", help="Custom LLM base URL for non-OpenAI models (defaults to LLM_URL or OPENAI_BASE_URL env vars)")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature for structure generation")
    parser.add_argument(
        "--max-structure-attempts",
        type=int,
        default=10,
        help="Regenerate the LLM structure up to this many times if it cannot map onto preset workers.",
    )
    parser.add_argument("--outdir", help="Optional output base directory")
    args = parser.parse_args()

    if args.variants and not args.critic:
        parser.error("--variants only applies when --critic is enabled")
    if args.max_structure_attempts < 1:
        parser.error("--max-structure-attempts must be at least 1")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key required. Pass --api-key or set OPENAI_API_KEY env var.")

    llm_url = args.llm_url or os.environ.get("LLM_URL") or os.environ.get("OPENAI_BASE_URL")
    if args.model != "gpt" and not llm_url:
        raise RuntimeError(
            "Non-GPT models require an LLM endpoint URL. Pass --llm-url or set LLM_URL/OPENAI_BASE_URL."
        )

    global generate_mission_description_from_config, generate_team_structure_with_llm
    generate_mission_description_from_config, generate_team_structure_with_llm = load_wildfire_utils()

    tasks = normalize_tasks(args.task)
    presets = create_level_presets()

    if tasks:
        missing = [task for task in tasks if task not in presets]
        if missing:
            raise ValueError(f"Unknown task(s): {', '.join(missing)}")
        selected = {task: presets[task] for task in tasks}
    else:
        selected = presets

    out_root = args.outdir or os.path.join(
        os.path.dirname(__file__),
        "pregenerated_team_configs",
        args.model,
        "critic" if args.critic else "no_critic",
    )
    ensure_dir(out_root)

    for task_name, preset in selected.items():
        print(f"Generating {task_name} ({'critic' if args.critic else 'no_critic'})")
        generate_for_level(
            level_name=task_name,
            preset=preset,
            model=args.model,
            critic=args.critic,
            variants=args.variants,
            llm_url=llm_url,
            temperature=args.temperature,
            out_root=out_root,
            api_key=api_key,
            model_name=args.model_name,
            reasoning_effort=args.reasoning_effort,
            max_structure_attempts=args.max_structure_attempts,
        )


if __name__ == "__main__":
    main()

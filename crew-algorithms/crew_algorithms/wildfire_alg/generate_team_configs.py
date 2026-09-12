"""
Generate team_config overrides for all ORCH experiment presets.

This script reads experiment_presets.json and generates the corresponding
team_config dictionaries in multiple output formats (JSON, OmegaConf CLI, Python dict).

Usage:
    cd crew-algorithms
    python crew_algorithms/wildfire_alg/generate_team_configs.py                    # Print all to stdout
    python crew_algorithms/wildfire_alg/generate_team_configs.py --output-file out.json  # Save JSON
    python crew_algorithms/wildfire_alg/generate_team_configs.py --format omegaconf     # OmegaConf format
    python crew_algorithms/wildfire_alg/generate_team_configs.py --presets "Cut_Trees*" # Filter by pattern
"""

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path


def preset_to_team_config(preset: dict) -> dict:
    """Convert preset hierarchy format to __main__.py team_config format.

    Preset: {"managers": ["AGENT_4"], "config": {"AGENT_4": {"children": ["AGENT_1"...], ...}}}
    team_config: {"humans": [], "managers": {4: {"children": [1...], "type": "...", "team_name": "..."}}}
    """
    hierarchy = preset["hierarchy"]
    managers = {}
    for agent_name, config in hierarchy["config"].items():
        agent_id = int(agent_name.split("_")[1])
        children_ids = [int(c.split("_")[1]) for c in config["children"]]
        managers[agent_id] = {
            "children": children_ids,
            "type": config["type"],
            "team_name": config["team_name"],
        }
    return {"humans": [], "managers": managers}


def dict_to_omegaconf(d):
    """Convert a Python dict to an OmegaConf override string."""
    if isinstance(d, dict):
        items = [f"{k}: {dict_to_omegaconf(v)}" for k, v in d.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(d, list):
        return "[" + ", ".join(dict_to_omegaconf(x) for x in d) + "]"
    elif isinstance(d, str):
        return f'"{d}"'
    else:
        return str(d)


def format_team_config_json(team_configs: dict) -> str:
    """Format team_configs as JSON."""
    return json.dumps(team_configs, indent=2)


def format_team_config_omegaconf(team_configs: dict) -> str:
    """Format team_configs as OmegaConf CLI overrides."""
    lines = []
    for preset_name, team_config in team_configs.items():
        omegaconf_str = dict_to_omegaconf(team_config)
        lines.append(f"# {preset_name}")
        lines.append(f"++envs.team_config={omegaconf_str}")
        lines.append("")
    return "\n".join(lines)


def format_team_config_python(team_configs: dict) -> str:
    """Format team_configs as Python code."""
    lines = ["team_configs = {"]
    for preset_name, team_config in team_configs.items():
        lines.append(f'    "{preset_name}": {team_config},')
    lines.append("}")
    return "\n".join(lines)


def format_team_config_env_vars(team_configs: dict) -> str:
    """Format team_configs as shell environment variables."""
    lines = []
    for preset_name, team_config in team_configs.items():
        env_var_name = f"TEAM_CONFIG_{preset_name.upper()}"
        json_str = json.dumps(team_config).replace('"', '\\"')
        lines.append(f'export {env_var_name}=\'{json.dumps(team_config)}\'')
    return "\n".join(lines)


def format_team_config_table(team_configs: dict) -> str:
    """Format team_configs as a readable table."""
    lines = ["Preset Name | Managers | Total Children | Manager Types | Team Names"]
    lines.append("-" * 100)
    
    for preset_name, team_config in team_configs.items():
        managers_dict = team_config.get("managers", {})
        num_managers = len(managers_dict)
        total_children = sum(len(cfg["children"]) for cfg in managers_dict.values())
        manager_types = ", ".join(set(cfg["type"] for cfg in managers_dict.values()))
        team_names = ", ".join(cfg["team_name"] for cfg in managers_dict.values())
        
        lines.append(f"{preset_name} | {num_managers} | {total_children} | {manager_types} | {team_names}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate team_config overrides for ORCH experiment presets"
    )
    parser.add_argument(
        "--presets-file",
        default=None,
        help="Path to experiment_presets.json (default: auto-detect)",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Filter presets by name pattern (glob). E.g. 'Cut_Trees*' 'Scout_Fire*'",
    )
    parser.add_argument(
        "--format",
        choices=["json", "omegaconf", "python", "env-vars", "table"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information including full team structures",
    )
    args = parser.parse_args()

    # Find presets file
    if args.presets_file:
        presets_path = args.presets_file
    else:
        # Same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        presets_path = os.path.join(script_dir, "experiment_presets.json")

    if not os.path.exists(presets_path):
        print(f"ERROR: Presets file not found: {presets_path}")
        print("Run generate_experiment_presets.py first to create it.")
        sys.exit(1)

    print(f"Loading presets from: {presets_path}", file=sys.stderr)

    with open(presets_path) as f:
        all_presets = json.load(f)

    print(f"Loaded {len(all_presets)} presets", file=sys.stderr)

    # Filter presets by pattern
    if args.presets:
        filtered = {}
        for pattern in args.presets:
            for key in all_presets:
                if fnmatch.fnmatch(key, pattern):
                    filtered[key] = all_presets[key]
        all_presets = filtered
        print(f"Filtered to {len(all_presets)} presets", file=sys.stderr)

    if not all_presets:
        print("No presets matched. Available presets:", file=sys.stderr)
        with open(presets_path) as f:
            available = json.load(f)
        for key in sorted(available.keys()):
            print(f"  {key}", file=sys.stderr)
        sys.exit(1)

    # Generate team_configs
    team_configs = {}
    for preset_key, preset in sorted(all_presets.items()):
        try:
            team_config = preset_to_team_config(preset)
            team_configs[preset_key] = team_config
        except Exception as e:
            print(f"ERROR generating team_config for {preset_key}: {e}", file=sys.stderr)
            continue

    print(f"Generated team_configs for {len(team_configs)} presets", file=sys.stderr)

    # Format output
    if args.format == "json":
        output = format_team_config_json(team_configs)
    elif args.format == "omegaconf":
        output = format_team_config_omegaconf(team_configs)
    elif args.format == "python":
        output = format_team_config_python(team_configs)
    elif args.format == "env-vars":
        output = format_team_config_env_vars(team_configs)
    else:  # table (default)
        output = format_team_config_table(team_configs)

    # Verbose output (detailed structures)
    if args.verbose:
        verbose_output = "\n\n" + "=" * 100 + "\n"
        verbose_output += "DETAILED TEAM STRUCTURES\n"
        verbose_output += "=" * 100 + "\n\n"
        for preset_name, team_config in sorted(team_configs.items()):
            verbose_output += f"Preset: {preset_name}\n"
            verbose_output += "-" * 50 + "\n"
            verbose_output += json.dumps(team_config, indent=2) + "\n\n"
        output += verbose_output

    # Write output
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output)
        print(f"Output written to: {args.output_file}", file=sys.stderr)
    else:
        print(output)

    print(f"Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

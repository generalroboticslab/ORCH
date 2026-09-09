"""
Compare WILDFIRE human preset team hierarchies against default hierarchies.

This compares only hierarchy structure:
  - manager IDs
  - manager types
  - child links

It intentionally ignores team names and all non-hierarchy preset metadata.

Usage:
    cd crew-algorithms
    python crew_algorithms/wildfire_alg/compare_team_hierarchies.py
    python crew_algorithms/wildfire_alg/compare_team_hierarchies.py --output-file hierarchy_comparison.csv
"""

import argparse
import csv
import json
from pathlib import Path


def load_presets(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def normalized_hierarchy(preset: dict) -> dict:
    """Return hierarchy structure without team names or unrelated metadata."""
    hierarchy = preset["hierarchy"]
    normalized_config = {}

    for manager in sorted(hierarchy["config"], key=agent_sort_key):
        config = hierarchy["config"][manager]
        normalized_config[manager] = {
            "type": config.get("type"),
            "children": sorted(config.get("children", []), key=agent_sort_key),
        }

    return {
        "managers": sorted(hierarchy.get("managers", []), key=agent_sort_key),
        "config": normalized_config,
    }


def agent_sort_key(agent_name: str) -> int:
    return int(agent_name.split("_", 1)[1])


def manager_types(hierarchy: dict) -> str:
    return ";".join(
        hierarchy["config"][manager]["type"]
        for manager in hierarchy["managers"]
    )


def manager_child_counts(hierarchy: dict) -> str:
    return ";".join(
        f"{manager}:{len(hierarchy['config'][manager]['children'])}"
        for manager in hierarchy["managers"]
    )


def edge_set(hierarchy: dict) -> set[tuple[str, str]]:
    return {
        (manager, child)
        for manager in hierarchy["managers"]
        for child in hierarchy["config"][manager]["children"]
    }


def type_mismatches(human: dict, default: dict) -> str:
    common_managers = set(human["config"]) & set(default["config"])
    mismatches = []
    for manager in sorted(common_managers, key=agent_sort_key):
        human_type = human["config"][manager]["type"]
        default_type = default["config"][manager]["type"]
        if human_type != default_type:
            mismatches.append(f"{manager}:{human_type}!={default_type}")
    return ";".join(mismatches)


def compare_task(task: str, human_preset: dict, default_preset: dict) -> dict:
    human = normalized_hierarchy(human_preset)
    default = normalized_hierarchy(default_preset)

    human_edges = edge_set(human)
    default_edges = edge_set(default)
    missing_edges = sorted(default_edges - human_edges)
    extra_edges = sorted(human_edges - default_edges)

    return {
        "task": task,
        "same_hierarchy": human == default,
        "human_manager_count": len(human["managers"]),
        "default_manager_count": len(default["managers"]),
        "same_manager_ids": human["managers"] == default["managers"],
        "human_managers": ";".join(human["managers"]),
        "default_managers": ";".join(default["managers"]),
        "same_manager_types": manager_types(human) == manager_types(default),
        "human_manager_types": manager_types(human),
        "default_manager_types": manager_types(default),
        "type_mismatches": type_mismatches(human, default),
        "same_child_links": human_edges == default_edges,
        "human_edge_count": len(human_edges),
        "default_edge_count": len(default_edges),
        "missing_default_edges": len(missing_edges),
        "extra_human_edges": len(extra_edges),
        "human_child_counts": manager_child_counts(human),
        "default_child_counts": manager_child_counts(default),
    }


def write_csv(rows: list[dict], output_file: Path) -> None:
    fieldnames = [
        "task",
        "same_hierarchy",
        "human_manager_count",
        "default_manager_count",
        "same_manager_ids",
        "human_managers",
        "default_managers",
        "same_manager_types",
        "human_manager_types",
        "default_manager_types",
        "type_mismatches",
        "same_child_links",
        "human_edge_count",
        "default_edge_count",
        "missing_default_edges",
        "extra_human_edges",
        "human_child_counts",
        "default_child_counts",
    ]

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_human_file = script_dir / "team_configs" / "preset" / "experiment_presets.json"
    default_default_file = script_dir / "team_configs" / "default" / "experiment_presets.json"
    default_output_file = script_dir / "team_configs" / "hierarchy_comparison.csv"

    parser = argparse.ArgumentParser(
        description="Compare human preset and default WILDFIRE team hierarchies"
    )
    parser.add_argument(
        "--human-file",
        type=Path,
        default=default_human_file,
        help="Human preset experiment_presets.json",
    )
    parser.add_argument(
        "--default-file",
        type=Path,
        default=default_default_file,
        help="Default team config experiment_presets.json",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=default_output_file,
        help="CSV file to write",
    )
    args = parser.parse_args()

    human_presets = load_presets(args.human_file)
    default_presets = load_presets(args.default_file)

    tasks = sorted(set(human_presets) & set(default_presets))
    missing_from_default = sorted(set(human_presets) - set(default_presets))
    missing_from_human = sorted(set(default_presets) - set(human_presets))

    if missing_from_default:
        print(f"WARNING: {len(missing_from_default)} task(s) missing from default file")
    if missing_from_human:
        print(f"WARNING: {len(missing_from_human)} task(s) missing from human file")

    rows = [
        compare_task(task, human_presets[task], default_presets[task])
        for task in tasks
    ]
    write_csv(rows, args.output_file)

    same_count = sum(row["same_hierarchy"] for row in rows)
    print(f"Wrote {len(rows)} task comparisons to {args.output_file}")
    print(f"Exact hierarchy matches: {same_count}/{len(rows)}")


if __name__ == "__main__":
    main()

"""
Generate an LLM team hierarchy and compare it with experiment_presets.json.

Usage from CREW-dev/crew-algorithms:
    python -m crew_algorithms.wildfire_alg.algorithms.WILDFIRE.compare_llm_team_config \
        --levels Cut_Trees_Sparse_small \
        --llm-model llama \
        --llm-url http://localhost:8000/v1

For GPT/OpenAI-compatible default usage:
    OPENAI_API_KEY=... python -m crew_algorithms.wildfire_alg.algorithms.WILDFIRE.compare_llm_team_config \
        --levels Cut_Trees_Sparse_small
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SKIP_LEVELS = {
    "VLM_Collection",
    "Demo_Level",
}


class MockEnvs:
    """Minimal cfg.envs object matching what the WILDFIRE LLM helpers need."""

    def __init__(self, preset: dict[str, Any]):
        for key, value in preset.items():
            setattr(self, key, value)

        defaults = {
            "map_size": 100,
            "game_type": 0,
            "known": True,
            "lines": False,
            "tree_count": 0,
            "trees_per_line": 1,
            "fire_spread_frequency": 0,
            "water": False,
            "civilian_count": 0,
            "civilian_clusters": 1,
            "civilian_move_frequency": 300,
            "starting_firefighter_agents": 0,
            "starting_bulldozer_agents": 0,
            "starting_drone_agents": 0,
            "starting_helicopter_agents": 0,
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)


class MockLLMs:
    """Minimal cfg.llms object matching what the WILDFIRE LLM helpers need."""

    structure_generator_temperature = 0.0
    use_structure_critic = True


class MockCfg:
    def __init__(self, preset: dict[str, Any]):
        self.envs = MockEnvs(preset)
        self.llms = MockLLMs()


def get_worker_counts(preset: dict[str, Any]) -> dict[str, int]:
    return {
        "firefighter": preset.get("starting_firefighter_agents", 0),
        "bulldozer": preset.get("starting_bulldozer_agents", 0),
        "drone": preset.get("starting_drone_agents", 0),
        "helicopter": preset.get("starting_helicopter_agents", 0),
    }


def llm_structure_to_preset_hierarchy(
    structure: dict[str, Any],
    worker_counts: dict[str, int],
) -> dict[str, Any]:
    worker_id_ranges = {}
    next_id = 1
    for agent_type in ["firefighter", "bulldozer", "drone", "helicopter"]:
        count = worker_counts.get(agent_type, 0)
        worker_id_ranges[agent_type] = list(range(next_id, next_id + count))
        next_id += count

    worker_usage = {agent_type: 0 for agent_type in worker_id_ranges}
    total_workers = sum(worker_counts.values())
    next_manager_id = [total_workers + 1]
    managers_list = []
    managers_config = {}

    def process_node(node: dict[str, Any]) -> list[str]:
        if "root_manager" in node:
            node = node["root_manager"]

        if node.get("type") == "worker":
            worker_type = node["worker_type"]
            count = node.get("count", 1)
            ids = []
            for _ in range(count):
                idx = worker_usage[worker_type]
                worker_usage[worker_type] += 1
                ids.append(f"AGENT_{worker_id_ranges[worker_type][idx]}")
            return ids

        if node.get("type") == "manager" or "manager_type" in node:
            manager_id = next_manager_id[0]
            next_manager_id[0] += 1
            manager_name = f"AGENT_{manager_id}"
            manager_type = (
                "horizontal"
                if node.get("manager_type") == "horizontal"
                else "vertical"
            )

            children_ids = []
            for child in node.get("children", []):
                children_ids.extend(process_node(child))

            managers_list.append(manager_name)
            managers_config[manager_name] = {
                "children": children_ids,
                "type": manager_type,
                "team_name": node.get("team_name", f"Team_{manager_id}"),
            }
            return [manager_name]

        return []

    process_node(structure)
    return {"managers": managers_list, "config": managers_config}


def _agent_num(agent_name: str) -> int:
    return int(agent_name.split("_", 1)[1])


def _normalize_hierarchy(hierarchy: dict[str, Any]) -> dict[str, Any]:
    """Return a stable hierarchy representation for exact JSON comparison."""
    config = hierarchy.get("config", {})
    normalized_config = {}

    for manager_name in sorted(config, key=_agent_num):
        manager = config[manager_name]
        normalized_config[manager_name] = {
            "children": sorted(manager.get("children", []), key=_agent_num),
            "type": manager.get("type"),
            "team_name": manager.get("team_name", ""),
        }

    return {
        "managers": sorted(hierarchy.get("managers", []), key=_agent_num),
        "config": normalized_config,
    }


def _edge_set(hierarchy: dict[str, Any]) -> set[tuple[str, str]]:
    edges = set()
    for manager_name, manager in hierarchy.get("config", {}).items():
        for child_name in manager.get("children", []):
            edges.add((manager_name, child_name))
    return edges


def _manager_types(hierarchy: dict[str, Any]) -> dict[str, str]:
    return {
        manager_name: manager.get("type")
        for manager_name, manager in hierarchy.get("config", {}).items()
    }


def _team_names(hierarchy: dict[str, Any]) -> dict[str, str]:
    return {
        manager_name: manager.get("team_name", "")
        for manager_name, manager in hierarchy.get("config", {}).items()
    }


def _format_edge(edge: tuple[str, str]) -> str:
    parent, child = edge
    return f"{parent}->{child}"


def _node_sort_key(node: str) -> tuple[int, int, str]:
    if node.startswith("AGENT_"):
        return (0, _agent_num(node), node)
    if node.startswith("GROUP_"):
        return (1, int(node.split("_", 1)[1]), node)
    return (2, 0, node)


def compare_hierarchies(
    generated_hierarchy: dict[str, Any],
    preset_hierarchy: dict[str, Any],
) -> dict[str, Any]:
    generated_normalized = _normalize_hierarchy(generated_hierarchy)
    preset_normalized = _normalize_hierarchy(preset_hierarchy)

    generated_edges = _edge_set(generated_normalized)
    preset_edges = _edge_set(preset_normalized)
    generated_types = _manager_types(generated_normalized)
    preset_types = _manager_types(preset_normalized)
    generated_names = _team_names(generated_normalized)
    preset_names = _team_names(preset_normalized)

    common_managers = sorted(
        set(generated_types) & set(preset_types),
        key=_agent_num,
    )
    type_diffs = {
        manager: {
            "generated": generated_types[manager],
            "preset": preset_types[manager],
        }
        for manager in common_managers
        if generated_types[manager] != preset_types[manager]
    }
    team_name_diffs = {
        manager: {
            "generated": generated_names[manager],
            "preset": preset_names[manager],
        }
        for manager in common_managers
        if generated_names[manager] != preset_names[manager]
    }

    return {
        "exact_match": generated_normalized == preset_normalized,
        "topology_match": generated_edges == preset_edges,
        "manager_type_match": not type_diffs
        and set(generated_types) == set(preset_types),
        "team_name_match": not team_name_diffs
        and set(generated_names) == set(preset_names),
        "generated_manager_count": len(generated_normalized["managers"]),
        "preset_manager_count": len(preset_normalized["managers"]),
        "generated_only_edges": sorted(
            [_format_edge(edge) for edge in generated_edges - preset_edges]
        ),
        "preset_only_edges": sorted(
            [_format_edge(edge) for edge in preset_edges - generated_edges]
        ),
        "generated_only_managers": sorted(
            set(generated_types) - set(preset_types),
            key=_agent_num,
        ),
        "preset_only_managers": sorted(
            set(preset_types) - set(generated_types),
            key=_agent_num,
        ),
        "manager_type_diffs": type_diffs,
        "team_name_diffs": team_name_diffs,
    }


def _worker_type_by_agent(worker_counts: dict[str, int]) -> dict[str, str]:
    worker_types = {}
    next_id = 1
    for worker_type in ["firefighter", "bulldozer", "drone", "helicopter"]:
        for _ in range(worker_counts.get(worker_type, 0)):
            worker_types[f"AGENT_{next_id}"] = worker_type
            next_id += 1
    return worker_types


def _wrap_label(text: str, max_chars: int = 22) -> list[str]:
    words = str(text).replace("_", " ").split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _hierarchy_roots(hierarchy: dict[str, Any]) -> list[str]:
    managers = set(hierarchy.get("config", {}))
    manager_children = {
        child
        for manager in hierarchy.get("config", {}).values()
        for child in manager.get("children", [])
        if child in managers
    }
    roots = sorted(managers - manager_children, key=_node_sort_key)
    return roots or sorted(managers, key=_node_sort_key)


def _hierarchy_depth(node: str, hierarchy: dict[str, Any]) -> int:
    children = hierarchy.get("config", {}).get(node, {}).get("children", [])
    if not children:
        return 1
    return 1 + max(_hierarchy_depth(child, hierarchy) for child in children)


def _leaf_count(node: str, hierarchy: dict[str, Any]) -> int:
    children = hierarchy.get("config", {}).get(node, {}).get("children", [])
    if not children:
        return 1
    return sum(_leaf_count(child, hierarchy) for child in children)


def _layout_hierarchy(
    hierarchy: dict[str, Any],
    roots: list[str],
    leaf_gap: int,
    level_gap: int,
    margin_x: int,
    margin_y: int,
) -> dict[str, tuple[float, float]]:
    positions = {}
    cursor = [margin_x]

    def place(node: str, depth: int) -> float:
        children = hierarchy.get("config", {}).get(node, {}).get("children", [])
        y = margin_y + depth * level_gap
        if not children:
            x = cursor[0]
            cursor[0] += leaf_gap
        else:
            child_xs = [place(child, depth + 1) for child in children]
            x = (min(child_xs) + max(child_xs)) / 2
        positions[node] = (x, y)
        return x

    for root in roots:
        place(root, 0)
        cursor[0] += leaf_gap

    return positions


def _compact_hierarchy_for_plot(
    hierarchy: dict[str, Any],
    worker_counts: dict[str, int],
    group_consecutive_workers: bool = True,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, str]]:
    hierarchy = _normalize_hierarchy(hierarchy)
    worker_types = _worker_type_by_agent(worker_counts)
    plot_hierarchy = {"managers": hierarchy["managers"], "config": {}}
    labels = {}
    node_types = {}
    group_index = 1

    for manager_name in hierarchy["config"]:
        manager = hierarchy["config"][manager_name]
        manager_type = manager.get("type", "manager")
        labels[manager_name] = [manager_name, manager_type]
        labels[manager_name].extend(_wrap_label(manager.get("team_name", ""), 18)[:2])
        node_types[manager_name] = manager_type

    def flush_group(
        children: list[str],
        grouped_children: list[str],
        current_group: list[str],
    ) -> None:
        nonlocal group_index
        if not current_group:
            return

        if len(current_group) == 1 or not group_consecutive_workers:
            child = current_group[0]
            grouped_children.append(child)
            worker_type = worker_types.get(child, "worker")
            labels[child] = [child, worker_type]
            node_types[child] = worker_type
            return

        first_num = _agent_num(current_group[0])
        last_num = _agent_num(current_group[-1])
        worker_type = worker_types.get(current_group[0], "worker")
        group_name = f"GROUP_{group_index}"
        group_index += 1
        grouped_children.append(group_name)
        labels[group_name] = [f"AGENT_{first_num}-{last_num}", worker_type, f"x{len(current_group)}"]
        node_types[group_name] = worker_type

    for manager_name, manager in hierarchy["config"].items():
        grouped_children = []
        current_group = []
        last_type = None
        last_num = None

        for child in sorted(manager.get("children", []), key=_node_sort_key):
            child_is_worker = child not in hierarchy["config"]
            child_type = worker_types.get(child)
            child_num = _agent_num(child) if child.startswith("AGENT_") else None
            can_group = (
                group_consecutive_workers
                and child_is_worker
                and child_type is not None
                and (not current_group or (child_type == last_type and child_num == last_num + 1))
            )

            if can_group:
                current_group.append(child)
                last_type = child_type
                last_num = child_num
            else:
                flush_group(manager.get("children", []), grouped_children, current_group)
                current_group = []
                last_type = None
                last_num = None

                if child_is_worker and child_type is not None:
                    current_group = [child]
                    last_type = child_type
                    last_num = child_num
                else:
                    grouped_children.append(child)

        flush_group(manager.get("children", []), grouped_children, current_group)
        plot_hierarchy["config"][manager_name] = {
            "children": grouped_children,
            "type": manager.get("type"),
            "team_name": manager.get("team_name", ""),
        }

    return plot_hierarchy, labels, node_types


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


class PngCanvas:
    def __init__(self, width: int, height: int, background: str = "#fbfcfd"):
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(_rgb(background) * self.width * self.height)

    def _set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset:offset + 3] = bytes(color)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: int = 1) -> None:
        x1, y1, x2, y2 = map(lambda value: int(round(value)), (x1, y1, x2, y2))
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        rgb = _rgb(color)
        radius = max(0, width // 2)

        while True:
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    self._set_pixel(x1 + ox, y1 + oy, rgb)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def rect(self, x: float, y: float, width: int, height: int, fill: str, stroke: str, stroke_width: int = 2) -> None:
        x = int(round(x))
        y = int(round(y))
        fill_rgb = _rgb(fill)
        stroke_rgb = _rgb(stroke)
        for py in range(y, y + height):
            for px in range(x, x + width):
                is_border = (
                    px < x + stroke_width
                    or px >= x + width - stroke_width
                    or py < y + stroke_width
                    or py >= y + height - stroke_width
                )
                self._set_pixel(px, py, stroke_rgb if is_border else fill_rgb)

    def text(self, text: str, x: float, y: float, color: str = "#17202a", scale: int = 2, center: bool = True) -> None:
        text = str(text).upper()
        char_width = 5 * scale
        char_gap = scale
        total_width = len(text) * char_width + max(0, len(text) - 1) * char_gap
        start_x = int(round(x - total_width / 2)) if center else int(round(x))
        start_y = int(round(y))
        rgb = _rgb(color)

        for char_index, char in enumerate(text):
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            char_x = start_x + char_index * (char_width + char_gap)
            for row_index, row in enumerate(glyph):
                for col_index, enabled in enumerate(row):
                    if enabled != "1":
                        continue
                    px0 = char_x + col_index * scale
                    py0 = start_y + row_index * scale
                    for oy in range(scale):
                        for ox in range(scale):
                            self._set_pixel(px0 + ox, py0 + oy, rgb)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_rows = []
        stride = self.width * 3
        for y in range(self.height):
            raw_rows.append(b"\x00" + self.pixels[y * stride:(y + 1) * stride])
        raw = b"".join(raw_rows)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=9))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)


def _plot_layout(
    hierarchy: dict[str, Any],
    worker_counts: dict[str, int],
    group_consecutive_workers: bool = True,
) -> dict[str, Any]:
    hierarchy, labels, node_types = _compact_hierarchy_for_plot(
        hierarchy,
        worker_counts,
        group_consecutive_workers=group_consecutive_workers,
    )
    roots = _hierarchy_roots(hierarchy)
    if not roots:
        raise ValueError("Cannot plot hierarchy with no manager roots")

    leaf_gap = 190
    level_gap = 145
    margin_x = 115
    margin_y = 105
    box_width = 160
    box_height = 76
    roots_leaf_count = sum(_leaf_count(root, hierarchy) for root in roots)
    max_depth = max(_hierarchy_depth(root, hierarchy) for root in roots)
    width = max(1000, roots_leaf_count * leaf_gap + 2 * margin_x)
    height = max(380, max_depth * level_gap + margin_y + 80)
    positions = _layout_hierarchy(
        hierarchy=hierarchy,
        roots=roots,
        leaf_gap=leaf_gap,
        level_gap=level_gap,
        margin_x=margin_x,
        margin_y=margin_y,
    )

    return {
        "hierarchy": hierarchy,
        "labels": labels,
        "node_types": node_types,
        "positions": positions,
        "width": width,
        "height": height,
        "box_width": box_width,
        "box_height": box_height,
    }


PLOT_COLORS = {
    "horizontal": ("#d9f0ff", "#2474a6"),
    "vertical": ("#eadcff", "#7652a6"),
    "firefighter": ("#ffe3d8", "#b65b37"),
    "bulldozer": ("#fff1c7", "#a97810"),
    "drone": ("#daf5e5", "#3a8f5a"),
    "helicopter": ("#e3e7ff", "#5463b7"),
    "worker": ("#edf1f5", "#697782"),
}


def _save_hierarchy_png_matplotlib(
    hierarchy: dict[str, Any],
    worker_counts: dict[str, int],
    title: str,
    output_path: Path,
    group_consecutive_workers: bool = True,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        return False

    layout = _plot_layout(
        hierarchy,
        worker_counts,
        group_consecutive_workers=group_consecutive_workers,
    )
    hierarchy = layout["hierarchy"]
    labels = layout["labels"]
    node_types = layout["node_types"]
    positions = layout["positions"]
    width = layout["width"]
    height = layout["height"]
    box_width = layout["box_width"]
    box_height = layout["box_height"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="semibold", pad=18)

    for manager_name, manager in hierarchy.get("config", {}).items():
        parent_x, parent_y = positions[manager_name]
        for child_name in manager.get("children", []):
            child_x, child_y = positions[child_name]
            ax.plot(
                [parent_x, child_x],
                [parent_y + box_height / 2, child_y - box_height / 2],
                color="#9aa7b2",
                linewidth=2,
                zorder=1,
            )

    for node in sorted(positions, key=_node_sort_key):
        x, y = positions[node]
        node_type = node_types.get(node, "worker")
        fill, stroke = PLOT_COLORS.get(node_type, PLOT_COLORS["worker"])
        rect = Rectangle(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=stroke,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            "\n".join(labels.get(node, [node])),
            ha="center",
            va="center",
            fontsize=10,
            color="#17202a",
            linespacing=0.9,
            zorder=3,
        )

    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _save_hierarchy_png_bitmap(
    hierarchy: dict[str, Any],
    worker_counts: dict[str, int],
    title: str,
    output_path: Path,
    group_consecutive_workers: bool = True,
) -> None:
    layout = _plot_layout(
        hierarchy,
        worker_counts,
        group_consecutive_workers=group_consecutive_workers,
    )
    hierarchy = layout["hierarchy"]
    labels = layout["labels"]
    node_types = layout["node_types"]
    positions = layout["positions"]
    width = layout["width"]
    height = layout["height"]
    box_width = layout["box_width"]
    box_height = layout["box_height"]

    canvas = PngCanvas(width, height)
    canvas.text(title, width / 2, 24, scale=3)

    for manager_name, manager in hierarchy.get("config", {}).items():
        parent_x, parent_y = positions[manager_name]
        for child_name in manager.get("children", []):
            child_x, child_y = positions[child_name]
            canvas.line(
                parent_x,
                parent_y + box_height / 2,
                child_x,
                child_y - box_height / 2,
                "#9aa7b2",
                width=3,
            )

    for node in sorted(positions, key=_node_sort_key):
        x, y = positions[node]
        node_type = node_types.get(node, "worker")
        fill, stroke = PLOT_COLORS.get(node_type, PLOT_COLORS["worker"])
        canvas.rect(
            x - box_width / 2,
            y - box_height / 2,
            box_width,
            box_height,
            fill=fill,
            stroke=stroke,
            stroke_width=3,
        )
        node_lines = labels.get(node, [node])
        total_text_height = len(node_lines) * 16
        text_y = y - total_text_height / 2 + 2
        for index, line in enumerate(node_lines):
            canvas.text(line, x, text_y + index * 16, scale=2)

    canvas.save(output_path)


def save_hierarchy_png(
    hierarchy: dict[str, Any],
    worker_counts: dict[str, int],
    title: str,
    output_path: Path,
    group_consecutive_workers: bool = True,
) -> None:
    if _save_hierarchy_png_matplotlib(
        hierarchy,
        worker_counts,
        title,
        output_path,
        group_consecutive_workers=group_consecutive_workers,
    ):
        return
    _save_hierarchy_png_bitmap(
        hierarchy,
        worker_counts,
        title,
        output_path,
        group_consecutive_workers=group_consecutive_workers,
    )


def save_result_plots(results: dict[str, Any], plot_dir: Path) -> dict[str, dict[str, str]]:
    plot_paths = {}
    for level, result in results.items():
        safe_level = "".join(c if c.isalnum() or c in "-_." else "_" for c in level)
        worker_counts = result.get("worker_counts", {})
        level_dir = plot_dir / safe_level
        generated_path = level_dir / "llm_generated_hierarchy.png"
        preset_path = level_dir / "preset_hierarchy.png"

        save_hierarchy_png(
            hierarchy=result["generated_hierarchy"],
            worker_counts=worker_counts,
            title=f"{level} - LLM generated hierarchy",
            output_path=generated_path,
        )
        save_hierarchy_png(
            hierarchy=result["preset_hierarchy"],
            worker_counts=worker_counts,
            title=f"{level} - preset hierarchy",
            output_path=preset_path,
        )
        plot_paths[level] = {
            "generated_plot": str(generated_path),
            "preset_plot": str(preset_path),
        }
        result["plots"] = plot_paths[level]
    return plot_paths


def _build_cfg(
    preset: dict[str, Any],
    llm_model: str,
    llm_url: str | None,
    temperature: float,
    use_critic: bool,
) -> MockCfg:
    cfg = MockCfg(preset)
    cfg.envs.llm_model = llm_model
    cfg.envs.llm_url = llm_url
    cfg.llms.structure_generator_temperature = temperature
    cfg.llms.use_structure_critic = use_critic
    return cfg


def _get_api_key(llm_model: str, explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key

    if llm_model == "gpt":
        env_var = "OPENAI_API_KEY"
    else:
        env_var = f"{llm_model.upper()}_API_KEY"

    api_key = os.environ.get(env_var)
    if api_key:
        return api_key

    if llm_model != "gpt":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key

    raise RuntimeError(
        f"Missing API key. Set {env_var}"
        + (" or OPENAI_API_KEY." if llm_model != "gpt" else ".")
    )


def _load_preset_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_presets_path() -> Path:
    return Path(__file__).resolve().parents[2] / "experiment_presets.json"


def _default_log_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "hierarchy_comparison_logs"


def _select_levels(
    requested_levels: list[str] | None,
    level_presets: dict[str, Any],
    preset_file_data: dict[str, Any],
    include_skipped: bool,
) -> list[str]:
    if requested_levels:
        missing = [
            level
            for level in requested_levels
            if level not in level_presets or level not in preset_file_data
        ]
        if missing:
            raise ValueError(
                "Levels missing from build_config.py or experiment_presets.json: "
                + ", ".join(missing)
            )
        return requested_levels

    levels = [
        level
        for level in preset_file_data
        if level in level_presets
        and (include_skipped or level not in DEFAULT_SKIP_LEVELS)
    ]
    return sorted(levels)


def generate_and_compare_level(
    level: str,
    level_preset: dict[str, Any],
    preset_entry: dict[str, Any],
    api_key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
        generate_mission_description_from_config,
        generate_team_structure_with_llm,
    )

    cfg = _build_cfg(
        level_preset,
        llm_model=args.llm_model,
        llm_url=args.llm_url,
        temperature=args.temperature,
        use_critic=not args.no_critic,
    )
    worker_counts = get_worker_counts(level_preset)
    mission_description = generate_mission_description_from_config(cfg)

    level_log_dir = args.log_dir / level
    level_log_dir.mkdir(parents=True, exist_ok=True)

    structure = generate_team_structure_with_llm(
        worker_counts=worker_counts,
        mission_description=mission_description,
        api_key=api_key,
        log_path=str(level_log_dir),
        cfg=cfg,
    )
    generated_hierarchy = llm_structure_to_preset_hierarchy(structure, worker_counts)
    preset_hierarchy = preset_entry["hierarchy"]
    comparison = compare_hierarchies(generated_hierarchy, preset_hierarchy)

    return {
        "name": preset_entry.get("name", level),
        "description": preset_entry.get("description", ""),
        "level": level,
        "seeds": preset_entry.get("seeds", [42]),
        "collaboration_mode": preset_entry.get("collaboration_mode", "ai_control"),
        "worker_counts": worker_counts,
        "mission_description": mission_description,
        "generated_structure": structure,
        "generated_hierarchy": generated_hierarchy,
        "preset_hierarchy": preset_hierarchy,
        "comparison": comparison,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("\nComparison summary")
    print("=" * 80)
    for level, result in results.items():
        comparison = result["comparison"]
        exact = "yes" if comparison["exact_match"] else "no"
        topology = "yes" if comparison["topology_match"] else "no"
        manager_types = "yes" if comparison["manager_type_match"] else "no"
        team_names = "yes" if comparison["team_name_match"] else "no"
        print(
            f"{level}: exact={exact}, topology={topology}, "
            f"types={manager_types}, names={team_names}, "
            f"managers={comparison['generated_manager_count']}/"
            f"{comparison['preset_manager_count']} generated/preset"
        )

        if comparison["generated_only_edges"]:
            print(f"  generated-only edges: {comparison['generated_only_edges']}")
        if comparison["preset_only_edges"]:
            print(f"  preset-only edges: {comparison['preset_only_edges']}")
        if comparison["manager_type_diffs"]:
            print(f"  manager type diffs: {comparison['manager_type_diffs']}")
        if comparison["team_name_diffs"]:
            print(f"  team name diffs: {comparison['team_name_diffs']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM team configs and compare them with "
            "crew_algorithms/wildfire_alg/experiment_presets.json."
        )
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        default=None,
        help=(
            "Specific levels to compare. If omitted, compares every level present "
            "in experiment_presets.json."
        ),
    )
    parser.add_argument(
        "--presets-file",
        type=Path,
        default=_default_presets_path(),
        help="Path to experiment_presets.json.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSON file for full generated/preset comparison output.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Read an existing comparison JSON and only create plots. "
            "This does not call the LLM."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("hierarchy_plots"),
        help="Directory where preset and LLM-generated hierarchy PNG plots are saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save hierarchy plots.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_default_log_dir(),
        help="Directory for LLM generation logs.",
    )
    parser.add_argument(
        "--llm-model",
        choices=["gpt", "qwen", "deepseek", "gemma", "glm", "llama"],
        default="gpt",
        help="LLM provider/model family used by WILDFIRE utils.",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        help="OpenAI-compatible base URL for non-GPT models.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key. Defaults to provider env var.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Team structure generator and critic temperature.",
    )
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable the structure critic and accept the first generated hierarchy.",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include utility/demo levels when --levels is omitted.",
    )
    return parser.parse_args()


def main() -> int:
    from crew_algorithms.wildfire_alg.config.build_config import create_level_presets

    args = parse_args()
    args.presets_file = args.presets_file.resolve()
    args.log_dir = args.log_dir.resolve()
    args.plot_dir = args.plot_dir.resolve()
    if args.output_file:
        args.output_file = args.output_file.resolve()
    if args.input_file:
        args.input_file = args.input_file.resolve()
        results = _load_preset_file(args.input_file)
        print(f"Loaded {len(results)} comparison result(s) from {args.input_file}")
        print_summary(results)
        if not args.no_plots:
            plot_paths = save_result_plots(results, args.plot_dir)
            print(f"\nWrote hierarchy plots to {args.plot_dir}")
            for level, paths in plot_paths.items():
                print(f"  {level}:")
                print(f"    generated: {paths['generated_plot']}")
                print(f"    preset:    {paths['preset_plot']}")
        return 0

    preset_file_data = _load_preset_file(args.presets_file)
    level_presets = create_level_presets()
    levels = _select_levels(
        args.levels,
        level_presets,
        preset_file_data,
        include_skipped=args.include_skipped,
    )
    api_key = _get_api_key(args.llm_model, args.api_key)

    results = {}
    print(f"Loaded {len(preset_file_data)} preset entries from {args.presets_file}")
    print(f"Comparing {len(levels)} level(s): {', '.join(levels)}")

    for level in levels:
        print("\n" + "=" * 80)
        print(f"Level: {level}")
        print("=" * 80)
        results[level] = generate_and_compare_level(
            level=level,
            level_preset=level_presets[level],
            preset_entry=preset_file_data[level],
            api_key=api_key,
            args=args,
        )

    print_summary(results)

    if not args.no_plots:
        plot_paths = save_result_plots(results, args.plot_dir)
        print(f"\nWrote hierarchy plots to {args.plot_dir}")
        for level, paths in plot_paths.items():
            print(f"  {level}:")
            print(f"    generated: {paths['generated_plot']}")
            print(f"    preset:    {paths['preset_plot']}")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote full comparison to {args.output_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

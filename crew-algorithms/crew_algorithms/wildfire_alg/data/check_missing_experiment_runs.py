#!/usr/bin/env python3
"""Report missing wildfire experiment runs by model, algorithm, task, and seed."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALGORITHMS = ("ORCH", "CAMON", "COELA", "Embodied", "HMAS_2")
DEFAULT_MODELS = ("gpt", "qwen", "deepseek")
RUN_RE = re.compile(r"_seed(?P<seed>\d+)(?:_run(?P<run>\d+))?\.log$")
EXIT_CODE_RE = re.compile(r"# Exit code: (?P<code>-?\d+)")
MODEL_RE = re.compile(
    r"(?:^|\s)(?:envs\.)?llm_model=(?P<model>[^\s]+)|--llm_model\s+(?P<arg_model>[^\s]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which wildfire experiment task/seed runs are missing."
    )
    parser.add_argument(
        "--presets-file",
        default=str(PROJECT_ROOT / "crew_algorithms/wildfire_alg/experiment_presets.json"),
        help="Path to experiment_presets.json.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(PROJECT_ROOT / "experiment_logs"),
        help="Directory containing timestamped experiment log folders.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "missing_experiment_runs.md"),
        help="Markdown report path to write.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        help="Algorithms expected for every task/seed.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="LLM models expected for every algorithm/task/seed.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task/preset names to check. Defaults to every preset.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Treat logs without '# Exit code: 0' as missing.",
    )
    return parser.parse_args()


def read_text(path: Path, limit: int | None = None) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(limit) if limit else f.read()


def model_from_log(path: Path) -> str:
    """Return the model in a log command header, or gpt for older default logs."""
    header = read_text(path, limit=4096)
    match = MODEL_RE.search(header)
    if not match:
        return "gpt"
    return match.group("model") or match.group("arg_model")


def has_success_exit(path: Path) -> bool:
    content = read_text(path)
    exit_codes = [int(match.group("code")) for match in EXIT_CODE_RE.finditer(content)]
    return bool(exit_codes) and exit_codes[-1] == 0


def parse_log_name(path: Path, algorithms: list[str]) -> tuple[str, str, int] | None:
    """Parse ALGORITHM_TASK_seedN.log, including algorithms with underscores."""
    name = path.name
    seed_match = RUN_RE.search(name)
    if not seed_match:
        return None

    prefix = name[: seed_match.start()]
    for algorithm in sorted(algorithms, key=len, reverse=True):
        marker = f"{algorithm}_"
        if prefix.startswith(marker):
            task = prefix[len(marker) :]
            return algorithm, task, int(seed_match.group("seed"))
    return None


def load_expected_runs(
    presets_path: Path, task_filter: list[str] | None
) -> dict[str, list[int]]:
    with presets_path.open("r", encoding="utf-8") as f:
        presets = json.load(f)

    if task_filter:
        missing_tasks = sorted(set(task_filter) - set(presets))
        if missing_tasks:
            raise SystemExit(f"Unknown task(s): {', '.join(missing_tasks)}")
        presets = {task: presets[task] for task in task_filter}

    return {task: list(config.get("seeds", [42])) for task, config in presets.items()}


def scan_logs(
    logs_dir: Path, algorithms: list[str], require_success: bool
) -> dict[tuple[str, str, str], set[int]]:
    observed: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    if not logs_dir.exists():
        return observed

    for log_path in logs_dir.rglob("*.log"):
        parsed = parse_log_name(log_path, algorithms)
        if not parsed:
            continue

        algorithm, task, seed = parsed
        if require_success and not has_success_exit(log_path):
            continue

        model = model_from_log(log_path)
        observed[(model, algorithm, task)].add(seed)

    return observed


def build_report(
    expected: dict[str, list[int]],
    observed: dict[tuple[str, str, str], set[int]],
    models: list[str],
    algorithms: list[str],
    require_success: bool,
) -> tuple[str, int, int]:
    total_expected = (
        len(models) * len(algorithms) * sum(len(seeds) for seeds in expected.values())
    )
    missing_count = 0
    lines = [
        "# Missing Experiment Runs",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Completion rule: {'successful log with # Exit code: 0' if require_success else 'any matching log file'}",
        "",
    ]

    for model in models:
        lines.extend([f"## Model: {model}", ""])
        model_missing = 0

        for algorithm in algorithms:
            algorithm_lines = []
            for task, seeds in sorted(expected.items()):
                missing = [
                    seed
                    for seed in seeds
                    if seed not in observed.get((model, algorithm, task), set())
                ]
                if missing:
                    model_missing += len(missing)
                    missing_count += len(missing)
                    algorithm_lines.append(f"- `{task}`: {', '.join(map(str, missing))}")

            if algorithm_lines:
                lines.extend([f"### {algorithm}", *algorithm_lines, ""])
            else:
                lines.extend([f"### {algorithm}", "No missing runs.", ""])

        if model_missing == 0:
            lines.append(f"All expected `{model}` runs are present.")
            lines.append("")

    completed_count = total_expected - missing_count
    lines.extend(
        [
            "## Summary",
            "",
            f"- Expected combinations: {total_expected}",
            f"- Present combinations: {completed_count}",
            f"- Missing combinations: {missing_count}",
            "",
        ]
    )
    return "\n".join(lines), total_expected, missing_count


def main() -> None:
    args = parse_args()
    presets_path = Path(args.presets_file)
    logs_dir = Path(args.logs_dir)
    output_path = Path(args.output)

    expected = load_expected_runs(presets_path, args.tasks)
    observed = scan_logs(logs_dir, args.algorithms, args.require_success)
    report, total_expected, missing_count = build_report(
        expected,
        observed,
        args.models,
        args.algorithms,
        args.require_success,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Expected combinations: {total_expected}")
    print(f"Missing combinations: {missing_count}")


if __name__ == "__main__":
    main()

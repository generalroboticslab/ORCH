"""Render all experiment trajectories in experiment_data/FINAL/ to video."""

from pathlib import Path
from data.render_logs import generate_all_videos

if __name__ == "__main__":
    data_root = Path(__file__).parent / "algorithms" / "ORCH" / "results" / "experiment_data" / "FINAL"
    print(f"Rendering all trajectories in {data_root}")
    import sys
    force = "--force" in sys.argv
    generate_all_videos(data_root, force=force)

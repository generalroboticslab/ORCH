import cv2
import os
from pathlib import Path
import re
import numpy as np


def sort_key(filename):
    match = re.search(r'capture_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def load_sorted_images(folder, repeat=1):
    image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')], key=sort_key)
    images = []
    for filename in image_files:
        img = cv2.imread(os.path.join(folder, filename))
        for _ in range(repeat):
            images.append(img)
    return images


def get_frame_or_last(images, index):
    if not images:
        return None
    if index < len(images):
        return images[index]
    else:
        return images[-1]  # Repeat last frame


def resize_to_match(image, target_size):
    return cv2.resize(image, target_size)


def compile_split_screen_video(base_folder, output_video_path, frame_rate=10):
    """Render video: accumulative map on left, agent grid on right.

    Each agent cell is minimap+POV concatenated horizontally.
    The agent grid prefers a vertical layout (more rows than columns).
    """
    import math

    base_path = Path(base_folder)
    agents = sorted(
        [f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("Agent_")],
        key=lambda p: int(re.search(r'(\d+)', p.name).group(1)),
    )

    if not agents:
        print(f"Warning: No Agent_* folders found in {base_folder}, skipping video compilation")
        return

    # Load agent feeds (minimap + pov per agent)
    agent_feeds = []
    max_frames = 0
    for agent in agents:
        minimap_folder = agent / "Minimap"
        pov_folder = agent / "POV"
        if minimap_folder.exists() and pov_folder.exists():
            minimap_imgs = load_sorted_images(minimap_folder)
            pov_imgs = load_sorted_images(pov_folder)
            if minimap_imgs and pov_imgs:
                max_frames = max(max_frames, len(minimap_imgs), len(pov_imgs))
                agent_feeds.append((minimap_imgs, pov_imgs))

    # Load accumulative map
    acc_folder = base_path / "Server_Accumulative"
    acc_imgs = load_sorted_images(acc_folder) if acc_folder.exists() else []
    max_frames = max(max_frames, len(acc_imgs))

    if not agent_feeds:
        print(f"Warning: No valid agent feeds in {base_folder}, skipping")
        return

    # Reference cell size from first agent minimap
    cell_h, cell_w, _ = agent_feeds[0][0][0].shape
    cell_size = (cell_w, cell_h)

    # Grid layout for agents — prefer vertical (more rows than cols)
    num_agents = len(agent_feeds)
    best_cols = 1
    best_score = float('inf')
    for cols in range(1, num_agents + 1):
        rows = math.ceil(num_agents / cols)
        if cols > rows:
            continue
        aspect_ratio = rows / cols
        penalty = aspect_ratio * 2 if aspect_ratio > 3 else 1
        score = cols * penalty
        if score < best_score:
            best_score = score
            best_cols = cols
    grid_cols = best_cols
    grid_rows = math.ceil(num_agents / grid_cols)

    # Each agent cell is minimap + pov side by side
    agent_cell_w = cell_w * 2
    agent_cell_h = cell_h
    grid_w = grid_cols * agent_cell_w
    grid_h = grid_rows * agent_cell_h

    # Accumulative map resized to match grid height
    if acc_imgs:
        acc_h_orig, acc_w_orig, _ = acc_imgs[0].shape
        acc_scale = grid_h / acc_h_orig
        acc_w = int(acc_w_orig * acc_scale)
        acc_h = grid_h
        acc_size = (acc_w, acc_h)
    else:
        acc_w = 0
        acc_size = None

    frame_w = acc_w + grid_w
    frame_h = grid_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video_path), fourcc, frame_rate, (frame_w, frame_h))

    for i in range(max_frames):
        # Build agent grid
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, (minimap_imgs, pov_imgs) in enumerate(agent_feeds):
            # Black out destroyed agents (no more captures)
            if i >= len(minimap_imgs) or i >= len(pov_imgs):
                continue
            row = idx // grid_cols
            col = idx % grid_cols
            minimap_frame = resize_to_match(minimap_imgs[i], cell_size)
            pov_frame = resize_to_match(pov_imgs[i], cell_size)
            agent_cell = cv2.hconcat([minimap_frame, pov_frame])
            y = row * agent_cell_h
            x = col * agent_cell_w
            grid[y:y + agent_cell_h, x:x + agent_cell_w] = agent_cell

        # Accumulative map on left, agent grid on right
        if acc_imgs and acc_size:
            acc_frame = resize_to_match(get_frame_or_last(acc_imgs, i), acc_size)
            final_frame = cv2.hconcat([acc_frame, grid])
        else:
            final_frame = grid

        video.write(final_frame)

    video.release()


def generate_all_videos(data_root, frame_rate=10, force=False):
    """Generate videos for all runs in the data directory."""
    data_path = Path(data_root)

    # Find timestamped run folders at {scenario}/{seed}/preset/{timestamp}
    run_folders = []
    for path in data_path.glob("*/*/*/*"):
        if path.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', path.name):
            run_folders.append(path)

    print(f"Found {len(run_folders)} run folders")

    for run_folder in sorted(run_folders):
        output_path = run_folder / "combined_output.mp4"

        # Skip if video already exists (unless force re-render)
        if output_path.exists() and not force:
            print(f"Skipping {run_folder.name} - video already exists")
            continue

        # Check if folder has required structure
        has_agents = any(f.is_dir() and f.name.startswith("Agent_") for f in run_folder.iterdir())

        if not has_agents:
            print(f"Skipping {run_folder.name} - no Agent folders found")
            continue

        if output_path.exists():
            print(f"Re-rendering {run_folder.name}...")
            output_path.unlink()
        else:
            print(f"Generating video for {run_folder.name}...")
        try:
            compile_split_screen_video(run_folder, output_path, frame_rate)
            if output_path.exists():
                print(f"  -> Saved to {output_path}")
        except Exception as e:
            print(f"  -> Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Generate videos for all runs
        data_root = Path(__file__).parent.parent / "algorithms" / "WILDFIRE" / "results" / "data"
        generate_all_videos(data_root)
    else:
        # Single folder mode
        if len(sys.argv) > 1:
            folder_path = sys.argv[1]
        else:
            folder_path = "./crew-algorithms/crew_algorithms/wildfire_alg/results/logs/CAMON/Cut_Trees_Sparse_small/483/2025-05-30-16-18-19"
        output_path = f"{folder_path}/combined_output.mp4"
        compile_split_screen_video(folder_path, output_path)

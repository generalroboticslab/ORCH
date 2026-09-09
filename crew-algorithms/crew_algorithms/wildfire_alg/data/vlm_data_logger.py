"""
VLM Data Logger

Utility to log ASCII grids, minimap images, and metadata for VLM comparison experiments.
This module integrates with the WILDFIRE algorithm to capture worker agent observations.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Tuple, List, Optional


class VLMDataLogger:
    """Logger for VLM comparison data collection"""

    def __init__(self, base_folder: str, enabled: bool = False):
        """
        Initialize VLM data logger

        Args:
            base_folder: Base folder path for saving VLM comparison data
            enabled: Whether data collection is enabled
        """
        self.enabled = enabled
        self.base_folder = Path(base_folder)
        self.sample_counter = 0

        if self.enabled:
            self.base_folder.mkdir(parents=True, exist_ok=True)
            print(f"[VLM Data Logger] Initialized. Saving to: {self.base_folder}")

    def log_worker_observation(self,
                               agent_id: int,
                               timestep: int,
                               ascii_grid: str,
                               agent_position: Tuple[int, int],
                               agent_type: int,
                               map_range: int,
                               extra_variables: List[float],
                               results_path: str):
        """
        Log worker agent observation data for VLM comparison

        This function copies the existing minimap image from the results folder
        and saves the ASCII grid + metadata alongside it.

        Args:
            agent_id: ID of the agent
            timestep: Current timestep
            ascii_grid: ASCII string representation of the minimap
            agent_position: (x, y) position of the agent
            agent_type: Type of agent (0-3)
            map_range: Size of the perception grid
            extra_variables: Additional state variables
            results_path: Base path to the results folder where minimap images are saved
        """
        if not self.enabled:
            return

        # Find the latest minimap image for this agent
        minimap_image_path = self._find_latest_minimap(results_path, agent_id)

        if not minimap_image_path:
            print(f"[VLM Data Logger] Warning: No minimap image found for agent {agent_id} at timestep {timestep}")
            return

        # Generate unique sample name
        sample_name = f"agent{agent_id}_t{timestep}_{self.sample_counter}"
        self.sample_counter += 1

        # Copy minimap image
        image_dest = self.base_folder / f"{sample_name}_image.png"
        try:
            shutil.copy2(minimap_image_path, image_dest)
        except Exception as e:
            print(f"[VLM Data Logger] Error copying minimap image: {e}")
            return

        # Save ASCII grid
        ascii_path = self.base_folder / f"{sample_name}_ascii.txt"
        with open(ascii_path, 'w') as f:
            f.write(ascii_grid)

        # Save metadata
        metadata = {
            "agent_id": agent_id,
            "timestep": timestep,
            "agent_position": list(agent_position),
            "agent_type": agent_type,
            "map_range": map_range,
            "extra_variables": list(extra_variables)
        }
        metadata_path = self.base_folder / f"{sample_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[VLM Data Logger] Logged sample: {sample_name}")

    def _find_latest_minimap(self, results_path: str, agent_id: int) -> Optional[Path]:
        """
        Find the latest minimap image for an agent

        Args:
            results_path: Base path to results folder
            agent_id: Agent ID

        Returns:
            Path to latest minimap image or None if not found
        """
        minimap_dir = Path(results_path) / f"Agent_{agent_id}" / "minimap"

        if not minimap_dir.exists():
            return None

        # Find all PNG files with capture_N.png pattern
        image_files = list(minimap_dir.glob("capture_*.png"))

        if not image_files:
            return None

        # Extract capture number and find the latest
        def get_capture_number(filepath):
            try:
                # Extract number from capture_N.png
                stem = filepath.stem  # e.g., "capture_5"
                return int(stem.split('_')[1])
            except:
                return -1

        latest_image = max(image_files, key=get_capture_number)
        return latest_image

    def get_sample_count(self) -> int:
        """Get number of samples logged"""
        return self.sample_counter


# Global logger instance
_vlm_logger: Optional[VLMDataLogger] = None


def init_vlm_logger(base_folder: str, enabled: bool = False) -> VLMDataLogger:
    """
    Initialize the global VLM data logger

    Args:
        base_folder: Base folder for saving data
        enabled: Whether to enable data collection

    Returns:
        VLMDataLogger instance
    """
    global _vlm_logger
    _vlm_logger = VLMDataLogger(base_folder, enabled)
    return _vlm_logger


def get_vlm_logger() -> Optional[VLMDataLogger]:
    """
    Get the global VLM data logger

    Returns:
        VLMDataLogger instance or None if not initialized
    """
    return _vlm_logger


def log_worker_observation(agent_id: int,
                          timestep: int,
                          ascii_grid: str,
                          agent_position: Tuple[int, int],
                          agent_type: int,
                          map_range: int,
                          extra_variables: List[float],
                          results_path: str):
    """
    Log worker observation using global logger

    Args:
        agent_id: ID of the agent
        timestep: Current timestep
        ascii_grid: ASCII string representation of the minimap
        agent_position: (x, y) position of the agent
        agent_type: Type of agent (0-3)
        map_range: Size of the perception grid
        extra_variables: Additional state variables
        results_path: Base path to the results folder
    """
    if _vlm_logger:
        _vlm_logger.log_worker_observation(
            agent_id, timestep, ascii_grid, agent_position,
            agent_type, map_range, extra_variables, results_path
        )

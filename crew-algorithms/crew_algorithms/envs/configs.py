from pathlib import Path
from typing import Callable

from attrs import define
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import os


@define(auto_attribs=True)
class EnvironmentConfig:
    name: str = MISSING
    num_channels: int = 3
    num_stacks: int = MISSING
    action_space: str = MISSING
    action_dims: int = MISSING
    unity_server_build_path: Path = MISSING
    unity_server_build_path_osx: Path = MISSING
    unity_server_build_path_linux: Path = MISSING
    unity_server_build_path_windows: Path = MISSING
    log_folder_path: Path = "../UnityLogs"
    human_delay_steps: int = 1  # number of steps to shift human feedback
    no_graphics: bool = False
    time_scale: float = 1.0
    seed: int = 42
    pretrained_encoder: bool = False  # whether to use a pretrained encoder
    additional_in_keys: dict = {}  # additional keys to include in the input
    target_img: Path = "assets/treasure.png"
    scale_reward: float = 1.0
    shift_reward: float = 0.0
    dense_reward_scale: float = 1.0
    credit_window_right: float = 4.0


@define(auto_attribs=True)
class BowlingConfig(EnvironmentConfig):
    name: str = "bowling"
    num_agents: int = 1
    num_channels: int = 1
    num_stacks: int = 1
    action_space: str = "continuous"
    action_dims: int = 3
    unity_server_build_path_linux: Path = (
        "../crew-dojo/Builds/Bowling-StandaloneLinux64-Server/Unity.x86_64"
    )
    unity_server_build_path_osx: Path = (
        "../crew-dojo/Builds/Bowling-StandaloneOSX-Server/Unity"
    )
    unity_server_build_path_windows: Path = (
        "../crew-dojo/Builds/Bowling-StandaloneWindows-Server/Unity"
    )
    log_folder_path: Path = "../UnityLogs"
    human_delay_steps: int = 4
    no_graphics: bool = False
    time_scale: float = 1.0
    seed: int = 42
    state_start_dim: int = 4
    state_end_dim: int = 36
    pretrained_encoder: bool = False
    crop_h: int = 60
    crop_w: int = 120
    action_low: float = -1.0
    action_high: float = 1.0
    scale_reward: float = 1.0
    shift_reward: float = 0.0
    dense_reward_scale: float = 5.0
    credit_window_right: float = 4.0


@define(auto_attribs=True)
class FindTreasureConfig(EnvironmentConfig):
    name: str = "find_treasure"
    num_agents: int = 1
    num_channels: int = 3
    num_stacks: int = 3
    rand_maze: bool = False
    action_space: str = "continuous"
    action_dims: int = 2
    unity_server_build_path_linux: Path = (
        "../crew-dojo/Builds/FindTreasure-StandaloneLinux64-Server/Unity.x86_64"
    )
    unity_server_build_path_osx: Path = (
        "../crew-dojo/Builds/FindTreasure-StandaloneOSX-Server/Unity"
    )
    unity_server_build_path_windows: Path = (
        "../crew-dojo/Builds/FindTreasure-StandaloneWindows-Server/Unity.exe"
    )
    log_folder_path: Path = "../UnityLogs"
    human_delay_steps: int = 2
    no_graphics: bool = False
    time_scale: float = 1.0
    seed: int = 42
    state_start_dim: int = 5
    state_end_dim: int = 9
    pretrained_encoder: bool = False
    additional_in_keys: dict = {"step_count": ("agents", "step_count")}
    crop_h: int = 100
    crop_w: int = 100
    action_low: float = -10.0
    action_high: float = 10.0
    target_img: Path = "assets/treasure.png"
    scale_reward: float = 11.0
    shift_reward: float = -1.0
    dense_reward_scale: float = 1.0
    credit_window_right: float = 1.0


@define(auto_attribs=True)
class HideAndSeek1V1Config(EnvironmentConfig):
    name: str = "hide_and_seek_1v1"
    num_agents: int = 1
    num_channels: int = 3
    num_stacks: int = 3
    rand_maze: bool = False
    action_space: str = "continuous"
    action_dims: int = 2
    unity_server_build_path_linux: Path = (
        "../crew-dojo/Builds/HideAndSeek_Single-StandaloneLinux64-Server/Unity.x86_64"
    )
    unity_server_build_path_osx: Path = (
        "../crew-dojo/Builds/HideAndSeek_Single-StandaloneOSX-Server/Unity"
    )
    unity_server_build_path_windows: Path = (
        "../crew-dojo/Builds/HideAndSeek_Single-StandaloneWindows-Server/Unity.exe"
    )
    log_folder_path: Path = "../UnityLogs"
    human_delay_steps: int = 2
    no_graphics: bool = False
    time_scale: float = 1.0
    seed: int = 42
    state_start_dim: int = 5
    state_end_dim: int = 9
    pretrained_encoder: bool = False
    additional_in_keys: dict = {"step_count": ("agents", "step_count")}
    crop_h: int = 100
    crop_w: int = 100
    action_low: float = -10.0
    action_high: float = 10.0
    target_img: Path = "assets/hider.png"
    scale_reward: float = 11.0
    shift_reward: float = -1.0
    dense_reward_scale: float = 1.0
    credit_window_right: float = 1.0


@define(auto_attribs=True)
class HideAndSeekConfig(EnvironmentConfig):
    name: str = "hide_and_seek"
    num_hiders: int = 2
    num_seekers: int = 2
    num_channels: int = 3
    num_stacks: int = 3
    rand_maze: bool = False
    action_space: str = "continuous"
    action_dims: int = 2
    unity_server_build_path_linux: Path = (
        "../crew-dojo/Builds/HideAndSeek-StandaloneLinux64-Server/Unity.x86_64"
    )
    unity_server_build_path_osx: Path = (
        "../crew-dojo/Builds/HideAndSeek-StandaloneOSX-Server/Unity"
    )
    unity_server_build_path_windows: Path = (
        "../crew-dojo/Builds/HideAndSeek-StandaloneWindows-Server/Unity.exe"
    )
    log_folder_path: Path = "../UnityLogs"
    human_delay_steps: int = 2
    no_graphics: bool = False
    time_scale: float = 1.0
    seed: int = 42
    state_start_dim: int = 3
    state_end_dim: int = 9
    pretrained_encoder: bool = False
    additional_in_keys: dict = {"step_count": ("agents", "step_count")}
    crop_h: int = 100
    crop_w: int = 100
    action_low: float = -10.0
    action_high: float = 10.0
    target_img: Path = "assets/hider.png"
    scale_reward: float = 11.0
    shift_reward: float = -1.0
    dense_reward_scale: float = 1.0
    credit_window_right: float = 1.0

    @property
    def num_player_args(self) -> list[str]:
        return [
            "-NumHiders",
            f"{self.num_hiders}",
            "-NumSeekers",
            f"{self.num_seekers}",
        ]
    

@define(auto_attribs=True)
class WildfireConfig(EnvironmentConfig):
    name: str = "wildfire"
    num_agents: int = 51
    level: str = "Level_1_small"
    starting_firefighter_agents: int = 0
    starting_bulldozer_agents: int = 0
    starting_drone_agents: int = 0
    starting_helicopter_agents: int = 0
    steps_per_decision: int =10
    unity_server_build_path_linux: Path = (
        "../crew-dojo/Builds/Wildfire-StandaloneLinux64-Server/Unity.x86_64"
    )
    unity_server_build_path_osx: Path = (
        "../../../crew-dojo/Builds/Wildfire-StandaloneOSX-Server/Unity"
    )
    unity_server_build_path_windows: Path = (
        "../../../crew-dojo/Builds/Wildfire-StandaloneWindows-Server/Unity.exe"
    )
    log_folder_path: Path = "../UnityLogs"
    render_folder_path: Path = Path(__file__).resolve().parent.parent
    max_steps: int = 10
    timestamp: str = ""

    no_graphics: bool = False
    time_scale: float = 1.0

    log_trajectory: bool = True


    map_size: int = 0
    seed: int = 0


    game_type: int = 1

    ## Cut Trees
    lines: bool = True
    tree_count: int = 0
    trees_per_line: int= 0

    ## Scout Fire

    fire_spread_frequency: int = 0

    ## Pick and Place

    ## Contain Fire

    water: bool = False

    ## Rescue Civilians

    civilian_count: int = 0
    civilian_clusters: int = 0
    civilian_move_frequency: int = 0

    algorithm: str = ""

    ## Both

    known: bool = True
    vegetation_density_offset: int = 0

    graph: str = "default"  # Options: "default", "simple", "llm_generated"
    team_generation_type: str = "default"  # Computed value passed to Unity for path structure

    collaboration_mode: str = "ai_control"
    team_config: dict = {}
    lobby_id: str = "default"
    scheduled_events: list = []

    llm_model: str = "gpt"
    llm_url: str = "https://api.openai.com/v1"
    
    manager_type: str = "both"
    as_pre_generated: bool = False
    pre_generated: bool = False
    pre_generated_team_config_root: str = ""

    @property
    def num_player_args(self) -> list[str]:
        return [
            "-NumAgents",
            f"{self.num_agents}",
            "-StartingFirefighterAgents",
            f"{self.starting_firefighter_agents}",
            "-StartingBulldozerAgents",
            f"{self.starting_bulldozer_agents}",
            "-StartingDroneAgents",
            f"{self.starting_drone_agents}",
            "-StartingHelicopterAgents",
            f"{self.starting_helicopter_agents}",

        ]


    


def register_env_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="envs", name="base_bowling", node=BowlingConfig)
    cs.store(group="envs", name="base_find_treasure", node=FindTreasureConfig)
    cs.store(group="envs", name="base_hide_and_seek_1v1", node=HideAndSeek1V1Config)
    cs.store(group="envs", name="base_hide_and_seek", node=HideAndSeekConfig)
    cs.store(group="envs", name="base_wildfire", node=WildfireConfig)

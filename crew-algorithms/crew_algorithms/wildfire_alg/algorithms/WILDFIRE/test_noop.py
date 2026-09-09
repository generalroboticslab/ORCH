"""
WILDFIRE No-Op Test Script

Runs the WILDFIRE algorithm with all agents doing nothing (idle actions)
while maintaining normal logging for debugging and testing purposes.

Usage:
    python -m crew_algorithms.wildfire_alg.algorithms.WILDFIRE.test_noop envs.level=Demo_Level envs.seed=42
"""

import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets, EVENT_ACTION_MAP
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import get_agent_observations, parse_game_data
import datetime
import csv
import certifi
import asyncio
import os
import time
from typing import Dict, List, Any, Tuple

import torch
from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.wildfire_alg.core.utils import make_env
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.worker_agent import WorkerAgent
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import Option, generate_graph
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.master_logger import init_master_logger, get_master_logger
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.agent import Agent
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.horizontal_manager_agent import HorizontalManagerAgent
import uuid


@define(auto_attribs=True)
class Config:
    envs: EnvironmentConfig = MISSING
    """Settings for the environment to use."""
    wandb: WandbConfig = WandbConfig(project="wildfire")
    """WandB logger configuration."""
    collect_data: bool = False
    """Whether or not to collect data and save a new dataset to WandB."""
    llms: LLMConfig = LLMConfig()


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
register_env_configs()


async def mock_status_phase(agents: List[Any], global_data: dict, logger) -> None:
    """
    Mock status phase - sets minimal state without LLM calls, still logs.
    """
    for agent in agents:
        # Set minimal mock state
        agent.perception_summary = "NOOP test mode - no perception"
        agent.status_summary = "Idle - NOOP test mode"

        if hasattr(agent, 'percent_complete'):
            agent.percent_complete = 0.0
        if hasattr(agent, 'phase_progress'):
            agent.phase_progress = 0.0
        if hasattr(agent, 'urgent'):
            agent.urgent = ""
        if hasattr(agent, 'status'):
            agent.status = 1  # Working status

        # Log STATUS_PHASE event
        logger.log_event(
            timestep=global_data.get("timestep", 0),
            agent_id=str(agent.id),
            event_type="STATUS_PHASE",
            details={
                "mission": "NOOP_TEST",
                "mission_percent": 0.0,
                "phase": "NOOP",
                "phase_percent": 0.0,
                "decision": "IDLE_TEST",
                "urgent": "",
                "options_list": []
            },
            phase="NOOP_TEST"
        )


async def mock_action_phase(worker_agents: List[Any], global_data: dict, logger) -> None:
    """
    Mock action phase - assigns idle options without LLM calls, still logs.
    """
    for agent in worker_agents:
        # Assign idle option
        idle_option = Option(
            type=0,
            param_1=0,
            param_2=0,
            description="idle - noop test mode"
        )
        agent.options = [idle_option]
        agent.past_options = []
        if hasattr(agent, 'action_queue'):
            agent.action_queue = []

        # Log ACTION_PHASE event
        logger.log_event(
            timestep=global_data.get("timestep", 0),
            agent_id=str(agent.id),
            event_type="ACTION_PHASE",
            details={
                "decision": "NOOP_TEST",
                "mission": "IDLE",
                "options_list": ["idle - noop test mode"]
            }
        )


def generate_noop_action(agent) -> list:
    """
    Generate a no-operation action for an agent.
    Action format: [action_type, x, y]
    action_type 0 = no operation / idle
    """
    if hasattr(agent, 'last_position') and agent.last_position is not None:
        x = int(agent.last_position[0])
        y = int(agent.last_position[1])
    else:
        x, y = 0, 0
    return [0, x, y]


@hydra.main(version_base=None, config_path="../../../conf", config_name="wildfire_alg")
def test_noop(cfg: Config):
    """No-op test entry point - wraps the async implementation"""
    asyncio.run(async_test_noop(cfg))


async def async_test_noop(cfg: Config):
    """
    Async implementation of the no-op test.
    Runs the environment with all agents doing nothing.
    """
    print("=" * 60)
    print("WILDFIRE NO-OP TEST")
    print("All agents will remain idle - no LLM calls")
    print("=" * 60)

    # Validate configuration
    if not hasattr(cfg, 'envs') or cfg.envs is None:
        raise ValueError("Configuration must have 'envs' section")
    if not hasattr(cfg.envs, 'level') or cfg.envs.level is None:
        raise ValueError("Configuration must specify 'level' in envs section")
    if not hasattr(cfg.envs, 'seed') or cfg.envs.seed is None:
        raise ValueError("Configuration must specify 'seed' in envs section")

    print(f"[NOOP] Level: {cfg.envs.level}")
    print(f"[NOOP] Seed: {cfg.envs.seed}")
    print(f"[NOOP] Max steps: {cfg.envs.max_steps}")

    # Device setup
    device = "cpu" if not torch.has_cuda else "cuda:0"
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    cfg.envs.algorithm = 'WILDFIRE'
    level = cfg.envs.level
    seed = cfg.envs.seed

    # Create level presets and update config
    levels = create_level_presets()
    firefighters = levels[level].get("starting_firefighter_agents", 0)
    bulldozers = levels[level].get("starting_bulldozer_agents", 0)
    drones = levels[level].get("starting_drone_agents", 0)
    helicopters = levels[level].get("starting_helicopter_agents", 0)

    update_config(preset=levels[level], config=cfg.envs, log_trajectory=True, seed=seed)

    # Extract scheduled events for dynamic levels (empty list for standard levels)
    scheduled_events = list(cfg.envs.get('scheduled_events', None) or [])
    if scheduled_events:
        print(f"[NOOP][EVENTS] Loaded {len(scheduled_events)} scheduled events: {scheduled_events}")

    cfg.envs.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Create environment
    print("[NOOP] Creating environment...")
    env = make_env(cfg.envs, toggle_timestep_channel, device)
    state = env.reset()
    print("[NOOP] Environment created and reset")

    # Setup paths and logging
    os.environ["SSL_CERT_FILE"] = certifi.where()
    api_key = os.environ.get('OPENAI_API_KEY', 'dummy_key_noop_test')

    path = os.path.join("results", "logs", "WILDFIRE_NOOP", level, str(seed), cfg.envs.timestamp)
    os.makedirs(path, exist_ok=True)

    # Initialize master logger
    master_logger = init_master_logger(log_dir=os.path.join(path, "master_logs"))
    print(f"[NOOP] Master logger initialized at {path}/master_logs")

    # Create worker agents
    worker_agents = []
    agent_id = 1

    for i in range(firefighters):
        a = WorkerAgent(id=agent_id, type=0, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        worker_agents.append(a)
        agent_id += 1

    for i in range(bulldozers):
        a = WorkerAgent(id=agent_id, type=1, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        worker_agents.append(a)
        agent_id += 1

    for i in range(drones):
        a = WorkerAgent(id=agent_id, type=2, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        worker_agents.append(a)
        agent_id += 1

    for i in range(helicopters):
        a = WorkerAgent(id=agent_id, type=3, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        worker_agents.append(a)
        agent_id += 1

    worker_agent_count = len(worker_agents)
    print(f"[NOOP] Created {worker_agent_count} worker agents")

    # Generate simple graph (flat hierarchy for noop test)
    graph, agent_types = generate_graph(worker_agent_count, "simple")

    # Create manager agents if needed
    agents = worker_agents[:]
    for g in range(worker_agent_count, len(graph)):
        agent_type_code, team_name = agent_types[g]
        role = f"AGENT_{g+1}"

        if agent_type_code == 1:  # Horizontal manager
            a = HorizontalManagerAgent(id=g+1, name=role, cfg=cfg, path=path, api_key=api_key, team_name=team_name)
        else:  # Full manager
            a = Agent(id=g+1, type=-1, name=role, cfg=cfg, path=path, api_key=api_key, team_name=team_name)
        agents.append(a)

    # Set up hierarchy
    for manager_idx, row in enumerate(graph):
        for child_idx, has_relation in enumerate(row):
            if has_relation == 1:
                agents[manager_idx].children.append(agents[child_idx])
                agents[child_idx].parent = agents[manager_idx]

    if len(agents) > worker_agent_count:
        agents[-1].leader = True
        print(f"[NOOP] Created {len(agents) - worker_agent_count} manager agents")

    # Initialize global data
    global_data = {
        'api_calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'score': 0,
        'rewards': [0] * 13,
        'timestep': 0,
        'agents': agents
    }

    # Setup CSV logging (same columns as __main__.py)
    header = ["timestep",
              "exploration", "trees_fire", "trees_agents", "fire_extinguished",
              "correct_trees_cut", "civilians_rescued", "firefighters_transported",
              "civilians_scouted", "fire_scouted", "water_scouted",
              "agents_destroyed", "civilians_destroyed", "buildings_destroyed",
              "cumulative_api_calls", "cumulative_input_tokens", "cumulative_output_tokens", "cumulative_cost",
              "cumulative_idle_steps", "cumulative_replans", "time"]
    csv_filename = os.path.join(path, "data.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"[NOOP] CSV logging initialized at {csv_filename}")

    # Main game loop
    print(f"[NOOP] Starting game loop for {cfg.envs.max_steps} timesteps")
    game_start_time = time.time()

    for t in range(cfg.envs.max_steps):
        print(f"[NOOP] TIME: {t}")

        # Parse game data
        game_data = parse_game_data(state, cfg)

        # Update global data
        global_data.update({
            'timestep': t,
            'score': game_data['score'],
            'rewards': game_data['rewards'],
            'game_data': game_data
        })

        # Write CSV row (no API calls in noop mode)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            r = global_data['rewards']
            writer.writerow([
                t,
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                0, 0, 0, 0,                     # api_calls, input_tokens, output_tokens, cost
                0, 0,                            # idle_steps, replans
                time.time() - game_start_time
            ])

        # Update worker observations
        for worker in worker_agents:
            observations = get_agent_observations(state, worker.id)
            if observations["agent_type"] >= 4:
                print(f"[NOOP] AGENT_{worker.id} DESTROYED")
                worker.alive = False

            worker.last_observation = observations["perception_grid"]
            worker.last_position = observations["position"]
            worker.current_cell = observations["current_cell"]
            worker.map_range = observations["map_range"]
            if "extra_variables" in observations:
                worker.extra_variables = observations["extra_variables"]

        # Mock phases (no LLM calls)
        logger = get_master_logger()
        await mock_status_phase(agents, global_data, logger)
        await mock_action_phase(worker_agents, global_data, logger)

        # Generate idle actions for all workers
        env_action = [[-1, 0, 0] for _ in range(cfg.envs.num_agents)]
        for agent in worker_agents:
            env_action[agent.id] = generate_noop_action(agent)

        # Check for scheduled events at this timestep
        scheduled_event_action = None
        events_this_step = [e for e in scheduled_events if e["timestep"] == t]
        if events_this_step:
            event = events_this_step[0]
            action_code = EVENT_ACTION_MAP[event["action"]]
            x = float(event.get("x", 0))
            y = float(event.get("y", 0))
            scheduled_event_action = [action_code, x, y]
            print(f"[NOOP][EVENT] Injected manager action at t={t}: {scheduled_event_action}")

        if scheduled_event_action is not None:
            env_action[0] = scheduled_event_action

        # Log FINAL_ACTION
        logger.log_event(
            timestep=t,
            agent_id="ALL",
            event_type="FINAL_ACTION",
            details={
                "action_tensor": env_action,
                "action_description": f"NOOP actions for {len(worker_agents)} agents at timestep {t}"
            }
        )

        print(f"[NOOP] Actions: {env_action}")

        # Step environment
        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]
        state["agents"]["valid_mask"] = newstate["next"]["agents"]["valid_mask"]
        state["agents"]["done"] = newstate["next"]["agents"]["done"]

    # Cleanup
    env.close()
    print("[NOOP] Environment closed")

    # Close master logger
    try:
        master_logger.close()
    except Exception as e:
        print(f"[NOOP] Warning: Failed to close master logger: {e}")

    print("=" * 60)
    print("WILDFIRE NO-OP TEST COMPLETE")
    print(f"Results saved to: {path}")
    print(f"  - data.csv: {csv_filename}")
    print(f"  - Master logs: {path}/master_logs/")
    print("=" * 60)


if __name__ == "__main__":
    test_noop()


import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig, QwenLLMConfig, DeepseekLLMConfig, GemmaLLMConfig, GlmLLMConfig, LlamaLLMConfig, ErnieLLMConfig, NemotronLLMConfig, MinimaxLLMConfig, KimiLLMConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets, EVENT_ACTION_MAP
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import get_agent_observations, generate_action_from_option, parse_game_data, check_if_option_done, check_game_done
import datetime
import csv
import certifi
import asyncio
# Lazy import to avoid OpenGL dependency issues in Docker
# from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import generate_graph, Action, _validate_graph_and_types

import argparse
import sys
import json
import threading
import time
import base64
import io
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from crew_algorithms.wildfire_alg.libraries.firefighter_action_library import Run_Firefighter_Action
from crew_algorithms.wildfire_alg.libraries.bulldozer_action_library import Run_Bulldozer_Action
from crew_algorithms.wildfire_alg.libraries.drone_action_library import Run_Drone_Action
from crew_algorithms.wildfire_alg.libraries.helicopter_action_library import Run_Helicopter_Action


def get_leaf_worker_ids(agent, only_alive: bool = True) -> List[int]:
    """Recursively collect all leaf worker IDs under a manager agent.

    For multi-level hierarchies (manager -> manager -> workers), this traverses
    the entire subtree and returns only the leaf workers (type != -1).

    Args:
        agent: The manager agent to collect leaf workers from
        only_alive: If True, only include workers where alive=True

    Returns:
        List of worker IDs in this manager's subtree
    """
    leaf_ids = []
    for child in getattr(agent, 'children', []):
        if child.type == -1:  # Child is also a manager, recurse
            leaf_ids.extend(get_leaf_worker_ids(child, only_alive))
        else:  # Child is a worker (leaf node)
            if only_alive and not getattr(child, 'alive', True):
                continue  # Skip destroyed workers
            leaf_ids.append(child.id)
    return leaf_ids


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


# Import human interface functions from utils
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
    submit_human_actions_batch, submit_human_action, get_human_observations,
    get_all_human_observations, is_session_active, set_session_active,
    update_human_observations, clear_human_actions, get_all_human_actions
)
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.agent_snapshot import (
    AgentSnapshot, take_snapshot
)
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.chat_worker import run_chat_worker
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.event_emitter import emit_event



async def async_status_phase(agents: List[Any], global_data: dict, agent_states: Dict[int, Tuple[int, int]], graph: List[List[int]] = None) -> None:
    """
    Execute status phase with hierarchical dependencies using adjacency matrix (bottom-up)
    """
    pass  # Status phase start

    if graph is None:
        # Fallback to parallel execution if no graph provided
        pass
        status_tasks = []
        for agent in agents:
            if hasattr(agent, 'async_status_phase'):
                task = agent.async_status_phase(global_data, agent_states)
                status_tasks.append(task)
            else:
                task = asyncio.create_task(asyncio.to_thread(agent.status_phase, global_data, agent_states))
                status_tasks.append(task)
        await asyncio.gather(*status_tasks)
        return

    # Use graph for topological ordering (status phase: children first, then parents - BOTTOM-UP)
    # Calculate out-degree for each agent (how many children they have)
    out_degree = [0] * len(agents)
    for manager_idx, row in enumerate(graph):
        for child_idx, has_relation in enumerate(row):
            if has_relation == 1:
                out_degree[manager_idx] += 1

    # Start with agents that have no children (out-degree 0) - leaf nodes first
    ready_queue = [i for i in range(len(agents)) if out_degree[i] == 0]

    while ready_queue:
        # Execute all agents ready at this level in parallel
        pass
        level_tasks = []

        for agent_idx in ready_queue:
            agent = agents[agent_idx]
            if hasattr(agent, 'async_status_phase'):
                task = agent.async_status_phase(global_data, agent_states)
                level_tasks.append(task)
            else:
                task = asyncio.create_task(asyncio.to_thread(agent.status_phase, global_data, agent_states))
                level_tasks.append(task)

        # Wait for all agents at this level to complete
        await asyncio.gather(*level_tasks)
        pass

        # Update out-degrees and find next ready agents (parents of completed children)
        next_ready = []
        for agent_idx in ready_queue:
            # For each parent of this completed agent, reduce their out-degree
            for parent_idx, row in enumerate(graph):
                if row[agent_idx] == 1:  # parent_idx is parent of agent_idx
                    out_degree[parent_idx] -= 1
                    if out_degree[parent_idx] == 0:
                        next_ready.append(parent_idx)

        ready_queue = next_ready

    pass


async def async_action_phase(agents: List[Any], global_data: dict, agent_states: Dict[int, Tuple[int, int]], graph: List[List[int]] = None) -> None:
    """
    Execute action phase with hierarchical dependencies using adjacency matrix (top-down)
    """
    pass  # Action phase start

    if graph is None:
        # Fallback to parallel execution if no graph provided
        pass
        action_tasks = []
        for agent in reversed(agents):
            if hasattr(agent, 'async_action_phase'):
                task = agent.async_action_phase(global_data, agent_states)
                action_tasks.append(task)
            else:
                task = asyncio.create_task(asyncio.to_thread(agent.action_phase, global_data, agent_states))
                action_tasks.append(task)
        await asyncio.gather(*action_tasks)
        return

    # Use graph for topological ordering (action phase: parents first, then children)
    # Calculate out-degree for each agent (how many children they have)
    out_degree = [0] * len(agents)
    for manager_idx, row in enumerate(graph):
        for child_idx, has_relation in enumerate(row):
            if has_relation == 1:
                out_degree[manager_idx] += 1

    # Start with agents that have no children (out-degree 0) - but we want parents first
    # So we need to reverse the topology: start with agents that have children but whose parents are done
    in_degree = [0] * len(agents)
    for manager_idx, row in enumerate(graph):
        for child_idx, has_relation in enumerate(row):
            if has_relation == 1:
                in_degree[child_idx] += 1

    # Start with agents that have no parents (root managers)
    ready_queue = [i for i in range(len(agents)) if in_degree[i] == 0]

    while ready_queue:
        # Execute all agents ready at this level in parallel
        pass
        level_tasks = []

        for agent_idx in ready_queue:
            agent = agents[agent_idx]
            if hasattr(agent, 'async_action_phase'):
                task = agent.async_action_phase(global_data, agent_states)
                level_tasks.append(task)
            else:
                task = asyncio.create_task(asyncio.to_thread(agent.action_phase, global_data, agent_states))
                level_tasks.append(task)

        # Wait for all agents at this level to complete
        await asyncio.gather(*level_tasks)
        pass

        # Update in-degrees and find next ready agents (children of completed agents)
        next_ready = []
        for agent_idx in ready_queue:
            # For each child of this completed agent, reduce their in-degree
            for child_idx, has_relation in enumerate(graph[agent_idx]):
                if has_relation == 1:
                    in_degree[child_idx] -= 1
                    if in_degree[child_idx] == 0:
                        next_ready.append(child_idx)

        ready_queue = next_ready

    pass


def wait_for_human_actions(agents: List[Any], timeout: float = 30.0):
    """Wait for human players to submit actions"""
    if not is_session_active():
        return
        
    start_time = time.time()
    human_agent_roles = set()
    
    # Find all human agents that need actions
    for agent in agents:
        if hasattr(agent, 'is_human') and agent.is_human and hasattr(agent, 'is_worker') and agent.is_worker():
            human_agent_roles.add(agent.name)
    
    # Wait for actions from all human agents
    while (time.time() - start_time) < timeout and human_agent_roles:
        # Check which agents have submitted actions
        all_actions = get_all_human_actions()
        submitted_roles = set(all_actions.keys())
        remaining_roles = human_agent_roles - submitted_roles
        
        if not remaining_roles:
            break
            
        time.sleep(0.1)  # Brief sleep to avoid busy waiting
    
    # Apply submitted actions to agents
    all_actions = get_all_human_actions()
    for agent in agents:
        if hasattr(agent, 'is_human') and agent.is_human and agent.name in all_actions:
            agent.submit_action(all_actions[agent.name])
    
    # Clear actions for next timestep
    clear_human_actions()


@hydra.main(version_base=None, config_path="../../../conf", config_name="wildfire_alg")
def wildfire_alg(cfg: Config):
    """Main entry point - wraps the async implementation"""
    asyncio.run(async_wildfire_alg(cfg))

async def async_wildfire_alg(cfg: Config):

    """An implementation of a wildfire alg."""
    import os
    import uuid
    from pathlib import Path


    import torch
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.wildfire_alg.core.utils import (
        make_env,
    )
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.agent import Agent
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.worker_agent import WorkerAgent
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import Option
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.master_logger import init_master_logger, get_master_logger
    from torchrl.record.loggers import generate_exp_name, get_logger

    # Validate configuration
    if not hasattr(cfg, 'envs') or cfg.envs is None:
        raise ValueError("Configuration must have 'envs' section")
    
    if not hasattr(cfg.envs, 'level') or cfg.envs.level is None:
        raise ValueError("Configuration must specify 'level' in envs section")
    
    if not hasattr(cfg.envs, 'seed') or cfg.envs.seed is None:
        raise ValueError("Configuration must specify 'seed' in envs section")
    
    if not hasattr(cfg, 'llms') or cfg.llms is None:
        raise ValueError("Configuration must have 'llms' section")
    
    
    if not hasattr(cfg.envs, 'collaboration_mode') or cfg.envs.collaboration_mode is None:
        raise ValueError("Configuration must have 'collaboration_mode' section")

    # print(f"[DEBUG] Starting WILDFIRE algorithm initialization")
    # print(f"[DEBUG] Configuration loaded: level={cfg.envs.level}, seed={cfg.envs.seed}")
    
    # wandb.login()
    # exp_name = generate_exp_name("Wildfire", f"random-{cfg.envs.name}")
    # logger = get_logger(
    #     "wandb", 
    #     logger_name=os.getcwd(),
    #     experiment_name=exp_name,
    #     wandb_kwargs=dict(
    #         entity=cfg.wandb.entity,
    #         project=cfg.wandb.project,
    #         settings=wandb.Settings(start_method="thread"),
    #         tags=["baseline", cfg.envs.name],
    #     ),
    # )
    #logger.log_hparams(cfg)
    device = "cpu" if not torch.has_cuda else "cuda:0"
    
    toggle_timestep_channel = ToggleTimestepChannel(uuid.uuid4())

    cfg.envs.algorithm = 'WILDFIRE'
    
    level = cfg.envs.level
    seed  = cfg.envs.seed


    levels = create_level_presets()

    firefighters = levels[level].get("starting_firefighter_agents",0)
    bulldozers = levels[level].get("starting_bulldozer_agents",0)
    drones = levels[level].get("starting_drone_agents",0)
    helicopters = levels[level].get("starting_helicopter_agents",0)


    update_config(preset=levels[level], config=cfg.envs, log_trajectory=True, seed=seed)

    # Extract scheduled events for dynamic levels (empty list for standard levels)
    scheduled_events = list(cfg.envs.get('scheduled_events', None) or [])
    if scheduled_events:
        print(f"[EVENTS] Loaded {len(scheduled_events)} scheduled events: {scheduled_events}")

    cfg.envs.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Determine team_generation_type BEFORE make_env() so Unity gets the correct value
    team_config = getattr(cfg.envs, 'team_config', None)
    if team_config:
        team_generation_type = "preset"
    else:
        team_generation_type = getattr(cfg.envs, 'graph', 'default')
        if team_generation_type == "llm_generated":
            as_pre_generated = str(getattr(cfg.envs, "as_pre_generated", False)).lower() in {"1", "true", "yes", "y"}
            pre_generated = str(getattr(cfg.envs, "pre_generated", False)).lower() in {"1", "true", "yes", "y"}
            if as_pre_generated or pre_generated:
                team_generation_type += "_pre-generated"
            team_generation_type += f"_{'critic' if cfg.llms.use_structure_critic else 'no-critic'}"
            team_generation_type += f"_{cfg.envs.manager_type}"  # Add manager type to team_generation_type for LLM-generated graphs
    cfg.envs.team_generation_type = team_generation_type

    # print(f"[DEBUG] Environment config: level={cfg.envs.level}, seed={cfg.envs.seed}, max_steps={cfg.envs.max_steps}")


    env = make_env(cfg.envs, toggle_timestep_channel, device)
    # print(f"[DEBUG] make_env() completed successfully!")
    

    state = env.reset()

    
    # Debug: Check if we have valid observations

    print(f"State keys: {list(state.keys()) if hasattr(state, 'keys') else 'No keys'}")




    collaboration_mode = cfg.envs.collaboration_mode
    # print(f"[DEBUG] Collaboration mode: {collaboration_mode}")
    
    # if collaboration_mode in ['human_control', 'human_feedback']:
    #     print(f"[DEBUG] Human interface mode activated ({collaboration_mode})")
    #     print(f"[DEBUG] Team config: {cfg.envs.team_config}")

    os.environ["SSL_CERT_FILE"] = certifi.where()

    llm_model = cfg.envs.llm_model
    allowed = ("gpt", "qwen", "deepseek", "gemma", "glm", "llama", "ernie", "nemotron","minimax","kimi")
    if llm_model not in allowed:
        raise ValueError(f"Unsupported llm_model '{llm_model}'. Supported: {allowed}")

    # If using Qwen, apply Qwen defaults to cfg.llms
    if llm_model != "gpt":
        gcfg = eval(f"{llm_model.capitalize()}LLMConfig()")
        cfg.llms.actor_model = gcfg.actor_model
        cfg.llms.critic_model = gcfg.critic_model
        cfg.llms.planner_model = gcfg.planner_model
        cfg.llms.translator_model = gcfg.translator_model
        cfg.llms.large_model = gcfg.large_model
        cfg.llms.small_model = gcfg.small_model
        cfg.llms.reasoning_model = gcfg.reasoning_model

    # API key selection: explicit config -> provider-specific env var -> OPENAI_API_KEY
    if llm_model == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv(f"{llm_model.upper()}_API_KEY","ernie-123")
    model_name = cfg.llms.small_model 
    
    os.environ["OPENAI_API_BASE"] = cfg.envs.llm_url

    # team_generation_type already computed above (before make_env)

    path = os.path.join("crew_algorithms", "wildfire_alg", "results", "logs", "WILDFIRE", llm_model, team_generation_type, level, str(seed), cfg.envs.timestamp)
    os.makedirs(path, exist_ok=True)
    
    # Initialize master logger
    master_logger = init_master_logger(log_dir=os.path.join(path, "master_logs"))
    print(f"made master logger")

    # Initialize VLM data logger if enabled
    from crew_algorithms.wildfire_alg.data.vlm_data_logger import init_vlm_logger
    vlm_data_enabled = getattr(cfg.llms, 'collect_vlm_data', False)
    vlm_data_folder = getattr(cfg.llms, 'vlm_data_folder', 'vlm_comparison_data')
    vlm_logger = init_vlm_logger(
        base_folder=os.path.join(path, vlm_data_folder),
        enabled=vlm_data_enabled
    )
    if vlm_data_enabled:
        print(f"[VLM Data Collection] Enabled - saving to {os.path.join(path, vlm_data_folder)}")


    # Parse new team config if present (using IDs, not names, and supporting Python dict literal)
    # New format: managers: {id: {children: [...], type: "horizontal"|"vertical", team_name: "..."}}
    team_config = getattr(cfg.envs, 'team_config', None)
    humans = set()
    manager_map = {}
    manager_types = {}
    manager_team_names = {}
    if team_config:
        print(f"[DEBUG] Parsing team_config (new format): {team_config}")
        humans = set(int(x) for x in team_config.get('humans', []))

        # Parse managers in new format: {id: {children: [...], type: "...", team_name: "..."}}
        raw_managers = team_config.get('managers', {})
        for k, v in raw_managers.items():
            manager_id = int(k)
            # New format: v is a dict with children, type, team_name
            manager_map[manager_id] = [int(i) for i in v['children']]
            manager_types[manager_id] = v['type']  # "horizontal" or "vertical"
            manager_team_names[manager_id] = v['team_name']

        print(f"[DEBUG] Humans (IDs): {humans}")
        print(f"[DEBUG] Manager map (IDs): {manager_map}")
        print(f"[DEBUG] Manager types: {manager_types}")
        print(f"[DEBUG] Manager team names: {manager_team_names}")
    # Create worker agents in the original order
    worker_agents = []
    agent_id = 1
    # Firefighters
    for i in range(firefighters):
        a = WorkerAgent(id=agent_id, type=0, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        a.human = agent_id in humans
        worker_agents.append(a)
        agent_id += 1
    # Bulldozers
    for i in range(bulldozers):
        a = WorkerAgent(id=agent_id, type=1, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        a.human = agent_id in humans
        worker_agents.append(a)
        agent_id += 1
    # Drones
    for i in range(drones):
        a = WorkerAgent(id=agent_id, type=2, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        a.human = agent_id in humans
        worker_agents.append(a)
        agent_id += 1
    # Helicopters
    for i in range(helicopters):
        a = WorkerAgent(id=agent_id, type=3, name=f"AGENT_{agent_id}", cfg=cfg, path=path, api_key=api_key)
        a.human = agent_id in humans
        worker_agents.append(a)
        agent_id += 1
    worker_agent_count = firefighters + bulldozers + drones + helicopters

    # Build the graph (adjacency matrix)
    if team_config:
        total_agents = worker_agent_count + len(manager_map)
        graph = [[0 for _ in range(total_agents)] for _ in range(total_agents)]
        for manager_id, children_ids in manager_map.items():
            for child_id in children_ids:
                graph[manager_id-1][child_id-1] = 1
        print("[DEBUG] Adjacency matrix (graph) from manager_map:")
        for row in graph:
            print(row)

        # Generate agent_types for the first approach
        agent_types = []

        # Add worker agents (type 0, no team name)
        for i in range(worker_agent_count):
            agent_types.append((0, ""))

        # Add manager agents with type and team_name from config
        for manager_id in sorted(manager_map.keys()):
            # Get manager type from config: "horizontal" -> 1, "vertical" -> 2
            manager_type_str = manager_types[manager_id]
            manager_type_code = 1 if manager_type_str == 'horizontal' else 2
            # Get team name from config
            team_name = manager_team_names[manager_id]
            agent_types.append((manager_type_code, team_name))

        print(f"[DEBUG] Generated agent_types: {agent_types}")

        # Validate the graph and agent_types
        _validate_graph_and_types(graph, agent_types)
        print("[DEBUG] agent_types validation passed")
    else:
        print(f"[DEBUG] No team_config, falling back to default logic for managers/hierarchy")
        graph_mode = getattr(cfg.envs, 'graph', 'default')

        if graph_mode == "llm_generated":
            try:
                print(f"Generating LLM-GENERATED graph for level: {level} with critic: {cfg.llms.use_structure_critic}")
                graph, agent_types = generate_graph(worker_agent_count, "llm_generated", api_key=api_key, cfg=cfg, worker_agents=worker_agents)
                print(f"Graph: {graph}")
            except Exception as e:
                print(f"[ERROR] Failed to generate graph: {e}")
                exit(1)
        elif graph_mode == "simple":
            print(f"Generating SIMPLE graph for level: {level}")
            graph, agent_types = generate_graph(worker_agent_count, "simple")
            print(f"Graph: {graph}")
        else:  # "default" or level-specific
            print(f"Generating graph for level: {level}")
            graph, agent_types = generate_graph(worker_agent_count, level)
            print(f"Graph: {graph}")


    # Now create manager agents for all nodes in the graph beyond the worker count
    # Use agent_types to determine what type of manager to create
    from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.horizontal_manager_agent import HorizontalManagerAgent

    agents = worker_agents[:]
    for g in range(worker_agent_count, len(graph)):
        agent_type_code, team_name = agent_types[g]
        role = f"AGENT_{g+1}"

        # Create appropriate agent type based on agent_type_code
        if agent_type_code == 1:  # Horizontal manager
            print(f"[DEBUG] Creating Horizontal Manager agent {g+1} for {team_name}")
            a = HorizontalManagerAgent(id=g+1, name=role, cfg=cfg, path=path, api_key=api_key, team_name=team_name)
        elif agent_type_code == 2:  # Full/vertical manager
            print(f"[DEBUG] Creating Full Manager agent {g+1} for {team_name}")
            a = Agent(id=g+1, type=-1, name=role, cfg=cfg, path=path, api_key=api_key, team_name=team_name)
        else:
            raise ValueError(f"Invalid agent type code {agent_type_code} for manager agent {g+1}")

        a.human = (g+1) in humans
        agents.append(a)
        print(f"[DEBUG] Manager agent created: {a.id}, type: {agent_type_code}, team: {team_name}, human: {a.human}")

    # Set up hierarchy from adjacency matrix (graph)
    for manager_idx, row in enumerate(graph):
        for child_idx, has_relation in enumerate(row):
            if has_relation == 1:
                agents[manager_idx].children.append(agents[child_idx])
                agents[child_idx].parent = agents[manager_idx]
    print("[DEBUG] Human agents (IDs):", [a.id for a in agents if getattr(a, 'human', False)])
    print("[DEBUG] Manager hierarchy (IDs):")
    for manager_id, children_ids in manager_map.items():
        print(f"  {manager_id} -> {children_ids}")
    
    agents[-1].leader = True

    # ---------------------------------------------------------------------------
    # Chat system shared state
    # ---------------------------------------------------------------------------
    lobby_id = getattr(cfg.envs, 'lobby_id', 'default')

    agent_snapshots: Dict[int, AgentSnapshot] = {}
    slow_status_queues: Dict[int, asyncio.Queue] = {a.id: asyncio.Queue() for a in agents}
    fast_feedback_queues: Dict[int, asyncio.Queue] = {
        a.id: asyncio.Queue()
        for a in agents
        if getattr(a, 'leader', False)
    }
    phase_tracker: Dict[str, Any] = {
        "current_phase": "idle",
        "timestep": 0,
        "manager_decision_done": False,
    }
    stop_chat_event = asyncio.Event()

    # Wire queues and lobby_id into each agent
    for agent in agents:
        agent._slow_status_queue = slow_status_queues[agent.id]
        if agent.id in fast_feedback_queues:
            agent._fast_feedback_queue = fast_feedback_queues[agent.id]
        agent._lobby_id = lobby_id

    def _write_phase_status(phase: str, timestep: int) -> None:
        """Write current phase status to /tmp/phase_status_{lobby_id}.json."""
        import tempfile as _tmp
        status_file = os.path.join(_tmp.gettempdir(), f"phase_status_{lobby_id}.json")
        try:
            with open(status_file, "w") as _f:
                json.dump({"current_phase": phase, "timestep": timestep}, _f)
        except Exception as _e:
            print(f"[DEBUG] Failed to write phase_status file: {_e}")

    # Start chat worker task (only in human_feedback or ai_control modes)
    chat_task: Optional[asyncio.Task] = None
    if collaboration_mode in ['human_feedback', 'ai_control']:
        chat_task = asyncio.create_task(
            run_chat_worker(
                lobby_id=lobby_id,
                agent_snapshots=agent_snapshots,
                slow_status_queues=slow_status_queues,
                fast_feedback_queues=fast_feedback_queues,
                phase_tracker=phase_tracker,
                stop_event=stop_chat_event,
                api_key=api_key,
                cfg=cfg,
                output_path=path,
            )
        )

    global_data = {}
    global_data.update({
        "api_calls" : 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "score": 0,
        "rewards": [0] * 13,
        "agents": agents,
        "timestep": 0,
        "results_path": path  # Add results path for VLM logging
    })
    print("Agent Count: " + str(len(agents)))

    cumulative_idle_steps = 0
    cumulative_replans = 0

    header = ["timestep",
              "exploration", "trees_fire", "trees_agents", "fire_extinguished",
              "correct_trees_cut", "civilians_rescued", "firefighters_transported",
              "civilians_scouted", "fire_scouted", "water_scouted",
              "agents_destroyed", "civilians_destroyed", "buildings_destroyed",
              "cumulative_api_calls", "cumulative_input_tokens", "cumulative_output_tokens", "cumulative_cost",
              "cumulative_idle_steps", "cumulative_replans", "time"]
    csv_filename = os.path.join(path, f"data.csv")

    with open(csv_filename, 'w', newline='') as f:
          writer = csv.writer(f)
          writer.writerow(header)

    f.close()

    print(f"Max Steps {cfg.envs.max_steps}")
    _past_tldrs = []
    
    # Activate session if in human interface mode
    if collaboration_mode in ['human_control', 'human_feedback']:
        set_session_active(True)
        print(f"Human interface mode activated ({collaboration_mode})")

    def _write_observations(
        _agents: List[Any],
        _t: int,
        _lobby_id: str,
        _collaboration_mode: str,
        _game_data: dict,
        _path: str,
    ) -> None:
        """Build per-agent obs_dicts and flush to the shared observations JSON file.

        Called twice per timestep: once before the status/action phases (so the
        frontend sees fresh pre-phase data) and once after the action phase
        completes (so the frontend receives a consistent post-phase snapshot).
        """
        if _collaboration_mode not in ["human_control", "human_feedback"]:
            return
        _MINIMAP_RANGES = [10, 10, 25, 30, 0, 0]  # MapManager.miniMapRanges
        _agent_lookup = {a.id: a for a in _agents}
        for _agent in _agents:
            if _agent.type == -1:  # Manager agent
                _children_ids = [
                    child.id for child in getattr(_agent, "children", [])
                ]
                _children_names = [
                    child.name for child in getattr(_agent, "children", [])
                ]
                _leaf_worker_ids = get_leaf_worker_ids(_agent, only_alive=True)
                _is_top_level = getattr(_agent, "parent", None) is None
                _obs_dict = {
                    "timestep": _t,
                    "id": _agent.id,
                    "name": _agent.name,
                    "type": _agent.type,
                    "children_ids": _children_ids,
                    "children_names": _children_names,
                    "leaf_worker_ids": _leaf_worker_ids,
                    "is_top_level": _is_top_level,
                    "base_path": _path,
                    "map_size": cfg.envs.map_size,
                    "worker_positions": {
                        wid: {
                            "position": getattr(
                                _agent_lookup.get(wid), "last_position", None
                            ),
                            "type": getattr(
                                _agent_lookup.get(wid), "type", 0
                            ),
                            "minimap_range": _MINIMAP_RANGES[
                                getattr(_agent_lookup.get(wid), "type", 0)
                            ],
                        }
                        for wid in _leaf_worker_ids
                    },
                    "perception_summary": getattr(
                        _agent, "perception_summary", None
                    ),
                    "status_summary": getattr(_agent, "status_summary", None),
                    "mission": getattr(_agent, "mission", None),
                    "current_phase": getattr(_agent, "current_phase", None),
                    "phase_history": getattr(_agent, "phase_history", None),
                    "future_phases": getattr(_agent, "future_phases", None),
                    "phase_progress": getattr(_agent, "phase_progress", None),
                    "phase_completion_condition": getattr(
                        _agent, "phase_completion_condition", None
                    ),
                    "percent_complete": getattr(
                        _agent, "percent_complete", None
                    ),
                    "urgent": getattr(_agent, "urgent", None),
                    "task_description": _game_data.get(
                        "task_description", "No task description available"
                    ),
                }
                if _is_top_level:
                    _obs_dict["rewards"] = _game_data.get("rewards", [0] * 13)
                    _obs_dict["game_type"] = cfg.envs.game_type
                    _obs_dict["reward_target"] = _get_reward_target()
            else:  # Worker agent
                _obs_dict = {
                    "timestep": _t,
                    "id": _agent.id,
                    "name": _agent.name,
                    "type": _agent.type,
                    "base_path": _path,
                    "alive": getattr(_agent, "alive", True),
                    "last_position": getattr(_agent, "last_position", None),
                    "last_current_cell": getattr(
                        _agent, "last_current_cell", None
                    ),
                    "map_range": getattr(_agent, "map_range", None),
                    "map_size": cfg.envs.map_size,
                    "minimap_range": _MINIMAP_RANGES[_agent.type]
                    if _agent.type in range(4)
                    else 0,
                    "extra_variables": getattr(
                        _agent, "extra_variables", None
                    ),
                    "perception_summary": getattr(
                        _agent, "perception_summary", None
                    ),
                    "status_summary": getattr(_agent, "status_summary", None),
                    "mission": getattr(_agent, "mission", None),
                    "percent_complete": getattr(
                        _agent, "percent_complete", None
                    ),
                    "urgent": getattr(_agent, "urgent", None),
                    "options": [
                        {
                            "description": (
                                opt.description
                                if hasattr(opt, "description")
                                else str(opt)
                            )
                        }
                        for opt in getattr(_agent, "options", [])
                    ],
                    "option_history": [
                        {
                            "description": (
                                opt.description
                                if hasattr(opt, "description")
                                else str(opt)
                            )
                        }
                        for opt in getattr(_agent, "past_options", [])
                    ],
                    "task_description": _game_data.get(
                        "task_description", "No task description available"
                    ),
                }

            if _collaboration_mode == "human_feedback":
                _obs_dict["llm_chats"] = getattr(_agent, "timestep_chats", [])

            update_human_observations(str(_agent.id), _obs_dict)

        import tempfile as _tempfile

        _obs_file = os.path.join(
            _tempfile.gettempdir(), f"observations_{_lobby_id}.json"
        )
        try:
            from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
                get_all_human_observations,
            )

            with open(_obs_file, "w") as _f:
                json.dump(get_all_human_observations(), _f)
        except Exception as _e:
            print(f"[DEBUG] Failed to write observations file: {_e}")

    def _get_reward_target() -> dict:
        """Return {index, target, label} for the current game type."""
        _gt = cfg.envs.game_type
        if _gt == 0:
            return {
                "index": 4,
                "target": cfg.envs.tree_count * cfg.envs.trees_per_line * 3,
                "label": "Trees Cut",
            }
        elif _gt == 1:
            return {"index": 8, "target": 2, "label": "Fire Scouted"}
        elif _gt == 2:
            return {
                "index": 6,
                "target": cfg.envs.starting_firefighter_agents,
                "label": "Agents Transported",
            }
        elif _gt == 4:
            return {
                "index": 5,
                "target": civilian_target,
                "label": "Civilians Rescued",
            }
        else:
            return {"index": -1, "target": 0, "label": "Ongoing"}

    # Time Loop
    game_start_time = time.time()
    civilian_target = cfg.envs.civilian_count * cfg.envs.civilian_clusters
    _pending_announcement = None  # Deferred announcement from previous timestep's scheduled event
    _announcement_history = []  # Accumulated history of all past announcements
    for t in range(cfg.envs.max_steps):

        print(f"TIME: {t}")
        game_data = parse_game_data(state, cfg)
        removelist = []
        past_score = global_data["score"]

        global_data.update({
            'firefighters': [],
            'bulldozers': [],
            'drones': [],
            'helicopters': [],
            'time': t,
            'score': game_data['score'],
            'rewards': game_data['rewards'],
            'game_data': game_data,  # Add game data to global data
            'timestep': t
        })

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            r = global_data['rewards']
            writer.writerow([
                global_data['timestep'],
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                global_data['api_calls'], global_data['input_tokens'], global_data['output_tokens'],
                global_data['input_tokens']*0.0000025 + global_data['output_tokens']*0.00001,
                cumulative_idle_steps, cumulative_replans,
                (time.time() - game_start_time)
            ])
        f.close()

        # Emit deferred announcement from previous timestep's scheduled event
        # (delayed so observations reflect the event's effect when the popup shows)
        if _pending_announcement is not None:
            emit_event(
                lobby_id,
                "announcement",
                agent_id=0,
                agent_name="GAME",
                timestep=t,
                detail=_pending_announcement["text"],
                data=_pending_announcement["data"],
            )
            global_data["active_announcements"] = [_pending_announcement["text"]]
            _announcement_history.append({"text": _pending_announcement["text"], "timestep": t})
            print(f"[EVENT] t={t}: (deferred) {_pending_announcement['text']}")
            _pending_announcement = None
        else:
            global_data["active_announcements"] = []

        global_data["announcement_history"] = _announcement_history

        # Check for scheduled dynamic game events at this timestep
        scheduled_event_action = None
        events_this_step = [e for e in scheduled_events if e["timestep"] == t]
        if events_this_step:
            event = events_this_step[0]
            action_code = EVENT_ACTION_MAP[event["action"]]
            x = float(event.get("x", 0))
            y = float(event.get("y", 0))
            scheduled_event_action = [action_code, x, y]

            announcement_text = event.get("announcement", f"Game event: {event['action']}")

            # Defer announcement to next timestep so observations match
            _pending_announcement = {
                "text": announcement_text,
                "data": {"event_action": event["action"], "x": x, "y": y},
            }
            print(f"[EVENT] t={t}: scheduled '{announcement_text}' (will announce at t={t+1})")

            # Update civilian target when new civilians are spawned
            if event["action"] == "spawn_civilian":
                civilian_target += 1
            elif event["action"] == "spawn_civilians_5":
                civilian_target += 5

        global_data["civilian_target"] = civilian_target

        # Update Workers
        agent_states = {}
        
        
        for worker in worker_agents:

            observations = get_agent_observations(state, worker.id)
            if observations["agent_type"] >=4:
                print(f"AGENT_{worker.id} DESTROYED")
                worker.alive = False
                removelist.append(worker)

                # Emit event for fire deaths (scheduled destroy_agent already has announcement)
                was_scheduled = any(
                    e["action"] == "destroy_agent" and int(e.get("x", -1)) == worker.id
                    for e in scheduled_events
                )
                if not was_scheduled:
                    worker_type_names = ["Firefighter", "Bulldozer", "Drone", "Helicopter"]
                    emit_event(
                        lobby_id,
                        "agent_destroyed",
                        agent_id=worker.id,
                        agent_name=worker.name,
                        timestep=t,
                        detail=f"{worker.name} ({worker_type_names[worker.type]}) has been destroyed!",
                    )
                continue
            elif observations["agent_type"] ==0:
                global_data["firefighters"].append(worker)
            elif observations["agent_type"] ==1:
                global_data["bulldozers"].append(worker)
            elif observations["agent_type"] ==2:
                global_data["drones"].append(worker)
            elif observations["agent_type"] ==3:
                global_data["helicopters"].append(worker)

            worker.last_observation = observations["perception_grid"]
            worker.last_position = observations["position"]
            worker.last_current_cell = observations["current_cell"]
            worker.map_range = observations["map_range"]
            worker.extra_variables = observations["extra_variables"]
            
            # Add error handling for None values
            if worker.last_position is not None:
                agent_states.update({worker.id: worker.last_position})
            else:
                print(f"Warning: AGENT_{worker.id} has no position data")
                agent_states.update({worker.id: (0, 0)})  # Default position
            
            check_if_option_done(agent=worker)

            if worker.type==0 and worker.extra_variables[2]==1:
                worker.options = [Option(type=0, param_1=0, param_2=0,description="ride helicopter")]

        # Log VLM comparison data for all workers (works in all modes: human_control, human_feedback, ai_control)
        from crew_algorithms.wildfire_alg.data.vlm_data_logger import log_worker_observation
        for worker in worker_agents:
            if worker.alive and worker.last_observation and worker.last_position:
                try:
                    log_worker_observation(
                        agent_id=worker.id,
                        timestep=t,
                        ascii_grid=worker.last_observation,
                        agent_position=worker.last_position,
                        agent_type=worker.type,
                        map_range=worker.map_range,
                        extra_variables=worker.extra_variables,
                        results_path=path
                    )
                except Exception as e:
                    # Silently fail to not disrupt normal operation
                    pass

        # Update observations for ALL agents (both human and AI) — pre-phase snapshot
        print("UPDATING OBSERVATIONS")
        _write_observations(agents, t, lobby_id, collaboration_mode, game_data, path)

        for r in removelist:
            worker_agents.remove(r)

        global_data.update({"workers":worker_agents})

        # Check Game
        if check_game_done(global_data=global_data, cfg= cfg.envs, past_score=past_score):
            break


        for a in agents:
            a.time = t

        # Set root manager's mission on first timestep (before status phase sees it)
        if t == 0:
            agents[-1].mission = game_data["task_description"]
            print(f"[INIT] Root manager mission set: {agents[-1].mission}")

        # Status Phase - Parallel execution with barrier
        pass
        phase_tracker["current_phase"] = "status"
        phase_tracker["timestep"] = t
        phase_tracker["manager_decision_done"] = False
        _write_phase_status("status", t)

        # Execute status phase for all agents in parallel
        # In human_control mode, disable AI processing; in human_feedback mode, allow AI processing
        if collaboration_mode == 'ai_control' or collaboration_mode == 'human_feedback':
            # Execute all agents with hierarchical dependencies using the graph
            # Feedback is now queue-based (via _slow_status_queue / _fast_feedback_queue)
            await async_status_phase(agents, global_data, agent_states, graph)

            # Update snapshots after status phase completes
            for agent in agents:
                agent_snapshots[agent.id] = take_snapshot(agent, t)
            phase_tracker["manager_decision_done"] = True

        # Action Phase - Parallel execution with barrier
        pass
        phase_tracker["current_phase"] = "action"
        phase_tracker["manager_decision_done"] = False
        _write_phase_status("action", t)

        # Handle root manager requesting new mission (status=0 means "requesting")
        # This catches the case where the leader completed its mission and wants a new one
        if agents[-1].status == 0:
            agents[-1].mission = game_data["task_description"]
            agents[-1].status = 5
            print(f"[MISSION] Root manager assigned mission: {agents[-1].mission[:80]}")

        # Execute action phase for all agents in parallel (only AI agents)
        # In human_control mode, disable AI processing; in human_feedback mode, allow AI processing
        if collaboration_mode == 'ai_control' or collaboration_mode == 'human_feedback':
            # Execute all agents with hierarchical dependencies using the graph
            await async_action_phase(agents, global_data, agent_states, graph)

            # Count manager replans this timestep
            for agent in agents:
                if agent.type == -1 and getattr(agent, '_did_replan', False):
                    cumulative_replans += 1

            # Update snapshots after action phase
            for agent in agents:
                agent_snapshots[agent.id] = take_snapshot(agent, t)

            # Post-action observation flush — gives frontend a consistent snapshot
            # that reflects decisions made during this full status+action cycle.
            _write_observations(
                agents, t, lobby_id, collaboration_mode, game_data, path
            )

            # Generate a TLDR summary for the human operator after each full cycle.
            if collaboration_mode == "human_feedback":
                try:
                    from openai import AsyncOpenAI as _AsyncOpenAI
                    import tempfile as _tmp_tldr

                    _root = agents[-1]

                    # Collect timestep_chats from the root manager only.
                    _all_chats = []
                    _root_chats = getattr(_root, "timestep_chats", [])
                    for _chat in _root_chats:
                        _all_chats.append(
                            f"{_chat.get('interaction', 'Unknown')}: "
                            f"{_chat.get('response', '')[:500]}"
                        )

                    if not _all_chats:
                        print(f"[TLDR] t={t}: No timestep chats, skipping TLDR")
                    else:
                        _current_data = "\n\n".join(_all_chats)
                        _prev_summary = ""
                        if _past_tldrs:
                            _prev_lines = [
                                f"[Turn {i+1}] {txt}"
                                for i, txt in enumerate(_past_tldrs[-5:])
                            ]
                            _prev_summary = (
                                "Previous updates:\n"
                                + "\n".join(_prev_lines)
                                + "\n\n"
                            )
                        _tldr_context = (
                            _prev_summary + "Current turn data:\n" + _current_data
                        )

                        # Force TLDR on first turn, replan, or fast feedback processed
                        _is_first = len(_past_tldrs) == 0
                        _had_replan = getattr(_root, "_did_replan", False)
                        _had_fast_feedback = len(getattr(_root, "_processed_fast_feedback", [])) > 0
                        _force_tldr = _is_first or _had_replan or _had_fast_feedback

                        if _force_tldr:
                            _tldr_system = (
                                "You are writing turn-by-turn updates for a wildfire "
                                "simulation operator. You MUST write an update — do "
                                "NOT respond with 'SKIP'. If this is the first update, "
                                "write 2-3 sentences summarizing the situation. "
                                "Otherwise write 1 concise sentence about what changed."
                            )
                        else:
                            _tldr_system = (
                                "You are writing turn-by-turn updates for a wildfire "
                                "simulation operator. Your DEFAULT response should be "
                                "'SKIP'. Only write if something significantly changed "
                                "— new plans, major fire spread, completed objectives, "
                                "or human feedback was received. If the data mentions a "
                                "human directive or urgent feedback, you MUST include "
                                "it — never skip human feedback. If the situation is "
                                "largely the same, respond with exactly 'SKIP'. When "
                                "you do write, keep it to 1 sentence. Do not repeat "
                                "previous updates."
                            )

                        if model_name.find("gpt") != -1:
                            _tldr_resp = await _AsyncOpenAI(
                                api_key=api_key,
                                timeout=3600,
                            ).chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": _tldr_system},
                                    {"role": "user", "content": _tldr_context},
                                ],
                            )
                        else:
                            _tldr_client = _AsyncOpenAI(
                                api_key=api_key,
                                base_url=cfg.envs.llm_url,
                                timeout=3600,
                            )
                            _tldr_resp = await _tldr_client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": _tldr_system},
                                    {"role": "user", "content": _tldr_context},
                                ],
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                            )
                        _tldr_text = (
                            _tldr_resp.choices[0].message.content or ""
                        ).strip()
                        if _tldr_text and _tldr_text.upper() != "SKIP":
                            _past_tldrs.append(_tldr_text)
                            _chat_out = os.path.join(
                                _tmp_tldr.gettempdir(), f"chat_out_{lobby_id}.jsonl"
                            )
                            _tldr_record = {
                                "in_reply_to": f"tldr_t{t}",
                                "type": "tldr",
                                "content": _tldr_text,
                                "target_agent_id": _root.id,
                                "timestamp": datetime.datetime.now().isoformat(),
                            }
                            with open(_chat_out, "a") as _cf:
                                _cf.write(json.dumps(_tldr_record) + "\n")
                            print(f"[TLDR] t={t}: {_tldr_text[:80]}")
                        else:
                            print(f"[TLDR] t={t}: Nothing notable, skipping TLDR")
                except Exception as _tldr_err:
                    print(f"[TLDR] Failed to generate TLDR at t={t}: {_tldr_err}")

        # END Algorithm

        env_action = [[-1,0,0] for _ in range(cfg.envs.num_agents)]

        # Fill in AI agent actions

        # In human_control mode, disable AI processing; in human_feedback mode, allow AI processing
        if collaboration_mode == 'ai_control' or collaboration_mode == 'human_feedback':
            for agent in worker_agents:

                    if agent.type==0 and agent.extra_variables[2]==1:
                        agent.options = [Option(type=0, param_1=0, param_2=0,description="ride helicopter")]
                    if len(agent.options) == 0:
                        agent.options = [Option(type=0, param_1=0, param_2=0,description="idle remaining on standby")]

                    # Detect idle (exclude helicopter passengers — riding is active transport)
                    _is_helicopter_passenger = (agent.type == 0 and agent.extra_variables[2] == 1)
                    _is_idle = (agent.options[0].type == 0) and not _is_helicopter_passenger

                    agent.log_chat("Executing Actions", [("system", agent.options[0])])
                    print(f"AGENT_{agent.id}: {agent.options[0].description}")
                    action_array = generate_action_from_option(agent=agent)
                    env_action[agent.id] = action_array

                    # Catch error-forced idle from generate_action_from_option
                    if not _is_idle and not _is_helicopter_passenger and getattr(agent, '_error_idle', False):
                        _is_idle = True
                    if _is_idle:
                        cumulative_idle_steps += 1
                        agent.idle_steps = getattr(agent, 'idle_steps', 0) + 1
                    else:
                        agent.idle_steps = 0

                    agent.log_chat(f"TIME = {t}", [("system", f"{t}: {action_array}")])


        # Wait for human actions if in human_control mode (AFTER AI action phases)
        if collaboration_mode == 'human_control':
            print("WAITING FOR HUMAN ACTIONS (file-based)")
            actions_file = os.path.join(tempfile.gettempdir(), f"actions_{lobby_id}_{t}.json")
            start_time = time.time()
            expected_human_ids = [str(agent.id) for agent in worker_agents if getattr(agent, 'human', False)]
            print(f"[DEBUG] Expected human agent IDs: {expected_human_ids}")
            print(f"[DEBUG] env_action length: {len(env_action)}")
            print(f"[DEBUG] Worker agent IDs: {[agent.id for agent in worker_agents]}")
            print(f"[DEBUG] Looking for actions file: {actions_file}")
            timeout = 600  # seconds
            all_actions = {}
            while True:
                if os.path.exists(actions_file):
                    try:
                        with open(actions_file, "r") as f:
                            all_actions = json.load(f)
                        print(f"[DEBUG] Read actions from file: {all_actions}")
                        print(f"[DEBUG] Keys in all_actions: {list(all_actions.keys())}")
                        if len(all_actions.keys()) == len(expected_human_ids):
                            print(f"[DEBUG] All expected human actions found, breaking loop")
                            break
                    except Exception as e:
                        print(f"[DEBUG] Error reading actions file: {e}")
                if time.time() - start_time > timeout:
                    print("Timeout waiting for human actions")
                    break
                time.sleep(0.1)
            # Assign actions to env_action for human agents
            for agent in worker_agents:
                if getattr(agent, 'human', True) and str(agent.id) in all_actions:
                    print(f"[DEBUG] Assigning action for agent {agent.id}: {all_actions[str(agent.id)]}")
                    if agent.id <= len(env_action):

                        libraries = {
                            0: Run_Firefighter_Action,
                            1: Run_Bulldozer_Action,
                            2: Run_Drone_Action,
                            3: Run_Helicopter_Action
                        }
                        env_action[agent.id] = libraries[agent.type](agent, Action(type=all_actions[str(agent.id)][0], param_1=all_actions[str(agent.id)][1], param_2=all_actions[str(agent.id)][2], description=""))
                        #env_action[agent.id] = all_actions[str(agent.id)]
                        print(f"[DEBUG] Successfully assigned action to env_action[{agent.id}]")
                    else:
                        print(f"[DEBUG] ERROR: agent.id {agent.id} is out of bounds for env_action length {len(env_action)}")
                else:
                    print(f"[DEBUG] No action found for agent {agent.id} (human: {getattr(agent, 'human', False)})")
            # Remove the actions file after reading
            if os.path.exists(actions_file):
                try:
                    os.remove(actions_file)
                except Exception as e:
                    print(f"[DEBUG] Failed to remove actions file: {e}")

        # Inject scheduled event into pseudo-manager slot (index 0)
        if scheduled_event_action is not None:
            env_action[0] = scheduled_event_action
            print(f"[EVENT] Injected manager action: {scheduled_event_action}")

        print(env_action)

        # Log final action tensor
        try:
            logger = get_master_logger()
            logger.log_event(
                timestep=t,
                agent_id="ALL",
                event_type="FINAL_ACTION",
                details={
                    "action_tensor": env_action,
                    "action_description": f"Actions for {len(worker_agents)} agents at timestep {t}"
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log final action tensor: {e}")

        phase_tracker["current_phase"] = "env_step"
        _write_phase_status("env_step", t)

        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]
        state["agents"]["valid_mask"] = newstate["next"]["agents"]["valid_mask"]
        state["agents"]["done"] = newstate["next"]["agents"]["done"]
        print("NEXT TIMESTEP")


    env.close()
    print("TEST COMPLETE")

    # Stop chat worker
    stop_chat_event.set()
    if chat_task is not None:
        try:
            await asyncio.wait_for(chat_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            chat_task.cancel()
    _write_phase_status("finished", -1)

    # Deactivate session if in human interface mode
    if collaboration_mode in ['human_control', 'human_feedback']:
        set_session_active(False)
        print(f"Human interface mode deactivated ({collaboration_mode})")
    
    # Close master logger
    try:
        master_logger.close()
    except Exception as e:
        print(f"Warning: Failed to close master logger: {e}")
    
    # Try to compile video only if OpenGL is available
    try:
        from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video
        compile_split_screen_video(path, os.path.join(path, "render.mp4"))
    except ImportError as e:
        print(f"Warning: Could not compile video due to missing dependencies: {e}")
    except Exception as e:
        print(f"Warning: Failed to compile video: {e}")


if __name__ == "__main__":
    try:
        wildfire_alg()
    except Exception as e:
        print(f"Error in wildfire_alg: {e}")
        exit(1)

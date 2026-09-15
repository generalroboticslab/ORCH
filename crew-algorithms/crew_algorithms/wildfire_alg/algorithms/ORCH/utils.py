import json
import re
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import os
import threading
import asyncio
from typing import Dict, List, Any
from crew_algorithms.wildfire_alg.config.configs import model_directory_name


def _get_llm_model(cfg) -> str:
    envs = getattr(cfg, "envs", cfg)
    return getattr(envs, "llm_model", "gpt")


def _get_llm_url(cfg):
    envs = getattr(cfg, "envs", cfg)
    return getattr(envs, "llm_url", None)

def _get_reasoning_effort(cfg):
    envs = getattr(cfg, "envs", cfg)
    effort = getattr(envs, "reasoning_effort", None)
    return None if effort in {None, "none"} else effort

def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)

def _use_pre_generated_team_config(cfg) -> bool:
    envs = getattr(cfg, "envs", cfg)
    return _as_bool(getattr(envs, "as_pre_generated", False)) or _as_bool(getattr(envs, "pre_generated", False))

def _pre_generated_variant(cfg) -> str:
    envs = getattr(cfg, "envs", cfg)
    manager_type = getattr(envs, "manager_type", "both")
    if manager_type in {"vertical", "full"}:
        return "full"
    if manager_type == "horizontal":
        return "horizontal"
    return "both"

def _team_config_manager_id(raw_id) -> int:
    text = str(raw_id)
    if text.startswith("AGENT_"):
        return int(text.split("_", 1)[1])
    return int(text)

def _team_config_child_id(raw_id) -> int:
    text = str(raw_id)
    if text.startswith("AGENT_"):
        return int(text.split("_", 1)[1])
    return int(text)

def _default_pre_generated_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "team_configs", "LLM_generated")
    )

def _pre_generated_team_config_path(cfg, level: str) -> str:
    envs = getattr(cfg, "envs", cfg)
    root = getattr(envs, "pre_generated_team_config_root", "") or _default_pre_generated_root()
    critic_dir = "critic" if getattr(cfg.llms, "use_structure_critic", True) else "no_critic"
    variant = _pre_generated_variant(cfg) if critic_dir == "critic" else "both"
    return os.path.join(root, model_directory_name(cfg), critic_dir, variant, f"{level}.json")

def team_config_to_graph(team_config: dict, worker_count: int) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Convert saved team_config JSON into graph and agent_types."""
    raw_managers = team_config.get("managers", {})
    managers = {}
    for raw_manager_id, manager_cfg in raw_managers.items():
        manager_id = _team_config_manager_id(raw_manager_id)
        managers[manager_id] = {
            "children": [_team_config_child_id(child_id) for child_id in manager_cfg.get("children", [])],
            "type": manager_cfg.get("type", "vertical"),
            "team_name": manager_cfg.get("team_name", f"TEAM_{manager_id}"),
        }

    total_agents = worker_count + len(managers)
    expected_ids = set(range(worker_count + 1, total_agents + 1))
    actual_ids = set(managers)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Pre-generated team_config manager IDs must be contiguous from {worker_count + 1} "
            f"to {total_agents}. Found: {sorted(actual_ids)}"
        )

    graph = [[0 for _ in range(total_agents)] for _ in range(total_agents)]
    for manager_id, manager_cfg in managers.items():
        for child_id in manager_cfg["children"]:
            if child_id < 1 or child_id > total_agents:
                raise ValueError(f"Invalid child id {child_id} for manager {manager_id}")
            graph[manager_id - 1][child_id - 1] = 1

    agent_types = [(0, "") for _ in range(worker_count)]
    for manager_id in sorted(managers):
        manager_cfg = managers[manager_id]
        manager_type = manager_cfg["type"]
        manager_type_code = 1 if manager_type == "horizontal" else 2
        agent_types.append((manager_type_code, manager_cfg["team_name"]))

    return graph, agent_types

def load_pre_generated_team_config_graph(cfg, level: str, worker_count: int) -> tuple[list[list[int]], list[tuple[int, str]]]:
    path = _pre_generated_team_config_path(cfg, level)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre-generated team config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    team_config = payload.get("team_config")
    if not team_config:
        raise ValueError(f"Pre-generated config has no team_config field: {path}")

    print(f"[TEAM STRUCTURE] Loading pre-generated team config: {path}")
    graph, agent_types = team_config_to_graph(team_config, worker_count)
    _validate_graph_and_types(graph, agent_types)
    return graph, agent_types

def _get_metric(cfg) -> str:
    if cfg.envs.manager_type == "both":
        prompt = """- Synchronized task decomposition where members must synchronously work through phases (vertical managers) 
    - Or scaling decomposition where there are too many independent agents for one team to coordinate (horizontal managers)"""
    elif cfg.envs.manager_type == "horizontal":
        prompt = """- Or scaling decomposition where there are too many independent agents for one team to coordinate (horizontal managers)"""
    else:
        prompt = """- Synchronized task decomposition where members must synchronously work through phases (vertical managers)"""
    return prompt

def _get_manager_type(cfg) -> str:
    if cfg.envs.manager_type == "both":
        manager_type = '"full" or "horizontal"'
        
    elif cfg.envs.manager_type == "horizontal":
        manager_type = '"horizontal"'

    else:
        manager_type = '"full"'
    
    return manager_type

def _get_manager_description(cfg) -> str:
    if cfg.envs.manager_type == "both":
        prompt = """1. **Full Manager** (Temporal Coordination)
    - Use when THIS manager's DIRECT children need sequential/realtime coordination. such as transporting agents
    - Children work through phases TOGETHER in lockstep
    - Question to ask: "Do my direct children need to wait for each other?"

    2. **Horizontal Manager** (Parallel Coordination)
    - Use when THIS manager's DIRECT children work on INDEPENDENT subtasks
    - Children execute simultaneously without inter-dependencies
    - Question to ask: "Can my direct children work independently?" """
        
    elif cfg.envs.manager_type == "horizontal":
        prompt = """1. **Horizontal Manager** (Parallel Coordination)
    - You can ONLY use horizontal manager for all the managers you want to create.
    - Use when THIS manager's DIRECT children work on INDEPENDENT subtasks
    - Children execute simultaneously without inter-dependencies
    - Question to ask: "Can my direct children work independently?" """

    else:
        prompt = """1. **Full Manager** (Temporal Coordination)
    - You can ONLY use full manager for all the managers you want to create.
    - Use when THIS manager's DIRECT children need sequential/realtime coordination. such as transporting agents
    - Children work through phases TOGETHER in lockstep
    - Question to ask: "Do my direct children need to wait for each other?" """
    
    return prompt

def _get_model_name(cfg, depth: str = "low") -> str:
    envs = getattr(cfg, "envs", cfg)
    explicit_model = getattr(envs, "model_name", None)
    if explicit_model:
        return explicit_model

    model_map = {
        "gpt": {
            "high": "gpt-5.4",
            "low": "gpt-5.4-nano",
        },
        "qwen": {
            "high": "Qwen/Qwen3.6-35B-A3B",
            "low": "Qwen/Qwen3.6-35B-A3B",
        },
        "deepseek": {
            "high": "deepseek-ai/DeepSeek-V4-Pro",
            "low": "deepseek-ai/DeepSeek-V4-Pro",
        },
        "gemma": {
            "high": "google/gemma-4-31B-it",
            "low": "google/gemma-4-31B-it",
        },
        "glm": {
            "high": "zai-org/GLM-5.1-FP8",
            "low": "zai-org/GLM-5.1-FP8",
        },
        "llama": {
            "high": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "low": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        },
        "ernie" :{
            "high": 'baidu/ERNIE-4.5-21B-A3B-PT',
            "low": 'baidu/ERNIE-4.5-21B-A3B-PT',
        },
        "nemotron":{
            "high": 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
            "low": 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
        },
        "minimax":{
            "high": 'MiniMaxAI/MiniMax-M3-MXFP8',
            "low": "MiniMaxAI/MiniMax-M3-MXFP8",
        },
        "kimi":{
            "high": 'moonshotai/Kimi-K2.6',
            "low": 'moonshotai/Kimi-K2.6',   
        }
        
    }
    llm_model = _get_llm_model(cfg)
    return model_map.get(llm_model, model_map["gpt"]).get(depth, model_map["gpt"]["low"])


class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the ORCH algorithm.
    
    Attributes:
        done (bool): Whether the task is complete
        action (int): The type of action to be performed
        x (int): X-coordinate parameter
        y (int): Y-coordinate parameter
        explanation (str): Human-readable description of the action
    """
    type: int
    param_1: int
    param_2: int
    description: str



class Option(BaseModel):
    type: int
    param_1: int
    param_2: int
    description: str
    def print_option(self)->None:
        print(self.type, self.param_1, self.param_2, self.description)

class OptionSequence(BaseModel):
    actions: list[str]
    reasonings: list[str]


class Options(BaseModel):
    actions: list[Option]


class Critique(BaseModel):
    judge: bool
    explanation: str


class TeamStructureCritique(BaseModel):
    approved: bool
    explanation: str
    suggestions: str


def _parse_tag_content(response: str, tag_name: str) -> str:
    """Parse content from a specific tag in the response."""
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"

    start_index = response.find(start_tag)
    end_index = response.find(end_tag)

    if start_index != -1 and end_index != -1 and end_index > start_index:
        content_start = start_index + len(start_tag)
        content = response[content_start:end_index].strip()
        return content
    else:
        return ""

def _parse_option_sequence_from_tags(response: str) -> OptionSequence:
    """Parse OptionSequence from tag-based response."""
    actions_content = _parse_tag_content(response, "actions")
    reasonings_content = _parse_tag_content(response, "reasonings")

    # Parse numbered or bulleted lists
    actions = []
    reasonings = []

    if actions_content:
        # Split by newlines and filter out empty lines
        action_lines = [line.strip() for line in actions_content.split('\n') if line.strip()]
        for line in action_lines:
            # Remove leading numbers like "1.", "2.", etc.
            cleaned = re.sub(r'^\d+\.\s*', '', line).strip()
            if cleaned:
                actions.append(cleaned)

    if reasonings_content:
        reasoning_lines = [line.strip() for line in reasonings_content.split('\n') if line.strip()]
        for line in reasoning_lines:
            cleaned = re.sub(r'^\d+\.\s*', '', line).strip()
            if cleaned:
                reasonings.append(cleaned)

    # Ensure both lists have the same length
    if len(reasonings) < len(actions):
        reasonings.extend(["No reasoning provided"] * (len(actions) - len(reasonings)))
    elif len(actions) < len(reasonings):
        reasonings = reasonings[:len(actions)]

    return OptionSequence(actions=actions, reasonings=reasonings)

def _parse_options_from_tags(response: str) -> Options:
    """Parse Options from tag-based response.

    Handles both formats:
    - Single <options> block with multiple <option> elements inside
    - Multiple <options> blocks each with one <option> element
    """
    options_list = []

    # Find ALL <option>...</option> blocks in the entire response
    # This handles both single and multiple <options> wrapper formats
    option_pattern = r'<option>(.*?)</option>'
    option_matches = re.findall(option_pattern, response, re.DOTALL)

    for option_text in option_matches:
        type_val = _parse_tag_content(option_text, "type")
        param_1_val = _parse_tag_content(option_text, "param_1")
        param_2_val = _parse_tag_content(option_text, "param_2")
        description_val = _parse_tag_content(option_text, "description")

        try:
            option = Option(
                type=int(type_val) if type_val else 0,
                param_1=int(param_1_val) if param_1_val else 0,
                param_2=int(param_2_val) if param_2_val else 0,
                description=description_val if description_val else "No description"
            )
            options_list.append(option)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to parse option: {e}")
            continue

    return Options(actions=options_list)

def _parse_options_from_pipes(response: str) -> Options:
    """Parse Options from pipe-delimited response format.

    Expected format per line: type|param_1|param_2|description
    Uses regex to find the pipe pattern anywhere in each line,
    ignoring stray prefixes like line numbers or agent labels.
    Valid action types: 0-7
    """
    VALID_ACTION_TYPES = set(range(8))
    PIPE_PATTERN = re.compile(r'(\d+)\|([\d.]+)\|([\d.]+)\|(.+)')
    options_list = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = PIPE_PATTERN.search(line)
        if not match:
            continue
        try:
            action_type = int(match.group(1))
            param_1 = int(float(match.group(2)))
            param_2 = int(float(match.group(3)))
            description = match.group(4).strip()
            if action_type not in VALID_ACTION_TYPES:
                print(f"Warning: Invalid action type {action_type}, skipping")
                continue
            option = Option(
                type=action_type,
                param_1=param_1,
                param_2=param_2,
                description=description
            )
            options_list.append(option)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to parse pipe option: {e}")
            continue
    return Options(actions=options_list)

def generate_graph(agent_count: int, level: str, cfg=None, worker_agents=None, api_key=None, log_path=None) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """
    Generate a hierarchical graph for agents with team structure.

    Args:
        agent_count: Total number of agents
        level: Level name to determine graph structure
        cfg: Configuration object (required for llm_generated mode)
        worker_agents: List of worker agent objects (required for llm_generated mode)
        api_key: OpenAI API key (required for llm_generated mode)
        log_path: Path to save logs (optional, for llm_generated mode)

    Returns:
        tuple containing:
        - graph: Adjacency matrix where graph[i][j]=1 means agent i manages agent j
        - agent_types: List of (agent_type, team_name) tuples where:
            - agent_type: 0 = worker, 1 = horizontal manager, 2 = full/vertical manager
            - team_name: Name of the team for managers (e.g., "TEAM_A"), empty string for workers
    """

    if level == "llm_generated":
        # AI-generated team structure mode
        if cfg is None or worker_agents is None:
            raise ValueError("llm_generated mode requires cfg and worker_agents parameters")

        if _use_pre_generated_team_config(cfg):
            task_level = getattr(cfg.envs, "level", None)
            if not task_level:
                raise ValueError("pre-generated llm_generated mode requires cfg.envs.level")
            graph, agent_types = load_pre_generated_team_config_graph(cfg, task_level, agent_count)
            return graph, agent_types

        if api_key is None:
            raise ValueError("llm_generated mode requires api_key unless envs.as_pre_generated is true")

        print("[TEAM STRUCTURE] Generating AI-powered team structure...")

        # Generate mission description from config
        mission_desc = generate_mission_description_from_config(cfg)
        print(f"[TEAM STRUCTURE] Mission: {mission_desc}")

        # Extract worker counts
        worker_counts = extract_worker_counts_from_config(cfg)
        print(f"[TEAM STRUCTURE] Workers: {worker_counts}")

        # Generate structure with LLM
        structure = generate_team_structure_with_llm(
            worker_counts=worker_counts,
            mission_description=mission_desc,
            api_key=api_key,
            log_path=log_path,
            cfg=cfg
        )

        # Convert to graph format
        graph, agent_types = convert_team_structure_to_graph(structure, worker_agents)

        return graph, agent_types

    elif level == "simple":
        graph = [[0 for _ in range(agent_count+1)] for _ in range(agent_count+1)]

        for i in range(agent_count):
            graph[agent_count][i]=1

        # Generate agent types: first agent_count are workers, last is full manager
        agent_types = [(0, "") for _ in range(agent_count)]
        agent_types.append((2, "TEAM_A"))  # Top-level manager

    else:
        # Load hierarchy from experiment_presets.json
        hierarchy = load_preset_hierarchy(level)
        if hierarchy is not None:
            graph, agent_types = hierarchy_to_graph(hierarchy, agent_count)
        else:
            raise ValueError(
                f"Generate_graph: Invalid level: {level}. "
                f"Not found in hardcoded cases or experiment_presets.json"
            )

    # Validate the graph and agent types
    _validate_graph_and_types(graph, agent_types)

    return graph, agent_types


def load_preset_hierarchy(level: str) -> dict | None:
    """Load hierarchy from experiment_presets.json for a given level."""
    presets_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiment_presets.json'
    )
    if not os.path.exists(presets_path):
        return None
    with open(presets_path) as f:
        presets = json.load(f)
    if level in presets and "hierarchy" in presets[level]:
        return presets[level]["hierarchy"]
    return None


def hierarchy_to_graph(hierarchy: dict, worker_count: int) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Convert experiment_presets.json hierarchy to (graph, agent_types) format."""
    config = hierarchy["config"]

    managers = {}
    for agent_name, cfg in config.items():
        agent_id = int(agent_name.split("_")[1])
        children_ids = [int(c.split("_")[1]) for c in cfg["children"]]
        managers[agent_id] = {
            "children": children_ids,
            "type": cfg["type"],
            "team_name": cfg["team_name"],
        }

    total_agents = worker_count + len(managers)
    graph = [[0] * total_agents for _ in range(total_agents)]

    for manager_id, info in managers.items():
        for child_id in info["children"]:
            graph[manager_id - 1][child_id - 1] = 1

    # Workers first, then managers sorted by ID
    agent_types = [(0, "") for _ in range(worker_count)]
    for manager_id in sorted(managers.keys()):
        m = managers[manager_id]
        type_code = 1 if m["type"] == "horizontal" else 2
        agent_types.append((type_code, m["team_name"]))

    return graph, agent_types

def _validate_graph_and_types(graph: list[list[int]], agent_types: list[tuple[int, str]]) -> None:
    """
    Validate that the graph adjacency matrix matches the agent types.

    Args:
        graph: Adjacency matrix where graph[i][j]=1 means agent i manages agent j
        agent_types: List of (agent_type, team_name) tuples

    Raises:
        ValueError: If validation fails
    """
    n = len(graph)

    # Check dimensions match
    if len(agent_types) != n:
        raise ValueError(f"Graph has {n} agents but agent_types has {len(agent_types)} entries")

    # Check square matrix
    for i, row in enumerate(graph):
        if len(row) != n:
            raise ValueError(f"Graph row {i} has {len(row)} entries, expected {n}")

    # Validate each agent
    for agent_idx in range(n):
        agent_type, team_name = agent_types[agent_idx]

        # Count children (how many agents this agent manages)
        children_count = sum(graph[agent_idx])

        # Validate based on agent type
        if agent_type == 0:  # Worker
            # Workers should not manage anyone
            if children_count > 0:
                raise ValueError(f"Agent {agent_idx} is a worker but manages {children_count} agents in graph")
            # Workers should not have team names
            if team_name != "":
                raise ValueError(f"Agent {agent_idx} is a worker but has team_name '{team_name}'")

        elif agent_type in [1, 2]:  # Manager (horizontal or full)
            # Managers should manage at least one agent
            if children_count == 0:
                raise ValueError(f"Agent {agent_idx} is a manager but manages 0 agents in graph")
            # Managers should have team names
            if team_name == "":
                raise ValueError(f"Agent {agent_idx} is a manager but has no team_name")

        else:
            raise ValueError(f"Agent {agent_idx} has invalid agent_type {agent_type}, must be 0, 1, or 2")

    # Check for cycles (DAG validation)
    def has_cycle(node, visited, rec_stack):
        visited[node] = True
        rec_stack[node] = True

        for child in range(n):
            if graph[node][child] == 1:
                if not visited[child]:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif rec_stack[child]:
                    return True

        rec_stack[node] = False
        return False

    visited = [False] * n
    rec_stack = [False] * n
    for node in range(n):
        if not visited[node]:
            if has_cycle(node, visited, rec_stack):
                raise ValueError("Graph contains a cycle - hierarchies must be acyclic (DAG)")

    print(f"[VALIDATION] Graph validated successfully: {n} agents")
    for i, (agent_type, team_name) in enumerate(agent_types):
        type_str = ["Worker", "Horizontal Manager", "Full Manager"][agent_type]
        children = [j for j in range(n) if graph[i][j] == 1]
        if agent_type == 0:
            print(f"  Agent {i}: {type_str}")
        else:
            print(f"  Agent {i}: {type_str} ({team_name}) managing agents {children}")

def request_options(agent, global_data: dict) -> OptionSequence:
    # DEPRECATED: USE ASYNC METHOD
    pass

async def async_request_options(agent, global_data: dict) -> OptionSequence:
    """
    Async version of request_options - Generate action options for an agent based on their current task and observations.

    Args:
        agent: The agent object containing task, observations, and configuration
        global_data: Global data containing API tracking information

    Returns:
        OptionSequence: A sequence of action descriptions and reasoning
    """

    # Load manager prompt for option generation
    agent_type_string = "Unknown"
    if hasattr(agent, 'type'):
        type_map = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter", -1: "manager"}
        agent_type_string = type_map.get(agent.type, "Unknown")


    prompt_path = os.path.join("algorithms", "ORCH", "prompts", "planner", f"{agent_type_string}_planner.txt")
    
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "planner", f"{agent_type_string}_planner.txt")
    
    with open(prompt_path, 'r') as file:
        prompt_content = file.read()

    # Replace placeholders in prompt
    prompt_content = prompt_content.replace("MAPSIZE-1", str(agent.cfg.envs.map_size-1))
    prompt_content = prompt_content.replace("MAPSIZE", str(agent.cfg.envs.map_size))
    prompt_content = prompt_content.replace("POSITION", str(getattr(agent, 'last_position', (0, 0))))
    prompt_content = prompt_content.replace("MAPRANGE", f"({getattr(agent, 'last_position', (0, 0))[0]-getattr(agent, 'map_range', 5)} to {getattr(agent, 'last_position', (0, 0))[0]+getattr(agent, 'map_range', 5)}, {getattr(agent, 'last_position', (0, 0))[1]-getattr(agent, 'map_range', 5)} to {getattr(agent, 'last_position', (0, 0))[1]+getattr(agent, 'map_range', 5)})")
    prompt_content = prompt_content.replace("OBS", getattr(agent, 'perception_summary', "No observations available"))
    prompt_content = prompt_content.replace("CURRCELL", getattr(agent, 'last_current_cell', "unknown"))
    prompt_content = prompt_content.replace("TASK", getattr(agent, 'mission', "No task assigned"))
    prompt_content = prompt_content.replace("OBSRANGE", str(getattr(agent, 'map_range', 5)))

    # Replace status placeholders based on agent type
    if hasattr(agent, 'type') and hasattr(agent, 'extra_variables') and agent.extra_variables:
        if agent.type == 0:  # Firefighter
            carrying = "Yes" if agent.extra_variables[0] == 1 else "No"
            water = int(agent.extra_variables[1]) if len(agent.extra_variables) > 1 else 0
            prompt_content = prompt_content.replace("{CARRYING_CIVILIAN}", carrying)
            prompt_content = prompt_content.replace("{WATER_LEVEL}", str(water))
        elif agent.type == 3:  # Helicopter
            carrying = int(agent.extra_variables[0]) if agent.extra_variables else 0
            water = int(agent.extra_variables[1]) if len(agent.extra_variables) > 1 else 0
            prompt_content = prompt_content.replace("{CARRYING_FIREFIGHTERS}", str(carrying))
            prompt_content = prompt_content.replace("{WATER_LEVEL}", str(water))

    # Append human feedback if present
    if getattr(agent, 'past_feedback', []):
        prompt_content += f"\n\nHuman Operator Feedback to incorporate:\n{agent._format_feedback()}\n"

    # Append game event announcements if present
    announcement_block = agent._build_announcement_block(global_data)
    if announcement_block:
        prompt_content += f"\n\n{announcement_block}"

    # Call AsyncOpenAI to generate options
    cfg = getattr(agent, "cfg", None)
    if _get_llm_model(cfg) == "gpt":
        client = AsyncOpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
    else:
        client = AsyncOpenAI(
            api_key=agent.api_key,
            base_url=_get_llm_url(cfg),
            timeout=3600,
        )

    option_sequence = None
    max_retries = 3
    retry_count = 0

    system_message = """You are a highly trained agent within a grid forest world. Your job is to break down a task into smaller actions to be performed by the agent.

Respond using the following tag format:

<actions>
1. First action description
2. Second action description
</actions>

Keep each action description to one short sentence. No elaboration needed."""

    while not option_sequence and retry_count < max_retries:
        try:
            if _get_llm_model(cfg) == "gpt":
                response = await client.chat.completions.create(
                    model=_get_model_name(cfg, "high"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt_content}
                    ],
                )
            else:
                response = await client.chat.completions.create(
                    model=_get_model_name(cfg, "high"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )

            response_content = response.choices[0].message.content
            option_sequence = _parse_option_sequence_from_tags(response_content)

            # Validate parsed result
            if not option_sequence.actions:
                raise ValueError("No actions parsed from response")

            agent.log_chat("Generate Options", [("user", prompt_content), ("assistant", response_content)])
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens

        except Exception as e:
            print(f"Error generating options for agent {agent.id}, retry {retry_count + 1}: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                # Create a default option sequence
                option_sequence = OptionSequence(
                    actions=["Move to current position and wait for instructions"],
                    reasonings=["Default action due to generation failure"]
                )

    await client.close()
    return option_sequence

def translate_options(agent, option_sequence: OptionSequence, global_data: dict) -> Options:
    # DEPRECATED: USE ASYNC METHOD
    pass

async def async_translate_options(agent, option_sequence: OptionSequence, global_data: dict) -> Options:
    """
    Async version of translate_options - Translate option sequence into structured Option objects.

    Args:
        agent: The agent object
        option_sequence: The raw option sequence from request_options
        global_data: Global data containing API tracking information

    Returns:
        Options: Structured options for the agent
    """

    # Determine agent type string
    if agent.type == 0:
        type_string = 'firefighter'
    elif agent.type == 1:
        type_string = 'bulldozer'
    elif agent.type == 2:
        type_string = 'drone'
    else:
        type_string = 'helicopter'

    system_message = f"""You are the controller of a highly trained {type_string} agent within a grid forest world.
Your job is to convert text actions into a structured format for robotic control.

For each input action, output EXACTLY one line in this exact format:
type|param_1|param_2|description

Every line MUST have exactly 4 pipe-separated fields. The first field is always the action type number.
The number of output lines MUST equal the number of input actions. Never split one input action into multiple lines, even if it mentions multiple entities.
Coordinates MUST be whole integers with no decimal points.

Do NOT add line numbers. Output ONLY the pipe-delimited lines, one per input action, in order."""

    # Load translator prompt
    prompt_path = os.path.join("algorithms", "ORCH", "prompts", "translator", f"{type_string}_translator.txt")
    
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "translator", f"{type_string}_translator.txt")
    
    optionstring = "\n".join(option_sequence.actions)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().replace("ACTIONS", str(optionstring))
    except FileNotFoundError:
        print(f"Warning: Prompt file not found for {type_string}, using default")
        prompt = ""

    # Call AsyncOpenAI
    cfg = getattr(agent, "cfg", None)
    if _get_llm_model(cfg) == "gpt":
        client = AsyncOpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
        response = await client.chat.completions.create(
            model=_get_model_name(cfg, "low"),
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
        )
    else:
        client = AsyncOpenAI(
            api_key=agent.api_key,
            base_url=_get_llm_url(cfg),
            timeout=3600,
        )
        response = await client.chat.completions.create(
            model=_get_model_name(cfg, "low"),
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    response_content = response.choices[0].message.content
    agent.log_chat("Translate Options", [("user", prompt), ("assistant", response_content)])

    # Try pipe format first, fall back to XML tags
    options = _parse_options_from_pipes(response_content)
    if not options.actions:
        options = _parse_options_from_tags(response_content)

    # Filter spurious do-nothing actions that appear alongside real actions
    if len(options.actions) > 1:
        filtered = [opt for opt in options.actions if opt.type != 0]
        if filtered:
            options = Options(actions=filtered)

    # Fallback: if no options parsed, create default
    if not options.actions:
        print(f"Warning: No options parsed for agent {agent.id}, creating default")
        options = Options(actions=[Option(type=0, param_1=0, param_2=0, description="Wait for instructions")])

    await client.close()
    return options

# Global variables for human interface API
human_actions: Dict[str, List[int]] = {}
human_observations: Dict[str, Dict[str, Any]] = {}
session_active = False
session_lock = threading.Lock()


def submit_human_actions_batch(actions: Dict[str, List[int]]) -> bool:
    """Submit batch of human actions"""
    global human_actions
    with session_lock:
        if session_active:
            human_actions.update(actions)
            return True
        return False

def submit_human_action(agent_role: str, action: List[int]) -> bool:
    """Submit action from human player"""
    global human_actions
    with session_lock:
        if session_active:
            human_actions[agent_role] = action
            return True
        return False

def get_human_observations(agent_role: str) -> Dict[str, Any]:
    """Get latest observations for human player"""
    with session_lock:
        if agent_role in human_observations:
            return human_observations[agent_role]
        return {}

def get_all_human_observations() -> Dict[str, Dict[str, Any]]:
    """Get all human observations"""
    with session_lock:
        return human_observations.copy()

def is_session_active() -> bool:
    """Check if session is active"""
    with session_lock:
        return session_active

def set_session_active(active: bool):
    """Set session active status"""
    global session_active
    with session_lock:
        session_active = active

def update_human_observations(agent_role: str, observations: Dict[str, Any]):
    """Update observations for a human agent"""
    global human_observations
    with session_lock:
        human_observations[agent_role] = observations

def clear_human_actions():
    """Clear all human actions"""
    global human_actions
    with session_lock:
        human_actions.clear()

def get_all_human_actions() -> Dict[str, List[int]]:
    """Get all human actions"""
    with session_lock:
        return human_actions.copy()


# ============================================================================
# AI Team Structure Generation
# ============================================================================

def generate_mission_description_from_config(cfg) -> str:
    """
    Generate a detailed mission description from configuration before game starts.

    Args:
        cfg: Configuration object with level presets

    Returns:
        str: Detailed mission description string
    """
    game_type = getattr(cfg.envs, 'game_type', None)
    map_size = getattr(cfg.envs, 'map_size', 100)

    if game_type is None:
        return "Generic wildfire response mission"

    mission_parts = []

    # Game type 0: Cut Trees
    if game_type == 0:
        lines = getattr(cfg.envs, 'lines', False)
        tree_count = getattr(cfg.envs, 'tree_count', 0)
        trees_per_line = getattr(cfg.envs, 'trees_per_line', 1)

        if lines:
            mission_parts.append(f"Cut trees along {tree_count} linear formations, with approximately {trees_per_line} trees per line")
        else:
            mission_parts.append(f"Cut {tree_count} trees scattered across the map")

    # Game type 1: Scout Fire
    elif game_type == 1:
        known = getattr(cfg.envs, 'known', False)
        if known:
            mission_parts.append("Scout and confirm the fire at a known general location (requires two agents directly over fire)")
        else:
            mission_parts.append("Search the entire map to locate and confirm fire (requires two agents directly over fire)")

    # Game type 2: Transport Firefighters
    elif game_type == 2:
        known = getattr(cfg.envs, 'known', False)
        if known:
            mission_parts.append("Transport all firefighter agents to a known deployment location")
        else:
            mission_parts.append("Transport all firefighter agents to deployment zone (location to be determined)")

    # Game type 3: Suppress/Contain Fire
    elif game_type == 3:
        known = getattr(cfg.envs, 'known', False)
        water = getattr(cfg.envs, 'water', False)

        if known and water:
            mission_parts.append("Suppress fire at known location using available water source for refilling")
        elif known and not water:
            mission_parts.append("Contain fire at known location by cutting firebreaks (NO water source available)")
        elif not known and water:
            mission_parts.append("Locate fire across the map, then suppress it using available water source for refilling")
        else:  # not known and not water
            mission_parts.append("Locate fire across the map, then contain it by cutting firebreaks (NO water source available)")

    # Game type 4: Rescue Civilians
    elif game_type == 4:
        known = getattr(cfg.envs, 'known', False)
        civilian_clusters = getattr(cfg.envs, 'civilian_clusters', 1)
        civilian_count = getattr(cfg.envs, 'civilian_count', 3)

        if known:
            mission_parts.append(f"Rescue {civilian_clusters} groups of civilians ({civilian_count} civilians per group) from known locations and transport them to safety zone")
        else:
            mission_parts.append(f"Search map for {civilian_clusters} civilian groups ({civilian_count} per group), rescue them, and transport to safety zone")

    # Game type 5: Combined Fire + Civilians
    elif game_type == 5:
        known = getattr(cfg.envs, 'known', False)
        water = getattr(cfg.envs, 'water', False)
        civilian_clusters = getattr(cfg.envs, 'civilian_clusters', 1)
        civilian_count = getattr(cfg.envs, 'civilian_count', 3)

        # Fire objective
        if known and water:
            mission_parts.append("Suppress fire at known location using available water source")
        elif known and not water:
            mission_parts.append("Contain fire at known location by cutting firebreaks (NO water available)")
        elif not known and water:
            mission_parts.append("Locate and suppress fire using available water source")
        else:
            mission_parts.append("Locate and contain fire by cutting firebreaks (NO water available)")

        # Civilian objective
        if known:
            mission_parts.append(f"Simultaneously rescue {civilian_clusters} groups of civilians ({civilian_count} per group) from known locations")
        else:
            mission_parts.append(f"Simultaneously search for and rescue {civilian_clusters} civilian groups ({civilian_count} per group)")

    # Add map context
    mission_parts.append(f"Map size: {map_size}x{map_size} grid")

    return ". ".join(mission_parts) + "."

def extract_worker_counts_from_config(cfg) -> Dict[str, int]:
    """
    Extract worker agent counts from configuration.

    Args:
        cfg: Configuration object

    Returns:
        dict: Worker counts by type {firefighter: N, bulldozer: M, drone: K, helicopter: J}
    """
    return {
        'firefighter': getattr(cfg.envs, 'starting_firefighter_agents', 0),
        'bulldozer': getattr(cfg.envs, 'starting_bulldozer_agents', 0),
        'drone': getattr(cfg.envs, 'starting_drone_agents', 0),
        'helicopter': getattr(cfg.envs, 'starting_helicopter_agents', 0)
    }

def critique_team_structure(
    structure: Dict[str, Any],
    worker_counts: Dict[str, int],
    mission_description: str,
    api_key: str,
    cfg = None
) -> tuple[TeamStructureCritique, Dict[str, Any]]:
    """
    Use LLM critic to evaluate the quality and appropriateness of a team structure.
    This includes validating agent counts as well as overall structure quality.

    Args:
        structure: The team structure to critique
        worker_counts: Dictionary of worker counts by type
        mission_description: Detailed mission description
        api_key: OpenAI API key

    Returns:
        tuple: (TeamStructureCritique, dict) where dict contains:
            - 'prompt': The critique prompt sent to LLM
            - 'response': The LLM response content
            - 'system_message': The system message
    """
    import json
    print("critic evaluating...")

    # First, validate agent counts
    def validate_agent_counts(structure, expected_counts):
        """
        Validate that the structure uses exactly the expected number of each agent type.
        Returns (is_valid, error_message)
        """
        actual_counts = {'firefighter': 0, 'bulldozer': 0, 'drone': 0, 'helicopter': 0}

        def count_workers(node):
            if 'root_manager' in node:
                node = node['root_manager']

            if 'children' in node:
                for child in node['children']:
                    if child.get('type') == 'worker':
                        worker_type = child.get('worker_type')
                        count = child.get('count', 1)
                        if worker_type in actual_counts:
                            actual_counts[worker_type] += count
                    elif child.get('type') == 'manager':
                        count_workers(child)

        count_workers(structure)

        # Check for discrepancies
        errors = []
        for agent_type, expected in expected_counts.items():
            actual = actual_counts.get(agent_type, 0)
            if actual != expected:
                if actual > expected:
                    errors.append(f"TOO MANY {agent_type}s: assigned {actual}, only have {expected}")
                else:
                    errors.append(f"NOT ENOUGH {agent_type}s: assigned {actual}, need to assign all {expected}")

        if errors:
            return False, "\n".join(errors)
        return True, ""

    # Build worker summary
    worker_summary = []
    for agent_type, count in worker_counts.items():
        if count > 0:
            worker_summary.append(f"{count} {agent_type}{'s' if count > 1 else ''}")
    worker_summary_str = ", ".join(worker_summary)

    # Format the structure for presentation
    structure_json = json.dumps(structure, indent=2)

    # Check agent counts first
    is_valid, error_msg = validate_agent_counts(structure, worker_counts)

    # Build critique prompt based on validation result
    if not is_valid:
        # If agent counts are invalid, include this in the critique prompt
        critique_prompt = f"""You are an expert critic evaluating team structures for multi-agent wildfire response missions.

        **MISSION**: {mission_description}

        **AVAILABLE AGENTS**: {worker_summary_str}
        - Firefighters: {worker_counts.get('firefighter', 0)}
        - Bulldozers: {worker_counts.get('bulldozer', 0)}
        - Drones: {worker_counts.get('drone', 0)}
        - Helicopters: {worker_counts.get('helicopter', 0)}

        **PROPOSED TEAM STRUCTURE**:
        ```json
        {structure_json}
        ```

        **CRITICAL ERROR - INVALID AGENT COUNTS**:
        {error_msg}

        The proposed structure has INCORRECT agent counts. This is a CRITICAL ERROR that must be rejected.

        **YOUR EVALUATION**:
        - Set `approved: false` because the agent counts are invalid
        - In your `explanation`, clearly state that the agent counts are wrong and specify the errors
        - In your `suggestions`, provide specific guidance on how to fix the agent count issues

        You MUST reject this structure due to invalid agent counts."""

    else:
        # If agent counts are valid, do normal quality critique
        critique_prompt = f"""
        You are an expert critic evaluating team structures for multi-agent missions.

        **MISSION**: {mission_description}
        IMPORTANT: Only expect what the mission demands. Do not expect possible situations that are not mentioned; 
        the mission is fully comprehenseive of the task to complete.


        **AVAILABLE AGENTS**: {worker_summary_str}
        [Agent counts have been validated and are correct.]

        **PROPOSED TEAM STRUCTURE**:
        ```json
        {structure_json}
        ```


        **AGENT CAPABILITIES**:
        - **Firefighter**: Can cut trees (slow), pick up/drop off civilians, spray water to extinguish fires, refill water
        - **Bulldozer**: Can cut trees (fast), cannot interact with fire or civilians
        - **Drone**: Can scout/observe areas, cannot manipulate environment
        - **Helicopter**: Can transport (up to 5) firefighters to different locations, can refill and deploy water on fire.

        ---

        **MANAGER TYPES:**

        {_get_manager_description(cfg)}

        ---

        Your task is to critically evaluate this team structure specifically for these points:

        **EVALUATION CRITERIA**:

        1. The structure is minimal{": There are no unnecessary managers and all full managers cannot be replaced by horizontal managers." if cfg.envs.manager_type == "both" else "."}
        2. There are no managers with only 1 workers/subteams.
        3. All agents of the same type and in the same direct team are grouped together as count:N
        4. Worker agents can plan multiple steps/actions independently, so they should not be micromanaged too much.
        5. The plan isn't build around possiblies outside of what is explicitly stated in the mission.
        6. The rule of thumb is being followed:
        
        **RULE OF THUMB**
            Rule of thumb: 
            - Start with the simplest team possible with just the root manager
            - Then add subteams and complexity from there. 
            - Only make subteams when there is a critical need for: 
                {_get_metric(cfg)}
    
        Return an overall consensus, a short explanation one sentence for the decision, and a single short suggestion for improvement.
    
        """

    system_message = "You are an expert critic of multi-agent team coordination structures. You provide honest, constructive feedback on team hierarchies."

    try:
        if _get_llm_model(cfg) == "gpt":
            client = OpenAI(
                api_key=api_key,
                base_url=_get_llm_url(cfg),
                timeout=3600,
            )
            reasoning_effort = _get_reasoning_effort(cfg)
            request_options = ({"reasoning_effort": reasoning_effort} if reasoning_effort else {"temperature": cfg.llms.structure_generator_temperature if cfg else 0.0})
            response = client.chat.completions.create(
                model=_get_model_name(cfg, "high"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": critique_prompt}
                ],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        "name": "team_structure_critique",
                        "schema": TeamStructureCritique.model_json_schema()
                    }
                },
                **request_options,
            )
        
        else:
            client = OpenAI(
                api_key=api_key,
                base_url=_get_llm_url(cfg),
                timeout=3600,
            )
            response = client.chat.completions.create(
                model=_get_model_name(cfg, "high"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": critique_prompt}
                ],
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        "name": "team_structure_critique",
                        "schema": TeamStructureCritique.model_json_schema()
                    }
                },
                temperature=cfg.llms.structure_generator_temperature if cfg else 0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            ) 
            
        client.close()

        critique = TeamStructureCritique.parse_raw(response.choices[0].message.content)

        # Return critique along with logging data
        log_data = {
            'prompt': critique_prompt,
            'response': response.choices[0].message.content,
            'system_message': system_message
        }

        return critique, log_data

    except Exception as e:
        print(f"[ERROR] Failed to get critique: {e}")
        # In case of error, default to approval to not block the process
        error_critique = TeamStructureCritique(
            approved=True,
            explanation="Critique failed due to error, defaulting to approval",
            suggestions=""
        )
        error_log_data = {
            'prompt': critique_prompt if 'critique_prompt' in locals() else "Error: prompt not created",
            'response': f"ERROR: {str(e)}",
            'system_message': system_message if 'system_message' in locals() else "Error: system message not created"
        }
        return error_critique, error_log_data

def generate_team_structure_with_llm(
    worker_counts: Dict[str, int],
    mission_description: str,
    api_key: str,
    log_path: str = None,
    cfg = None
) -> Dict[str, Any]:
    """
    Use LLM to generate an optimal team structure for the mission.

    Args:
        worker_counts: Dictionary of worker counts by type
        mission_description: Detailed mission description
        api_key: OpenAI API key
        log_path: Optional path to save LLM interaction logs

    Returns:
        dict: Team structure in hierarchical JSON format
    """
    import json
    from datetime import datetime

    # Build worker summary
    worker_summary = []
    total_workers = 0
    for agent_type, count in worker_counts.items():
        if count > 0:
            worker_summary.append(f"{count} {agent_type}{'s' if count > 1 else ''}")
            total_workers += count

    worker_summary_str = ", ".join(worker_summary)

    prompt = f"""You are an expert in multi-agent team coordination for wildfire response/search and rescue/scouting/firebreak creation missions. Design an optimal team hierarchy for the following scenario:

    **MISSION**: {mission_description}

    IMPORTANT: Only expect what the mission demands. Do not expect possible situations that are not mentioned; the mission is fully comprehenseive of the task to complete.

    **AVAILABLE AGENTS**: {worker_summary_str} (Total: {total_workers} workers)

    **YOU MUST USE EXACTLY THESE AGENTS**:
    - Firefighters: {worker_counts.get('firefighter', 0)}
    - Bulldozers: {worker_counts.get('bulldozer', 0)}
    - Drones: {worker_counts.get('drone', 0)}
    - Helicopters: {worker_counts.get('helicopter', 0)}

    **AGENT CAPABILITIES**:
    - **Firefighter**: Can cut trees (slow), pick up/drop off civilians, spray water to extinguish fires, refill water
        - NOTE: cannot supply water to eachother, so do not make separate water refill and fire suppression teams
    - **Bulldozer**: Can cut trees (fast), cannot interact with fire or civilians
    - **Drone**: Can scout/observe areas, cannot manipulate environment
    - **Helicopter**: Can transport (up to 5) firefighters to different locations, can refill and deploy water on fire.

    ---

    **MANAGER TYPES - CHOOSE CAREFULLY**:
    
    {_get_manager_description(cfg)}

    ---

    **TEAM NAMES**:
    - Should be **descriptive and role-specific**: Example: "FIRE_SUPPRESS_ALPHA", "RECON TEAM", "HYBRID TEAM", "GROUND TEAM"
    - Communicates the team's **PURPOSE and SCOPE**
    - Helps team members understand their mission context immediately

    ---


    **Mixed Children**: Managers can have BOTH worker children AND sub-manager children
    - Example: Coordination manager with direct scout drones + two firefighting sub-teams

    ---

    **CONSTRAINTS AND NOTES**:
    - Exactly ONE root manager (required)
    - Every manager must have at least TWO children
    - Use ALL workers exactly (no more, no less)
    - No cycles in hierarchy
    - Manager can have workers, sub-managers, or both as children
    - Managers with only one child are redundant
    - Worker agents already come with some internal planning
    - You can have heterogenous teams with mixed agent types
    - Keep the structure as minimal and simple as possible.

    ---

    **OUTPUT FORMAT EXAMPLE** (JSON only):

    {{
    "explanation": "Brief explanation of the strategic reasoning behind this team structure, including why you chose full vs horizontal managers and how the hierarchy supports the mission",
    "root_manager": {{
        "manager_type": {_get_manager_type(cfg)},
        "team_name": "DESCRIPTIVE_NAME",
        "children": [
        {{
            "type": "manager",
            "manager_type": {_get_manager_type(cfg)},
            "team_name": "DESCRIPTIVE_NAME",
            "children": [
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            ]
        }},
        {{
            "type": "manager",
            "manager_type": {_get_manager_type(cfg)},
            "team_name": "DESCRIPTIVE_NAME",
            "children": [
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}},
            ]
        }},
            {{"type": "worker", "worker_type": "firefighter" or "bulldozer" or "drone" or "helicopter", "count": number of agents of this type(must be integer)}}
        ]
    }}
    }}

    **IMPORTANT**:
    - Think strategically about the mission's coordination needs
    - Choose manager types based on whether direct children need sequential + parallel or just parallel coordination
    - Create meaningful team names that communicate purpose
    - Ensure all {total_workers} workers are assigned
    - Include a clear explanation of your design choices
    - Managers with only one worker are redunant. Workers also can handle some planning and thinking on its own.
    - count MUST be an INTEGER only, no decimals, or text, just a number representing how many agents of that type are in that team.
    - count MUST be >= 1
    - count MUST NEVER be:
        - a float (e.g. 1.5)
        - a string (e.g. "3")
        - null
        - text descriptions (e.g. "all remaining firefighters")

    **RULE OF THUMB**
    Keep teams simple when possible. 
    Rule of thumb: 
    - Start with the simplest team possible with just the root manager
    - Then add subteams and complexity from there. 
    - Only make subteams when there is a critical need for: 
        {_get_metric(cfg)}
    - Think less about dividing teams up by type or function, but rather by what groups are nessesary to work together to achieve subgoals.
    - Keep teams as simple as possible

    Respond with ONLY the JSON structure (including explanation field), no additional text."""

    
    def extract_json_text(text: str) -> str:
        """Extract a JSON object from common LLM wrapper formats."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        return text[start:end + 1]

    # Iterative generation with critic-based validation
    max_attempts = 5
    conversation = [
        {"role": "system", "content": "You are an expert multi-agent coordination strategist. You design optimal team hierarchies for complex missions. Always respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    structure = None
    all_attempts = []  # Track all attempts for logging

    try:
        for attempt in range(max_attempts):
            print(f"[TEAM STRUCTURE] Attempt {attempt + 1}/{max_attempts}")

            if _get_llm_model(cfg) == "gpt":
                client = OpenAI(
                    api_key=api_key,
                    base_url=_get_llm_url(cfg),
                    timeout=3600,
                )
                temp = cfg.llms.structure_generator_temperature if cfg else 0.0
                reasoning_effort = _get_reasoning_effort(cfg)
                request_options = ({"reasoning_effort": reasoning_effort} if reasoning_effort else {"temperature": temp})
                response = client.chat.completions.create(
                    model=_get_model_name(cfg, "low"),
                    messages=conversation,
                    **request_options,
                )
            
            else:
                print(_get_llm_url(cfg))
                client = OpenAI(
                    api_key=api_key,
                    base_url=_get_llm_url(cfg),
                    timeout=3600,
                )
                # Use configurable temperature, default to 0 if cfg not provided
                temp = cfg.llms.structure_generator_temperature if cfg else 0.0
                response = client.chat.completions.create(
                    model=_get_model_name(cfg, "low"),
                    messages=conversation,
                    temperature=temp,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
            client.close()

            raw_response_text = response.choices[0].message.content.strip()
            
            #get rid of everything before </think> tags if they exist
            raw_response_text = raw_response_text.split("</think>", 1)[-1].strip()

            try:
                response_text = extract_json_text(raw_response_text)
                structure = json.loads(response_text)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ERROR] Failed to parse LLM response as JSON on attempt {attempt + 1}: {e}")
                print(f"[ERROR] Response was: {raw_response_text}")
                all_attempts.append({
                    "attempt": attempt + 1,
                    "structure": None,
                    "response_text": raw_response_text,
                    "parse_error": str(e),
                })

                if attempt < max_attempts - 1:
                    conversation.append({"role": "assistant", "content": raw_response_text})
                    conversation.append({
                        "role": "user",
                        "content": f"""Your previous generated team structure was not valid JSON and could not be parsed.

Previous generated team structure:
```text
{raw_response_text}
```

Parser error:
{e}

Regenerate the same intended team structure as ONE valid JSON object only.
Do not include markdown fences, prose, comments, trailing commas, null values, or strings for count.
Every worker count must be an integer literal, for example 1, 2, 5, or 10.
Respond with ONLY the corrected JSON object."""
                    })
                    continue

                raise ValueError("LLM did not return valid JSON team structure")
            conversation.append({"role": "assistant", "content": response_text})

            # Log this attempt
            all_attempts.append({
                "attempt": attempt + 1,
                "structure": structure,
                "response_text": response_text
            })

            # Show the generated structure explanation
            if 'explanation' in structure:
                print(f"[TEAM STRUCTURE] Generator explanation: {structure['explanation']}")

            # Check if critic is enabled
            use_critic = cfg.llms.use_structure_critic if cfg else True

            if use_critic:
                # Send to critic for evaluation (includes agent count validation)
                print("[TEAM STRUCTURE] Sending to critic for evaluation...")
                critique, critique_log_data = critique_team_structure(
                    structure=structure,
                    worker_counts=worker_counts,
                    mission_description=mission_description,
                    api_key=api_key,
                    cfg=cfg
                )

                # Log the critique (including prompt and response)
                all_attempts[-1]['critique'] = {
                    'approved': critique.approved,
                    'explanation': critique.explanation,
                    'suggestions': critique.suggestions,
                    'prompt': critique_log_data['prompt'],
                    'response': critique_log_data['response'],
                    'system_message': critique_log_data['system_message']
                }

                if critique.approved:
                    print("[TEAM STRUCTURE] ✓ Critic approved the structure!")
                    print(f"  Critique: {critique.explanation}")
                    break
                else:
                    print(f"[TEAM STRUCTURE] ✗ Critic rejected the structure")
                    print(f"  Reason: {critique.explanation}")
                    if critique.suggestions:
                        print(f"  Suggestions: {critique.suggestions}")

                    if attempt < max_attempts - 1:
                        # Provide critic feedback for next attempt
                        feedback = f"""Your team structure was REJECTED by the critic for the following reasons:

                        **CRITIQUE**: {critique.explanation}

                        **SUGGESTIONS FOR IMPROVEMENT**: {critique.suggestions}

                        Please revise your team structure to address these concerns. Remember:
                        - You have EXACTLY these agents: Firefighters: {worker_counts.get('firefighter', 0)}, Bulldozers: {worker_counts.get('bulldozer', 0)}, Drones: {worker_counts.get('drone', 0)}, Helicopters: {worker_counts.get('helicopter', 0)}
                        - Focus on mission alignment and effectiveness
                        - Use appropriate manager types (full vs horizontal)
                        - Keep the structure efficient (avoid redundancy)

                        Respond with an improved JSON structure using the same format."""

                        conversation.append({"role": "user", "content": feedback})
                    else:
                        print(f"[TEAM STRUCTURE] Max attempts reached. Using last structure despite critic rejection.")
            else:
                # Critic disabled - accept the first generated structure
                print("[TEAM STRUCTURE] Critic disabled - accepting generated structure")
                all_attempts[-1]['critique'] = {
                    'approved': True,
                    'explanation': 'Critic disabled',
                    'suggestions': ''
                }
                break

        if structure is None:
            raise ValueError("Failed to generate any structure")

        print("[TEAM STRUCTURE] Final team structure:")
        print(json.dumps(structure, indent=2))

        # Log the LLM interaction if log_path is provided
        if log_path:
            try:
                if os.path.isdir(log_path) or not os.path.splitext(log_path)[1]:
                    os.makedirs(log_path, exist_ok=True)
                    log_file = os.path.join(log_path, "team_structure_generation.txt")
                else:
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    log_file = log_path
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("TEAM STRUCTURE GENERATION LOG\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

                    f.write("MISSION: " + mission_description + "\n")
                    f.write(f"Workers: {worker_summary_str}\n\n")

                    f.write("-" * 80 + "\n")
                    f.write("INITIAL PROMPT SENT TO LLM:\n")
                    f.write("-" * 80 + "\n")
                    f.write(prompt + "\n\n")

                    # Log all attempts
                    if len(all_attempts) > 1:
                        f.write("=" * 80 + "\n")
                        f.write(f"GENERATION ATTEMPTS (Total: {len(all_attempts)})\n")
                        f.write("=" * 80 + "\n\n")

                        for attempt_data in all_attempts:
                            f.write(f"--- ATTEMPT {attempt_data['attempt']} ---\n\n")
                            f.write("Structure:\n")
                            f.write(json.dumps(attempt_data['structure'], indent=2) + "\n\n")

                            # Check if critique exists for this attempt
                            if 'critique' in attempt_data:
                                critique_data = attempt_data['critique']
                                f.write("-" * 40 + "\n")
                                f.write("CRITIC EVALUATION\n")
                                f.write("-" * 40 + "\n")
                                if critique_data['approved']:
                                    f.write("Status: ✓ APPROVED\n")
                                else:
                                    f.write("Status: ✗ REJECTED\n")
                                f.write(f"Explanation: {critique_data['explanation']}\n")
                                if critique_data['suggestions']:
                                    f.write(f"Suggestions: {critique_data['suggestions']}\n")
                                f.write("\n")

                                # Log the full critic conversation (system message, prompt, response)
                                if 'system_message' in critique_data and 'prompt' in critique_data and 'response' in critique_data:
                                    f.write("-" * 40 + "\n")
                                    f.write("CRITIC CONVERSATION\n")
                                    f.write("-" * 40 + "\n")
                                    f.write("System Message:\n")
                                    f.write(critique_data['system_message'] + "\n\n")
                                    f.write("Critic Prompt:\n")
                                    f.write(critique_data['prompt'] + "\n\n")
                                    f.write("Critic Response:\n")
                                    f.write(critique_data['response'] + "\n\n")
                                f.write("\n")
                            else:
                                f.write("Critic Evaluation: Not evaluated\n\n")
                    else:
                        f.write("-" * 80 + "\n")
                        f.write("LLM RESPONSE (First attempt succeeded):\n")
                        f.write("-" * 80 + "\n")
                        f.write(json.dumps(structure, indent=2) + "\n\n")

                        # Log critic evaluation if it exists
                        if all_attempts and 'critique' in all_attempts[0]:
                            critique_data = all_attempts[0]['critique']
                            f.write("-" * 40 + "\n")
                            f.write("CRITIC EVALUATION\n")
                            f.write("-" * 40 + "\n")
                            if critique_data['approved']:
                                f.write("Status: ✓ APPROVED\n")
                            else:
                                f.write("Status: ✗ REJECTED\n")
                            f.write(f"Explanation: {critique_data['explanation']}\n")
                            if critique_data['suggestions']:
                                f.write(f"Suggestions: {critique_data['suggestions']}\n")
                            f.write("\n")

                            # Log the full critic conversation
                            if 'system_message' in critique_data and 'prompt' in critique_data and 'response' in critique_data:
                                f.write("-" * 40 + "\n")
                                f.write("CRITIC CONVERSATION\n")
                                f.write("-" * 40 + "\n")
                                f.write("System Message:\n")
                                f.write(critique_data['system_message'] + "\n\n")
                                f.write("Critic Prompt:\n")
                                f.write(critique_data['prompt'] + "\n\n")
                                f.write("Critic Response:\n")
                                f.write(critique_data['response'] + "\n\n")

                    # Add formatted visual representation
                    f.write("=" * 80 + "\n")
                    f.write("TEAM HIERARCHY VISUALIZATION\n")
                    f.write("=" * 80 + "\n\n")

                    def format_hierarchy(node, indent=0, prefix=""):
                        """Recursively format the hierarchy as a tree structure"""
                        lines = []

                        if 'root_manager' in node:
                            node = node['root_manager']

                        # Print manager info
                        if 'manager_type' in node:
                            manager_type_label = "FULL MANAGER" if node['manager_type'] == 'full' else "HORIZONTAL MANAGER"
                            team_name = node.get('team_name', 'UNNAMED')
                            lines.append(f"{prefix}[{manager_type_label}] {team_name}")

                            # Print children
                            children = node.get('children', [])
                            for i, child in enumerate(children):
                                is_last = (i == len(children) - 1)
                                child_prefix = prefix + ("└── " if is_last else "├── ")
                                continuation_prefix = prefix + ("    " if is_last else "│   ")

                                if child.get('type') == 'worker':
                                    worker_type = child.get('worker_type', 'unknown')
                                    count = child.get('count', 1)
                                    lines.append(f"{child_prefix}{count}x {worker_type.upper()}(s)")
                                elif child.get('type') == 'manager':
                                    lines.extend(format_hierarchy(child, indent + 1, continuation_prefix))

                        return lines

                    hierarchy_lines = format_hierarchy(structure)
                    f.write("\n".join(hierarchy_lines) + "\n\n")

                    # Add summary statistics
                    def count_agents(node, counts=None):
                        """Count total agents by type"""
                        if counts is None:
                            counts = {'managers': 0, 'full_managers': 0, 'horizontal_managers': 0, 'workers': {}}

                        if 'root_manager' in node:
                            node = node['root_manager']

                        if 'manager_type' in node:
                            counts['managers'] += 1
                            if node['manager_type'] == 'full':
                                counts['full_managers'] += 1
                            else:
                                counts['horizontal_managers'] += 1

                            for child in node.get('children', []):
                                if child.get('type') == 'worker':
                                    worker_type = child.get('worker_type', 'unknown')
                                    count = child.get('count', 1)
                                    counts['workers'][worker_type] = counts['workers'].get(worker_type, 0) + count
                                elif child.get('type') == 'manager':
                                    count_agents(child, counts)

                        return counts

                    stats = count_agents(structure)

                    f.write("-" * 80 + "\n")
                    f.write("TEAM STATISTICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Total Managers: {stats['managers']}\n")
                    f.write(f"  - Full Managers (temporal coordination): {stats['full_managers']}\n")
                    f.write(f"  - Horizontal Managers (parallel coordination): {stats['horizontal_managers']}\n")
                    f.write(f"Total Workers: {sum(stats['workers'].values())}\n")
                    for worker_type, count in sorted(stats['workers'].items()):
                        f.write(f"  - {worker_type.capitalize()}s: {count}\n")
                    f.write("\n")

                print(f"[TEAM STRUCTURE] Logged to: {log_file}")
            except Exception as log_error:
                print(f"[WARNING] Failed to write log: {log_error}")

        return structure

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
        print(f"[ERROR] Response was: {response_text}")
        raise ValueError("LLM did not return valid JSON team structure")
    except Exception as e:
        print(f"[ERROR] Failed to generate team structure: {e}")
        raise

def convert_team_structure_to_graph(
    structure: Dict[str, Any],
    worker_agents: List[Any]
) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """
    Convert LLM-generated team structure to adjacency matrix and agent_types list.

    Args:
        structure: Hierarchical team structure from LLM
        worker_agents: List of worker agent objects

    Returns:
        tuple: (graph, agent_types) where:
            - graph: Adjacency matrix
            - agent_types: List of (agent_type_code, team_name) tuples
    """
    import json

    # Organize workers by type
    workers_by_type = {
        'firefighter': [],
        'bulldozer': [],
        'drone': [],
        'helicopter': []
    }

    for worker in worker_agents:
        if worker.type == 0:
            workers_by_type['firefighter'].append(worker)
        elif worker.type == 1:
            workers_by_type['bulldozer'].append(worker)
        elif worker.type == 2:
            workers_by_type['drone'].append(worker)
        elif worker.type == 3:
            workers_by_type['helicopter'].append(worker)

    # Track agent assignments
    agent_id_map = {}  # Maps (type, index) to agent_id
    next_manager_id = len(worker_agents)  # Manager IDs start after workers
    agent_types = [None] * len(worker_agents)  # Will grow as we add managers

    # Initialize worker agent_types (all workers have type 0, no team name)
    for i in range(len(worker_agents)):
        agent_types[i] = (0, "")

    # Track worker usage
    worker_usage = {k: 0 for k in workers_by_type.keys()}

    # Build hierarchy data structures
    parent_map = {}  # child_id -> parent_id
    manager_data = {}  # manager_id -> (manager_type_code, team_name)

    def process_node(node, parent_id=None):
        """Recursively process structure nodes and assign IDs."""
        nonlocal next_manager_id

        if node.get('type') == 'worker':
            # Assign worker agents
            worker_type = node['worker_type']
            count = node['count']

            assigned_workers = []
            for _ in range(count):
                if worker_usage[worker_type] >= len(workers_by_type[worker_type]):
                    raise ValueError(f"Not enough {worker_type} agents! Need {count}, have {len(workers_by_type[worker_type])}")

                worker = workers_by_type[worker_type][worker_usage[worker_type]]
                worker_usage[worker_type] += 1
                assigned_workers.append(worker.id - 1)  # Convert to 0-indexed

            # Record parent relationship for these workers
            if parent_id is not None:
                for worker_id in assigned_workers:
                    parent_map[worker_id] = parent_id

            return assigned_workers

        elif node.get('type') == 'manager' or 'root_manager' in node:
            # Process manager node
            if 'root_manager' in node:
                node = node['root_manager']

            manager_id = next_manager_id
            next_manager_id += 1

            manager_type = node['manager_type']
            team_name = node['team_name']

            # Map manager type string to code
            manager_type_code = 2 if manager_type == 'full' else 1

            # Store manager data
            manager_data[manager_id] = (manager_type_code, team_name)
            agent_types.append((manager_type_code, team_name))

            # Record parent relationship
            if parent_id is not None:
                parent_map[manager_id] = parent_id

            # Process children
            children = node.get('children', [])
            for child in children:
                process_node(child, parent_id=manager_id)

            return manager_id

        else:
            raise ValueError(f"Unknown node type: {node}")

    # Process the structure starting from root
    root_id = process_node(structure)

    # Validate all workers were assigned
    total_expected = sum(len(workers) for workers in workers_by_type.values())
    total_assigned = sum(worker_usage.values())
    if total_assigned != total_expected:
        raise ValueError(f"Not all workers assigned! Expected {total_expected}, assigned {total_assigned}")

    # Build adjacency matrix
    total_agents = len(agent_types)
    graph = [[0 for _ in range(total_agents)] for _ in range(total_agents)]

    for child_id, parent_id in parent_map.items():
        graph[parent_id][child_id] = 1

    # Topological sort: reorder rows so children come before parents (root last)
    # Workers must come first, then managers in topological order
    # This ensures the bottom-left diagonal is all 0s (no row links to a later row)

    # Separate workers and managers
    num_workers = len(worker_agents)
    workers = [i for i in range(num_workers)]
    managers = [i for i in range(num_workers, total_agents)]

    # Topologically sort managers only (children before parents, root last)
    visited = [False] * total_agents
    manager_topo_order = []

    def topo_dfs(node):
        visited[node] = True
        # Visit all children that are managers first
        for child in range(total_agents):
            if graph[node][child] == 1 and child >= num_workers and not visited[child]:
                topo_dfs(child)
        # Add this manager after all its manager children (post-order)
        manager_topo_order.append(node)

    # Process all managers in topological order
    for mgr in managers:
        if not visited[mgr]:
            topo_dfs(mgr)

    # Post-order DFS already gives us the correct order: child managers first, then parents
    # No need to reverse - the last manager added will be the root
    # Final order: all workers first, then managers in topological order
    topo_order = workers + manager_topo_order

    # Create index mapping: old_index -> new_index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(topo_order)}

    # Reorder graph rows and columns
    new_graph = [[0 for _ in range(total_agents)] for _ in range(total_agents)]
    for old_parent in range(total_agents):
        for old_child in range(total_agents):
            if graph[old_parent][old_child] == 1:
                new_parent = old_to_new[old_parent]
                new_child = old_to_new[old_child]
                new_graph[new_parent][new_child] = 1

    # Reorder agent_types to match
    new_agent_types = [agent_types[old_idx] for old_idx in topo_order]

    # Update root_id
    new_root_id = old_to_new[root_id]

    graph = new_graph
    agent_types = new_agent_types
    root_id = new_root_id

    print(f"[TEAM STRUCTURE] Converted to graph:")
    print(f"  Total agents: {total_agents}")
    print(f"  Workers: {len(worker_agents)}")
    print(f"  Managers: {total_agents - len(worker_agents)}")
    print(f"  Root manager ID: {root_id}")

    # Validate the generated structure
    _validate_graph_and_types(graph, agent_types)

    return graph, agent_types

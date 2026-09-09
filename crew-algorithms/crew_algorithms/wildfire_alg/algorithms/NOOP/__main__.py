import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import get_agent_observations, generate_action_from_option, parse_game_data, check_if_option_done, check_game_done
import datetime
import csv
import time
import certifi
from crew_algorithms.wildfire_alg.config.build_config import EVENT_ACTION_MAP
# from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video

api_key = None

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


@hydra.main(version_base=None, config_path="../../../conf", config_name="wildfire_alg")


def wildfire_alg(cfg: Config):

    """An implementation of a wildfire alg."""
    import os
    import uuid
    import torch
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.wildfire_alg.core.utils import (
        make_env,
    )
    from crew_algorithms.wildfire_alg.algorithms.NOOP.agent import Agent
    from torchrl.record.loggers import generate_exp_name, get_logger

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


    cfg.envs.algorithm = 'NOOP'
    
    level = cfg.envs.level
    seed  = cfg.envs.seed

    levels = create_level_presets()


    firefighters = levels[level].get("starting_firefighter_agents",0)
    bulldozers = levels[level].get("starting_bulldozer_agents",0)
    drones = levels[level].get("starting_drone_agents",0)
    helicopters = levels[level].get("starting_helicopter_agents",0)
    agent_count = firefighters + bulldozers + drones + helicopters


    update_config(preset=levels[level], config=cfg.envs, log_trajectory=True, seed=seed)
    scheduled_events = list(cfg.envs.get('scheduled_events', None) or [])
    if scheduled_events:
        print(f"[EVENTS] Loaded {len(scheduled_events)} scheduled events: {scheduled_events}")
    cfg.envs.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env = make_env(cfg.envs, toggle_timestep_channel, device)
    state = env.reset()

    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["OPENAI_API_BASE"] = cfg.envs.llm_url
    
    path = os.path.join("crew_algorithms", "wildfire_alg","results", "logs", "NOOP", level, str(seed), cfg.envs.timestamp)
    os.makedirs(path, exist_ok=True)

    
    
    agents = []
    game_data = parse_game_data(state, cfg)
    print(f"Task: {game_data['task_description']}")
    for i in range(firefighters):
        a = Agent(i+1, 0, cfg, path, current_task=game_data["task_description"], api_key=api_key)
        agents.append(a)
    for i in range(bulldozers):
        a = Agent(firefighters+i+1, 1, cfg, path, current_task=game_data["task_description"], api_key=api_key)
        agents.append(a)
    for i in range(drones):
        a = Agent(firefighters+bulldozers+i+1, 2, cfg, path, current_task=game_data["task_description"], api_key=api_key)
        agents.append(a)
    for i in range(helicopters):
        a = Agent(firefighters+bulldozers+drones+i+1, 3, cfg, path, current_task=game_data["task_description"], api_key=api_key)
        agents.append(a)


    leader = agents[0]
    global_data = {}
    global_data.update({
        "api_calls" : 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "score": 0,
        "rewards": [0] * 13,
        "large_model": cfg.llms.large_model,
        "small_model": cfg.llms.small_model,
    })
    print("Agent Count: " + str(len(agents)))

    cumulative_idle_steps = 0
    cumulative_replans = 0
    game_start_time = time.time()

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

    print(f"Max Steps: {cfg.envs.max_steps}")
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
            'leader_agent': leader,
            'time': t,
            'score': game_data['score'],
            'rewards': game_data['rewards'],
        })

        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            r = global_data['rewards']
            writer.writerow([
                t,
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12],
                global_data['api_calls'], global_data['input_tokens'], global_data['output_tokens'],
                global_data['input_tokens']*0.0000025 + global_data['output_tokens']*0.00001,
                cumulative_idle_steps, cumulative_replans,
                (time.time() - game_start_time)
            ])
        f.close()
        agent_states = {}

        if check_game_done(global_data=global_data, cfg= cfg.envs, past_score=past_score):
            break
        


        env_action = [[0,0,0] for _ in range(cfg.envs.num_agents)]


        # Scheduled event injection
        scheduled_event_action = None
        events_this_step = [e for e in scheduled_events if e["timestep"] == t]
        if events_this_step:
            event = events_this_step[0]
            action_code = EVENT_ACTION_MAP[event["action"]]
            x = float(event.get("x", 0))
            y = float(event.get("y", 0))
            scheduled_event_action = [action_code, x, y]
            print(f"[EVENT] t={t}: {event.get('announcement', event['action'])}")
        if scheduled_event_action is not None:
            env_action[0] = scheduled_event_action



        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]


    env.close()
    print("TEST COMPLETE")
    # compile_split_screen_video(path, os.path.join(path, "render.mp4"))

if __name__ == "__main__":
    try:
        wildfire_alg()
    except Exception as e:
        print(f"Error in wildfire_alg: {e}")
        exit(1)
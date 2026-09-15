import hydra
from attrs import define
from crew_algorithms.envs.configs import EnvironmentConfig, register_env_configs
from crew_algorithms.wildfire_alg.config.configs import LLMConfig, configure_model, model_directory_name
from crew_algorithms.utils.wandb_utils import WandbConfig
from crew_algorithms.wildfire_alg.config.build_config import update_config, create_level_presets
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import numpy as np
from crew_algorithms.wildfire_alg.core.alg_utils import get_agent_observations, parse_game_data, check_game_done
import datetime
import csv
import time
from openai import OpenAI
from crew_algorithms.wildfire_alg.libraries.firefighter_action_library import Run_Firefighter_Action
from crew_algorithms.wildfire_alg.libraries.bulldozer_action_library import Run_Bulldozer_Action
from crew_algorithms.wildfire_alg.libraries.drone_action_library import Run_Drone_Action
from crew_algorithms.wildfire_alg.libraries.helicopter_action_library import Run_Helicopter_Action
import certifi
from crew_algorithms.wildfire_alg.config.build_config import EVENT_ACTION_MAP
#from crew_algorithms.wildfire_alg.data.render_logs import compile_split_screen_video

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
    from pathlib import Path

    import torch
    from crew_algorithms.envs.channels import ToggleTimestepChannel
    from crew_algorithms.wildfire_alg.core.utils import (
        make_env,
    )
    from crew_algorithms.wildfire_alg.algorithms.HMAS_2.agent import Agent
    from crew_algorithms.wildfire_alg.algorithms.HMAS_2.utils import propose_actions, provide_feedback, translate_action
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

    # Sets up environment/game configs

    cfg.envs.algorithm = 'HMAS_2'
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

    # Read selected LLM model from top-level config and validate
    llm_model = cfg.envs.llm_model
    configure_model(cfg)
        
        
    # Local OpenAI-compatible servers receive a harmless placeholder when no
    # API key is configured.
    if llm_model == "gpt":
        api_key = os.getenv("BASELINE_API_KEY") or cfg.envs.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API_KEY or OPENAI_API_KEY is required for the OpenAI API")
    else:
        api_key = os.getenv("BASELINE_API_KEY") or cfg.envs.api_key or "EMPTY"
    
    os.environ["OPENAI_API_BASE"] = cfg.envs.llm_url
    
    path = os.path.join("crew_algorithms", "wildfire_alg", "results", "logs", "HMAS_2", model_directory_name(cfg), level, str(seed), cfg.envs.timestamp)
    os.makedirs(path, exist_ok=True)
    os.environ["SSL_CERT_FILE"] = certifi.where()
    
    agents = []
    game_data = parse_game_data(state, cfg)
    print(f"Task: {game_data['task_description']}")

    for i in range(firefighters):
        a = Agent(i+1, 0, cfg, path, current_task=game_data["task_description"], api_key=api_key, agent_count = agent_count)
        agents.append(a)
    for i in range(bulldozers):
        a = Agent(firefighters+i+1, 1, cfg, path, current_task=game_data["task_description"], api_key=api_key, agent_count = agent_count)
        agents.append(a)
    for i in range(drones):
        a = Agent(firefighters+bulldozers+i+1, 2, cfg, path, current_task=game_data["task_description"], api_key=api_key, agent_count = agent_count)
        agents.append(a)
    for i in range(helicopters):
        a = Agent(firefighters+bulldozers+drones+i+1, 3, cfg, path, current_task=game_data["task_description"], api_key=api_key, agent_count = agent_count)
        agents.append(a)

    global_data = {}
    global_data.update({
        "api_calls" : 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "score": 0,
        "rewards": [0] * 13,
        'step_history': {},
        "api_key": api_key,
        "llm_url": cfg.envs.llm_url,
        "large_model": cfg.llms.large_model,
        "small_model": cfg.llms.small_model,
    })
    print("Agent Count: " + str(len(agents)))

    cumulative_idle_steps = 0
    cumulative_replans = 0
    max_feedback_rounds = 0
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

    if llm_model == "gpt":
        client = OpenAI(
            api_key=api_key,
            timeout=3600,
        )
    else:
        client = OpenAI(
            api_key=api_key,
            base_url=cfg.envs.llm_url,
            timeout=3600,
        )
        
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
            'agents': agents,
            'time': t,
            'client': client,
            'path': path,
            'score': game_data["score"],
            'rewards': game_data['rewards'],
        })

        state_action_max_length = 5
        remove_times = []
        for state_time in global_data["step_history"].keys():
            if t-int(state_time[-1])>state_action_max_length:
                remove_times.append(state_time)
        for remove_time in remove_times:
            global_data["step_history"].pop(remove_time)



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

        for agent in agents:
            observations = get_agent_observations(state, agent.id)
            if observations["agent_type"] >=4:
                print(f"Agent {agent.id} DESTROYED")
                removelist.append(agent)
                continue
            
            elif observations["agent_type"] ==0:
                global_data["firefighters"].append(agent)
            elif observations["agent_type"] ==1:
                global_data["bulldozers"].append(agent)
            elif observations["agent_type"] ==2:
                global_data["drones"].append(agent)
            elif observations["agent_type"] ==3:
                global_data["helicopters"].append(agent)
            
            agent.last_observation = observations["perception_grid"]
            agent.last_position = observations["position"]
            agent.last_current_cell = observations["current_cell"]
            agent.map_range = observations["map_range"]
            agent_states.update({agent.id: agent.last_position})
            agent.extra_variables = observations["extra_variables"]

        for r in removelist:
            agents.remove(r)
        
        if check_game_done(global_data=global_data, cfg= cfg.envs, past_score=past_score):
            break
       
        for agent in agents:
            agent.generate_perception(cfg.envs, agent_states, global_data)

        
        proposed_actions, messages = propose_actions(global_data=global_data, past_conversation=[])
        feedback = {}

        feedback_round = 0
        while True:
            satisfactory = True
            feedback = {}
            for agent in agents:
                agent_feedback = provide_feedback(agent=agent, global_data=global_data, proposed_actions=proposed_actions)
                agent_feedback = agent_feedback or "NO FEEDBACK FOUND"
                if agent_feedback.strip().strip("'\"").upper() != "ACCEPT":
                    satisfactory=False
                feedback.update({f"AGENT_{agent.id}": agent_feedback})
            
            if satisfactory:
                break
            if feedback_round >= max_feedback_rounds:
                print(
                    f"[WARN] Feedback did not converge after {max_feedback_rounds} replans at timestep {t}. "
                    "Proceeding with the latest proposed actions."
                )
                print(f"[WARN] Final feedback: {feedback}")
                break
            else:
                feedback_round += 1
                cumulative_replans += 1
                messages.append({"role": "user", "content": f"""
                                 This is your team's feedback. Adjust the plan accordingly. 
                                 Specify the revised action plan in the same format.
                                 
                                 {feedback}"""})
                proposed_actions, messages = propose_actions(global_data=global_data, past_conversation=messages)
                
            # print("FEEDBACK:", feedback)
            # print("REVISED PROPOSED ACTIONS:", proposed_actions)
        
        env_action = [[0,0,0] for _ in range(cfg.envs.num_agents)]

        global_data["step_history"].update({f"time: {t}": {}})
        for agent in agents:
            action_str = proposed_actions.get(f"AGENT_{agent.id}")
            action = translate_action(action_str=action_str, type = agent.type, global_data=global_data)
            agent.log_chat("Executing Actions", [("system", action)])
            print(f"AGENT_{agent.id}: {action.description}")
            try:
                libraries = {
                    0: Run_Firefighter_Action,
                    1: Run_Bulldozer_Action,
                    2: Run_Drone_Action,
                    3: Run_Helicopter_Action
                }
                if agent.type in libraries:
                    action_array = libraries[agent.type](agent, action)
                    env_action[agent.id] = action_array
                    agent.log_chat("", [("system", action_array)])
                    global_data["step_history"][f"time: {t}"].update({f"AGENT_{agent.id}": {"state":agent.last_position, "action": action.description}})

            except:
                agent.log_chat("ERROR", [("system", f"ERROR EXECUTING ACTION: {action}")])
                agent.past_actions.append(f"ERROR EXECUTING ACTION: {action.description}")
                global_data["step_history"][f"time: {t}"].update({f"AGENT_{agent.id}": {"state":agent.last_position, "action": f"ERROR EXECUTING ACTION: {action.description}"}})

            # Track idle steps
            _is_helicopter_passenger = (agent.type == 0 and agent.extra_variables[2] == 1)
            
            try:
                if action.type == 0 and not _is_helicopter_passenger:
                    cumulative_idle_steps += 1
            except:
                print(f"Error processing action for agent {agent.id}: {action}")
                pass

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

        for i in range(len(env_action)):
            if env_action[i] is None:
                env_action[i] = [0,0,0]

        action_tensor = torch.from_numpy(np.array(env_action)).to(device)
        state["agents"]["action"] = action_tensor
        newstate = env.step(state)
        state["agents"]["observation"] = newstate["next"]["agents"]["observation"]

            
        

        
    env.close()
    print("TEST COMPLETE")
    #compile_split_screen_video(path, os.path.join(path, "render.mp4"))

        

            

if __name__ == "__main__":
    try:
        wildfire_alg()
    except Exception as e:
        print(f"Error in wildfire_alg: {e}")
        exit(1)

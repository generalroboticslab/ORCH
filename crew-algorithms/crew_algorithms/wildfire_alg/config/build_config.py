import json

# Maps human-readable event action names to Unity pseudo-manager action codes (5f-11f)
EVENT_ACTION_MAP = {
    "start_fire": 5,        # SetFire(x, y)
    "sim_steps": 6,         # Trigger X sim steps (x = count, y ignored)
    "add_water": 7,         # Add permanent water at (x, y)
    "remove_water": 8,      # Remove water at (x, y)
    "spawn_civilian": 9,    # Spawn 1 civilian at (x, y)
    "spawn_civilians_5": 10,  # Spawn 5 civilians at (x, y)
    "destroy_agent": 11,    # Destroy agent by ID (x = agent_id, y ignored)
}

def create_level_presets():
    presets = {}

    # Cut Trees: Sparse
    presets['Cut_Trees_Sparse_small'] = {
        'name': 'Cut Trees: Sparse (Small)',
        'description': 'Small map with sparse trees to cut down',
        'max_steps': 30,
        'game_type': 0, 'map_size': 30, 'lines': False, 'tree_count': 6, 'trees_per_line': 1,
        'starting_firefighter_agents': 3, 'known': True,
    }
    presets['Cut_Trees_Sparse_large'] = {
        'name': 'Cut Trees: Sparse (Large)',
        'description': 'Large map with many sparse trees to cut down',
        'max_steps': 50,
        'game_type': 0, 'map_size': 60, 'lines': False, 'tree_count': 25, 'trees_per_line': 1,
        'starting_firefighter_agents': 10, 'known': True,
    }

    # Cut Trees: Lines
    presets['Cut_Trees_Lines_small'] = {
        'name': 'Cut Trees: Lines (Small)',
        'description': 'Small map with trees in lines to create firebreaks',
        'max_steps': 30,
        'game_type': 0, 'map_size': 30, 'lines': True, 'tree_count': 2, 'trees_per_line': 5,
        'starting_firefighter_agents': 2, 'starting_bulldozer_agents': 1, 'vegetation_density_offset': 30, 'known': True,
    }
    presets['Cut_Trees_Lines_large'] = {
        'name': 'Cut Trees: Lines (Large)',
        'description': 'Large map with multiple tree lines to cut',
        'max_steps': 30,
        'game_type': 0, 'map_size': 60, 'lines': True, 'tree_count': 5, 'trees_per_line': 7,
        'starting_firefighter_agents': 4, 'starting_bulldozer_agents': 3, 'vegetation_density_offset': 30, 'known': True,
    }

    # Scout Fire
    presets['Scout_Fire_small'] = {
        'name': 'Scout Fire (Small)',
        'description': 'Scout for fires in a small area using drones',
        'max_steps': 30,
        'game_type': 1, 'map_size': 100, 'fire_spread_frequency': 200,'known': False,
        'starting_drone_agents': 3
    }
    presets['Scout_Fire_large'] = {
        'name': 'Scout Fire (Large)',
        'description': 'Scout for fires across a large area',
        'max_steps': 30,
        'game_type': 1, 'map_size': 250, 'fire_spread_frequency': 300,'known': False,
        'starting_drone_agents': 5
    }

    # Transport Firefighters
    presets['Transport_Firefighters_small'] = {
        'name': 'Transport Firefighters (Small)',
        'description': 'Use helicopters to transport firefighters',
        'max_steps': 15,
        'game_type': 2, 'map_size': 100,
        'starting_firefighter_agents': 6, 'starting_helicopter_agents': 1, 'known': True,
    }
    presets['Transport_Firefighters_large'] = {
        'name': 'Transport Firefighters (Large)',
        'description': 'Large-scale firefighter transport operations',
        'max_steps': 35,
        'game_type': 2, 'map_size': 250,
        'starting_firefighter_agents': 12, 'starting_helicopter_agents': 2, 'known': True,
    }

    # Rescue Civilians: Known Location
    presets['Rescue_Civilians_Known_Location_small'] = {
        'name': 'Rescue Civilians: Known Location (Small)',
        'description': 'Rescue civilians from known locations',
        'max_steps': 25,
        'game_type': 4, 'map_size': 40, 'civilian_count': 3, 'civilian_clusters': 1, 'civilian_move_frequency': 500, 'known': True,
        'starting_firefighter_agents': 3
    }
    presets['Rescue_Civilians_Known_Location_large'] = {
        'name': 'Rescue Civilians: Known Location (Large)',
        'description': 'Large-scale civilian rescue from known locations',
        'max_steps': 35,
        'game_type': 4, 'map_size': 80, 'civilian_count': 3, 'civilian_clusters': 3, 'civilian_move_frequency': 500, 'known': True,
        'starting_firefighter_agents': 5
    }

    # Suppress Fire: Contain
    presets['Suppress_Fire_Contain'] = {
        'name': 'Suppress Fire: Contain',
        'description': 'Contain fire using firefighters and bulldozers',
        'max_steps': 100,
        'game_type': 3, 'map_size': 60, 'water': False, 'fire_spread_frequency': 500,
        'starting_firefighter_agents': 5, 'starting_bulldozer_agents': 2, 'vegetation_density_offset': 40, 'known': True,
    }

    # Suppress Fire: Extinguish
    presets['Suppress_Fire_Extinguish'] = {
        'name': 'Suppress Fire: Extinguish',
        'description': 'Extinguish fire using water from firefighters',
        'max_steps': 100,
        'game_type': 3, 'map_size': 60, 'water': True, 'fire_spread_frequency': 500,
        'starting_firefighter_agents': 8, 'vegetation_density_offset': 40, 'known': True,
    }

    # Rescue Civilians: Search and Rescue
    presets['Rescue_Civilians_Search_and_Rescue'] = {
        'name': 'Rescue Civilians: Search and Rescue',
        'description': 'Search for and rescue civilians using combined forces',
        'max_steps': 40,
        'game_type': 4, 'map_size': 100, 'civilian_count': 5, 'civilian_clusters': 1, 'civilian_move_frequency': 600,
        'starting_firefighter_agents': 5, 'starting_drone_agents': 2, 'known': False,
    }

    # Suppress Fire: Locate and Suppress
    presets['Suppress_Fire_Locate_and_Suppress'] = {
        'name': 'Suppress Fire: Locate and Suppress',
        'description': 'Locate fires and suppress them with mixed units',
        'max_steps': 100,
        'game_type': 3, 'map_size': 120, 'water': True, 'fire_spread_frequency': 600,
        'starting_firefighter_agents': 5, 'starting_drone_agents': 2, 'starting_bulldozer_agents': 1, 'vegetation_density_offset': 40, 'known': False,
    }

    # Suppress Fire: Locate + Deploy + Suppress
    presets['Suppress_Fire_Locate_Deploy_Suppress'] = {
        'name': 'Suppress Fire: Locate, Deploy & Suppress',
        'description': 'Complex fire suppression with all unit types',
        'max_steps': 120,
        'game_type': 3, 'map_size': 150, 'water': True, 'fire_spread_frequency': 600,
        'starting_firefighter_agents': 10, 'starting_drone_agents': 2, 'starting_helicopter_agents': 2, 'vegetation_density_offset': 40, 'known': False,
    }

    # Rescue Civilians: Search + Rescue + Transport
    presets['Rescue_Civilians_Search_Rescue_Transport'] = {
        'name': 'Rescue Civilians: Search, Rescue & Transport',
        'description': 'Complete civilian rescue operations with transport',
        'max_steps': 60,
        'game_type': 4, 'map_size': 150, 'civilian_count': 5, 'civilian_clusters': 2, 'civilian_move_frequency': 600, 'known': False,
        'starting_firefighter_agents': 10, 'starting_drone_agents': 2, 'starting_helicopter_agents': 2
    }


    # Full Game
    presets['Full_Game'] = {
        'name': 'Full Game',
        'description': 'Complete wildfire response scenario with all objectives',
        'max_steps': 120,
        'game_type': 5, 'map_size': 220, 'fire_spread_frequency': 450,
        'civilian_count': 5, 'civilian_clusters': 1, 'civilian_move_frequency': 300,
        'starting_firefighter_agents': 10, 'starting_bulldozer_agents': 1, 'starting_drone_agents': 2, 'starting_helicopter_agents': 2, 'vegetation_density_offset': 40, 'known': False,
        'water': True,
    }

    presets['Demo_Level'] = {
        'name': 'Demo Level',
        'description': 'Suppress fire with civilians',
        'max_steps': 20,
        'game_type': 5, 'map_size': 100, 'fire_spread_frequency': 500,
        'civilian_count': 5, 'civilian_clusters': 1, 'civilian_move_frequency': 300,
        'starting_firefighter_agents': 5, 'starting_bulldozer_agents': 0, 'starting_drone_agents': 1, 'starting_helicopter_agents': 0, 'vegetation_density_offset': 30, 'known': True,
        'water': True,
    }

    presets['Scale_Level_Complex'] = {
        'name': 'Scale Level: Complex',
        'description': 'Large-scale scenario with many agents of all types and civilians.',
        'max_steps': 200,
        'game_type': 5,
        'map_size': 200,
        'fire_spread_frequency': 400,
        'civilian_count': 5,
        'civilian_clusters': 1,
        'civilian_move_frequency': 300,
        'starting_firefighter_agents': 25,
        'starting_bulldozer_agents': 5,
        'starting_drone_agents': 10,
        'starting_helicopter_agents': 10,
        'vegetation_density_offset': 40,
        'known': False,
        'water': True,
    }
    
    presets['Scale_Level_Simple'] = {
        'name': 'Scale Level: Simple',
        'description': 'Large map with many trees and firefighters, simple scenario.',
        'max_steps': 100,
        'game_type': 0,
        'map_size': 200,
        'lines': False,
        'tree_count': 100,
        'trees_per_line': 1,
        'starting_firefighter_agents': 50,
        'vegetation_density_offset': 40,
        'known': True,
    }
    
    presets['DARPA_500'] = {
        'name': 'DARPA 500',
        'description': 'Large tree-cutting task with 500 firefighters on a 1000x1000 map',
        'max_steps': 200,
        'game_type': 0,
        'map_size': 1000,
        'lines': False,
        'tree_count': 500,
        'trees_per_line': 1,
        'starting_firefighter_agents': 500,
        'vegetation_density_offset': 40,
        'known': True,
    }
    
    presets['DARPA_stress'] = {
        'name': 'DARPA stress',
        'description': 'Large tree-cutting task with 5000 firefighters on a 1000x1000 map',
        'max_steps': 200,
        'game_type': 0,
        'map_size': 1000,
        'lines': False,
        'tree_count': 500,
        'trees_per_line': 1,
        'starting_firefighter_agents': 5000,
        'vegetation_density_offset': 40,
        'known': True,
    }

    presets['DARPA_stress_2'] = {
        'name': 'DARPA stress 2',
        'description': 'Large tree-cutting task with 10000 firefighters on a 1000x1000 map',
        'max_steps': 200,
        'game_type': 0,
        'map_size': 1000,
        'lines': False,
        'tree_count': 500,
        'trees_per_line': 1,
        'starting_firefighter_agents': 100_000,
        'vegetation_density_offset': 40,
        'known': True,
    }

    presets['VLM_Collection'] = {
        'name': 'VLM Collection',
        'description': 'Tiny wildfire scenario with 1 firefighter and a few civilians.',
        'max_steps': 100,
        'game_type': 5,
        'map_size': 20,
        'fire_spread_frequency': 500,
        'civilian_count': 3,
        'civilian_clusters': 1,
        'civilian_move_frequency': 500,
        'starting_firefighter_agents': 1,
        'vegetation_density_offset': 30,
        'known': True,
        'water': True,
    }

    # --- Dynamic "Special Levels" with scheduled mid-game events ---

    # Scout Fire: Drone Lost — a drone goes missing mid-game
    # Agent IDs: drones 1-3
    presets['Scout_Fire_Drone_Lost'] = {
        'name': 'Scout Fire: Drone Lost',
        'description': 'A drone goes missing mid-game — adapt with remaining scouts',
        'max_steps': 35,
        'game_type': 1, 'map_size': 150, 'fire_spread_frequency': 200, 'known': False,
        'starting_drone_agents': 3,
        'scheduled_events': [
            {"timestep": 4, "action": "destroy_agent", "x": 1, "y": 0,
             "announcement": "Communication lost with AGENT_1. A drone has gone missing."},
        ],
    }

    # Transport Firefighters: Helicopter Down — a helicopter goes out of commission
    # Agent IDs: firefighters 1-10, helicopters 11-12
    presets['Transport_Helicopter_Down'] = {
        'name': 'Transport: Helicopter Down',
        'description': 'A helicopter goes out of commission mid-transport',
        'max_steps': 35,
        'game_type': 2, 'map_size': 250,
        'starting_firefighter_agents': 10, 'starting_helicopter_agents': 2, 'known': True,
        'scheduled_events': [
            {"timestep": 4, "action": "destroy_agent", "x": 11, "y": 0,
             "announcement": "Helicopter down! AGENT_11 is out of commission."},
        ],
    }

    # Rescue Civilians: Surprise — extra civilians appear mid-game
    # Agent IDs: firefighters 1-3
    presets['Rescue_Civilians_Surprise'] = {
        'name': 'Rescue: Surprise Civilians',
        'description': 'Extra civilians appear mid-game in a new area',
        'max_steps': 60,
        'game_type': 4, 'map_size': 80, 'civilian_count': 5, 'civilian_clusters': 2,
        'civilian_move_frequency': 800, 'known': True,
        'starting_firefighter_agents': 5,
        'scheduled_events': [
            {"timestep": 9, "action": "spawn_civilians_5", "x": 70, "y": 70,
             "announcement": "Additional civilians have been spotted near 70,70!"},
        ],
    }

    # Suppress Fire: Extinguish + Second Fire — a second fire spawns mid-game
    # Agent IDs: firefighters 1-8
    presets['Suppress_Fire_Extinguish_Second_Fire'] = {
        'name': 'Suppress Fire: Second Fire',
        'description': 'A second fire breaks out mid-game — split resources to handle both',
        'max_steps': 100,
        'game_type': 3, 'map_size': 80, 'water': True, 'fire_spread_frequency':400,
        'starting_firefighter_agents': 8, 'vegetation_density_offset': 30, 'known': True,
        'scheduled_events': [
            {"timestep": 18, "action": "start_fire", "x": 70, "y": 70,
             "announcement": "A second fire has broken out at coordinates (70, 70)!"},
        ],
    }

    # Suppress Fire: Contain + Water Source — a water source appears mid-game
    # Agent IDs: firefighters 1-5
    presets['Suppress_Fire_Contain_Water_Source'] = {
        'name': 'Suppress Fire: Water Source Appears',
        'description': 'Start without water — a water source appears mid-game, unlocking spray',
        'max_steps': 100,
        'game_type': 3, 'map_size': 80, 'water': False, 'fire_spread_frequency': 400,
        'starting_firefighter_agents': 5,
        'vegetation_density_offset': 30, 'known': True,
        'scheduled_events': [
            {"timestep": 10, "action": "add_water", "x": 30, "y": 30,
             "announcement": "A water source has appeared at coordinates (30, 30)! Firefighters can now refill water to extinguish fires. This permnmently OVERRIDES all assumptions of no water"},
        ],
    }

    # Suppress Fire: Extinguish + Rapid Growth — fire skips 5 sim steps, growing much larger
    # Agent IDs: firefighters 1-5, drone 6, helicopter 7
    presets['Suppress_Fire_Extinguish_Rapid_Growth'] = {
        'name': 'Suppress Fire: Rapid Growth',
        'description': 'Fire suddenly spreads rapidly due to worsening conditions',
        'max_steps': 100,
        'game_type': 3, 'map_size': 80, 'water': True, 'fire_spread_frequency': 400,
        'starting_firefighter_agents': 5, 'starting_drone_agents': 1, 'starting_helicopter_agents': 1,
        'vegetation_density_offset': 30, 'known': True,
        'scheduled_events': [
            {"timestep": 15, "action": "sim_steps", "x": 5, "y": 0,
             "announcement": "Conditions have worsened! High winds have caused the fire to spread rapidly."},
        ],
    }

    return presets


def get_levels_for_frontend():
    """Generate frontend-compatible level data from presets (single source of truth).

    Returns a dict matching the format previously in shared_levels.json:
    { "Level_Key": { "name": ..., "description": ..., "map_size": ..., "max_steps": ..., "agents": {...} } }
    """
    presets = create_level_presets()
    levels = {}

    agent_type_keys = [
        ('starting_firefighter_agents', 'firefighters'),
        ('starting_bulldozer_agents', 'bulldozers'),
        ('starting_drone_agents', 'drones'),
        ('starting_helicopter_agents', 'helicopters'),
    ]

    for key, preset in presets.items():
        agents = {}
        for preset_key, frontend_key in agent_type_keys:
            agents[frontend_key] = preset.get(preset_key, 0)

        levels[key] = {
            "name": preset.get("name", key),
            "description": preset.get("description", ""),
            "map_size": preset.get("map_size", 100),
            "max_steps": preset.get("max_steps", 100),
            "agents": agents,
        }

    return levels


def update_config(preset, seed, config, log_trajectory=False):
    # Define all possible game-related config keys
    game_config_keys = {
        'map_size', 'lines', 'tree_count', 'trees_per_line',
        'fire_spread_frequency', 'water', 'civilian_count', 'civilian_clusters', 'civilian_move_frequency', 'known'
    }

    # Define task-specific required variables
    task_variables = {
        0: {'map_size', 'lines', 'tree_count', 'trees_per_line'},
        1: {'map_size', 'fire_spread_frequency'},
        2: {'map_size', },
        3: {'map_size', 'water', 'fire_spread_frequency', 'known'},
        4: {'map_size', 'civilian_count', 'civilian_clusters', 'civilian_move_frequency', 'known'},
        5: {'map_size', 'fire_spread_frequency', 'civilian_count', 'civilian_clusters', 'civilian_move_frequency', 'known', 'water'}
    }

    # Default adjustable parameters
    default_values = {
        'starting_firefighter_agents': 0,
        'starting_bulldozer_agents': 0,
        'starting_drone_agents': 0,
        'starting_helicopter_agents': 0,
        'steps_per_decision': 10,
        'vegetation_density_offset':20,
        'log_trajectory': log_trajectory
    }

    if 'game_type' not in preset:
        raise ValueError("Preset must include 'game_type' key.")

    game_type = preset['game_type']
    if game_type not in task_variables:
        raise ValueError(f"Invalid game type: {game_type}")

    # Update game-related keys
    for key in game_config_keys:
        if key in task_variables[game_type]:
            if key in preset:
                config[key] = preset[key]
            else:
                raise ValueError(f"Missing required key '{key}' for game type {game_type}.")
        else:
            config[key] = 0  # Irrelevant for this task, set to 0
    config['game_type'] = game_type

    # Set max_steps from preset (single source of truth)
    config['max_steps'] = preset.get('max_steps', 100)

    # Always update seed and log_trajectory directly
    config['seed'] = seed
    config['log_trajectory'] = log_trajectory

    # Update adjustable agent settings from preset if given, else use default
    for key, value in default_values.items():
        config[key] = preset.get(key, value)

    # Pass through scheduled events with same-timestep collision resolution
    raw_events = preset.get('scheduled_events', [])
    events = sorted(raw_events, key=lambda e: e['timestep'])
    seen_timesteps = set()
    for evt in events:
        while evt['timestep'] in seen_timesteps:
            evt['timestep'] += 1
        seen_timesteps.add(evt['timestep'])
    config['scheduled_events'] = events

from enum import Enum
from crew_algorithms.wildfire_alg.core.gpt import Action


# class Firefighter_Option_Library(Enum):
#     "move_to_location" = 1
#     "cut_x_trees" = 2
#     "cut_trees_until_x_remain" = 3
#     "cut_all_trees" = 4


def Run_Bulldozer_Option(agent, option):
    """
    Convert high-level bulldozer options into a sequence of low-level actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
            - action_queue: Queue to store generated actions
        option: Option object containing:
            - type: Type of option to perform:
                1: Move to location without cutting
                2: Move to location while cutting trees
                0: Become idle
            - param_1: First parameter (x-coordinate)
            - param_2: Second parameter (y-coordinate)
            - description: Human-readable description of the option
    
    Effects:
        Appends Action objects to agent.action_queue. Each Action contains:
            - done: Whether this is the last action in the sequence
            - action: Type of action (0: move, 1: move while cutting)
            - x: x-coordinate for the action
            - y: y-coordinate for the action
            - explanation: Description of the action
            
    Raises:
        ValueError: If target location is out of map bounds
    """

    match option.type:

        # Move to location not cutting
        case 1:
            if option.param_1 < 0 or option.param_1 > agent.cfg.envs.map_size or option.param_2 < 0 or option.param_2 > agent.cfg.envs.map_size:
                option.description += ": OUT OF BOUNDS"
                raise ValueError("out of bounds target")
            
            if agent.last_position == (option.param_1, option.param_2):
                a = Action(done = True, action=0, x=agent.last_position[0], y=agent.last_position[1],explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = False, action=0, x=int(option.param_1), y=int(option.param_2), explanation="")
                agent.action_queue.append(a)
        
        # Move to location cutting
        case 2:

            if agent.last_position == (option.param_1, option.param_2):
                a = Action(done = True, action=0,x=0,y=0,explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = False, action=1, x=int(option.param_1), y=int(option.param_2), explanation="")
                agent.action_queue.append(a)
        
        
        # Become Idle
        case 0:
            a = Action(done=True, action= 0, x=agent.last_position[0], y=agent.last_position[1], explanation="")
            agent.action_queue.append(a)



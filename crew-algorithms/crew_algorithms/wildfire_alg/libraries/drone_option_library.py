from enum import Enum
from crew_algorithms.wildfire_alg.core.gpt import Action


def Run_Drone_Option(agent, action):
    """
    Convert high-level drone options into a sequence of low-level actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
            - action_queue: Queue to store generated actions
        action: Action object containing:
            - type: Type of option to perform:
                1: Move to location
                0: Become idle
            - param_1: First parameter (x-coordinate)
            - param_2: Second parameter (y-coordinate)
            - description: Human-readable description of the action
    
    Effects:
        Appends Action objects to agent.action_queue. Each Action contains:
            - done: Whether this is the last action in the sequence
            - action: Type of action (0: move)
            - x: x-coordinate for the action
            - y: y-coordinate for the action
            - explanation: Description of the action
            
    Raises:
        ValueError: If target location is out of map bounds
    """

    match action.type:

        # Move to location
        case 1:
            if action.param_1 < 0 or action.param_1 > agent.cfg.envs.map_size or action.param_2 < 0 or action.param_2 > agent.cfg.envs.map_size:
                action.description += ": OUT OF BOUNDS"
                raise ValueError("out of bounds target")
            
            if agent.last_position == (action.param_1, action.param_2):
                a = Action(done = True, action=0, x=agent.last_position[0], y=agent.last_position[1],explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = False, action=0, x=int(action.param_1), y=int(action.param_2), explanation="")
                agent.action_queue.append(a)
        
        
        # Become Idle
        case 0:
            a = Action(done=True, action= 0, x=agent.last_position[0], y=agent.last_position[1], explanation="")
            agent.action_queue.append(a)



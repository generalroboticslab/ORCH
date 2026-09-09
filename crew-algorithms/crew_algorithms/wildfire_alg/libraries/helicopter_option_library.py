from enum import Enum
from crew_algorithms.wildfire_alg.core.gpt import Action


    # - Move to any coordinate location. Distance does not matter, you will reach that location in one move.
    # - Pick up nearby Firefighter Agents.
    # - Drop off all carried Firefighter Agents.
    # - Refill water storage.
    # - Deploy water directly below.
    # - Do nothing, remaining on standby and conserving energy. This will last until the rest of the task.

def Run_Helicopter_Option(agent, option):
    """
    Convert high-level helicopter options into a sequence of low-level actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
            - action_queue: Queue to store generated actions
        option: Option object containing:
            - type: Type of option to perform:
                1: Move to location
                2: Pick up firefighters
                3: Drop off firefighters
                4: Refill water
                5: Deploy water
                0: Become idle
            - param_1: First parameter (x-coordinate for movement)
            - param_2: Second parameter (y-coordinate for movement)
            - description: Human-readable description of the option
    
    Effects:
        Appends Action objects to agent.action_queue. Each Action contains:
            - done: Whether this is the last action in the sequence
            - action: Type of action (0: move, 1: pickup, 2: refill, 3: drop/deploy)
            - x: x-coordinate for the action
            - y: y-coordinate for the action
            - explanation: Description of the action
            
    Raises:
        ValueError: If target location is out of map bounds
    """

    match option.type:

        # Move to location
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
        
        # Pick up firefighters
        case 2:
            a = Action(done = True, action=1, x=0, y=0, explanation="")
            agent.action_queue.append(a)

        
        # Drop off firefighters
        case 3:
            a = Action(done = True, action=3, x=0, y=0, explanation="")
            agent.action_queue.append(a)


        # Refill Water
        case 4:
            a = Action(done = True, action=2, x=0, y=0, explanation="")
            agent.action_queue.append(a)

        # Deploy Water
        case 5:
            a = Action(done = True, action=3, x=0, y=0, explanation="")
            agent.action_queue.append(a)

        
        # Become Idle
        case 0:
            a = Action(done=True, action= 0, x=agent.last_position[0], y=agent.last_position[1], explanation="")
            agent.action_queue.append(a)



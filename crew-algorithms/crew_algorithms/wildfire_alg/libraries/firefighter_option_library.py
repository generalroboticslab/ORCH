from enum import Enum
from crew_algorithms.wildfire_alg.core.gpt import Action



# class Firefighter_Option_Library(Enum):
#     "move_to_location" = 1
#     "cut_x_trees" = 2
#     "cut_trees_until_x_remain" = 3
#     "cut_all_trees" = 4


def Run_Firefighter_Option(agent, option):
    """
    Convert high-level firefighter options into a sequence of low-level actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
            - last_current_cell: Value of the cell at current position
            - extra_variables: List of state variables (extra_variables[0] is civilian carry state)
            - action_queue: Queue to store generated actions
        option: Option object containing:
            - type: Type of option to perform:
                1: Move to location
                2: Cut x trees
                3: Cut all trees
                4: Pick up civilian
                5: Drop off civilian
                6: Spray water
                7: Refill water
                0: Become idle
            - param_1: First parameter (x-coordinate or number of trees)
            - param_2: Second parameter (y-coordinate)
    
    Effects:
        Appends Action objects to agent.action_queue. Each Action contains:
            - done: Whether this is the last action in the sequence
            - action: Type of action (0: move, 1: cut, 2: interact, 3: spray, 4: refill)
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
                raise ValueError("out of bounds target")

            if agent.last_position == (option.param_1, option.param_2):
                a = Action(done = True, action=0, x=agent.last_position[0], y=agent.last_position[1],explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = False, action=0, x=int(option.param_1), y=int(option.param_2), explanation="")
                agent.action_queue.append(a)
        
        # Cut x trees
        case 2:

            for _ in range(option.param_1-1):
                a = Action(done = False, action=1, x=0, y=0, explanation="")
                agent.action_queue.append(a)
            

            a = Action(done = True, action=1,x=0,y=0,explanation="")
            agent.action_queue.append(a)
        

        # Cut all trees
        case 3:

            if agent.last_current_cell == '0':
                a = Action(done = True, action=0,x=0,y=0,explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = False, action=1, x=0, y=0, explanation="")
                agent.action_queue.append(a)

        # Pick up
        case 4:
            if agent.extra_variables[0]==0:
                a = Action(done = True, action=2,  x=agent.last_position[0], y=agent.last_position[1], explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = True, action=0,  x=agent.last_position[0], y=agent.last_position[1], explanation="")
                agent.action_queue.append(a)

        # Drop off Carried Civilian
        case 5:

            if agent.extra_variables[0]==1:
                a = Action(done = True, action=2,  x=agent.last_position[0], y=agent.last_position[1], explanation="")
                agent.action_queue.append(a)
            else:
                a = Action(done = True, action=0,  x=agent.last_position[0], y=agent.last_position[1], explanation="")
                agent.action_queue.append(a)
            

        # Spray Water
        case 6: 
            a = Action(done = True, action=3,  x=int(option.param_1), y=int(option.param_2), explanation="")
            agent.action_queue.append(a)


        # Refill Water
        case 7: 
            a = Action(done = True, action=4,  x=agent.last_position[0], y=agent.last_position[1], explanation="")
            agent.action_queue.append(a)
            

        
        # Become Idle
        case 0:
            a = Action(done=True, action= 0, x=agent.last_position[0], y=agent.last_position[1], explanation="")
            agent.action_queue.append(a)



def Run_Firefighter_Action(agent, action):
    """
    Convert high-level firefighter actions into low-level environment actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - extra_variables: List of state variables (extra_variables[0] is civilian carry state)
            - last_position: Tuple of (x, y) coordinates of last position
        action: Action object containing:
            - type: Type of action to perform:
                1: Move to location
                2: Cut tree
                3: Pick up civilian
                4: Drop off civilian
                5: Spray water
                6: Refill water
                0: Do nothing
            - param_1: First parameter (typically x-coordinate)
            - param_2: Second parameter (typically y-coordinate)
            - description: Human-readable description of the action
    
    Returns:
        list: Low-level action representation as [action_type, x, y] where action_type is:
            0: Move
            1: Cut tree
            2: Interact with civilian (pickup/dropoff)
            3: Spray water
            4: Refill water
            
    Raises:
        ValueError: If target location is out of map bounds
    """

    match action.type:

        # Move towards location
        case 1:
            if action.param_1 < 0 or action.param_1 > agent.cfg.envs.map_size or action.param_2 < 0 or action.param_2 > agent.cfg.envs.map_size:
                action.description += ": OUT OF BOUNDS"
                raise ValueError("out of bounds target")
            return [0, int(action.param_1), int(action.param_2)]

        # Cut 1 tree
        case 2:
            return [1, 0, 0]
        
        # Pick up civilian
        case 3:
            if agent.extra_variables[0]==0:
                return [2, agent.last_position[0], agent.last_position[1]]
            else:
                return [0, agent.last_position[0], agent.last_position[1]]

        # Drop off civilian
        case 4:
            if agent.extra_variables[0]==1:
                return [2, agent.last_position[0], agent.last_position[1]]
            else:
                return [0, agent.last_position[0], agent.last_position[1]]

        # Spray Water
        case 5: 
            return [3, int(action.param_1), int(action.param_2)]

        # Refill Water
        case 6: 
            return [4, agent.last_position[0], agent.last_position[1]]

        # Do Nothing
        case 0:
            return [0, agent.last_position[0], agent.last_position[1]]




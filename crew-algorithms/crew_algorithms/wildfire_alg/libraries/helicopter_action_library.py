def Run_Helicopter_Action(agent, action):
    """
    Convert high-level helicopter actions into low-level environment actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
        action: Action object containing:
            - type: Type of action to perform:
                1: Move to location
                2: Pick up firefighters
                3: Drop off firefighters
                4: Refill water
                5: Deploy water
                0: Do nothing (become idle)
            - param_1: First parameter (x-coordinate for movement)
            - param_2: Second parameter (y-coordinate for movement)
            - description: Human-readable description of the action
    
    Returns:
        list: Low-level action representation as [action_type, x, y] where action_type is:
            0: Move
            1: Pick up firefighters
            2: Refill water
            3: Drop off firefighters/Deploy water
            
    Raises:
        ValueError: If target location is out of map bounds
    """

    match action.type:

        # Move to location
        case 1:
            if action.param_1 < 0 or action.param_1 > agent.cfg.envs.map_size or action.param_2 < 0 or action.param_2 > agent.cfg.envs.map_size:
                action.description += ": OUT OF BOUNDS"
                raise ValueError("out of bounds target")

            return [0, int(action.param_1), int(action.param_2)]
        
        # Pick up firefighters
        case 2:
            return [1, 0, 0]
        
        # Drop off firefighters
        case 3:
            return [3, 0, 0]

        # Refill Water
        case 4:
            return [2, 0, 0]
        
        # Deploy Water
        case 5:
            return [3, 0, 0]
        
        # Become Idle
        case 0:
            return [0, agent.last_position[0], agent.last_position[1]]



def Run_Drone_Action(agent, action):
    """
    Convert high-level drone actions into low-level environment actions.
    
    Args:
        agent: The agent object containing:
            - cfg: Configuration object with environment settings
            - last_position: Tuple of (x, y) coordinates of last position
        action: Action object containing:
            - type: Type of action to perform:
                1: Move to location
                0: Do nothing
            - param_1: First parameter (x-coordinate)
            - param_2: Second parameter (y-coordinate)
            - description: Human-readable description of the action
    
    Returns:
        list: Low-level action representation as [action_type, x, y] where action_type is:
            0: Move
            
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
        
        # Do Nothing
        case 0:
            return [0, agent.last_position[0], agent.last_position[1]]



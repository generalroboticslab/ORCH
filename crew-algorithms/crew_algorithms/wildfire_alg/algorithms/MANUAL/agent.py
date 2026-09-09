from crew_algorithms.wildfire_alg.algorithms.MANUAL.__main__ import Config
from typing import List, Tuple, Dict

class Agent:
    def __init__(
        self,
        id: int,
        type: int,
        cfg: Config,
        path: str,
        current_task: str,
        api_key: str,
    ) -> None:
        self.id = id
        self.type = type
        self.cfg = cfg
        self.path = path
        self.current_task = current_task
        self.api_key = api_key
        # Chat history for each other agent and a GLOBAL channel
        

        self.past_actions: List[str] = []
        self.last_observation: str = None
        self.last_position: Tuple[int, int] = None
        self.last_current_cell: str = None
        self.last_perceptions: str = None
        self.map_range: int = 0
        self.extra_variables = []


                    
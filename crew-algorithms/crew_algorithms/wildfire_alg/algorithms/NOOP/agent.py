import os
from openai import OpenAI, BadRequestError
from crew_algorithms.wildfire_alg.algorithms.NOOP.__main__ import Config
from typing import List, Tuple, Dict

class Agent:
    def __init__(self, id:int, type:int, cfg:Config, path, current_task:str, api_key) -> None:
        self.id = id
        self.type = type
        self.cfg = cfg

        self.options = []
        self.action_queue = []
        self.past_options = []

        self.chat_history = {}  # list of past communications

        self.current_task = current_task
        self.last_observation = None
        self.last_position = None
        self.last_current_cell = None
        self.last_perception = None
        self.map_range = 0
        self.path = path
        self.model_client = None
        self.api_key = api_key
        self.extra_variables = []
        os.makedirs(os.path.join(self.path, f"Agent_{self.id}"), exist_ok=True)


import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.__main__ import Config
from typing import List, Tuple, Dict
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.agent import Agent
from crew_algorithms.wildfire_alg.algorithms.WILDFIRE.utils import (
    Options, OptionSequence,
    request_options, async_request_options,
    translate_options, async_translate_options
)
from .master_logger import get_master_logger
from .event_emitter import emit_event


class WorkerAgent(Agent):
    def __init__(self, id:int, type:int, name: str, cfg:Config, path, api_key) -> None:

        super().__init__(id=id, name=name, cfg=cfg, api_key=api_key, path=path, team_name="NULL", type=type)

        self.type = type

        self.last_observation = None
        self.last_position = None
        self.last_current_cell = None
        
        self.model_client = None
        self.api_key = api_key
        self.extra_variables = []

        self.children_count[int(type)]=1

        # 0: Firefighter
        # 1: Bulldozer
        # 2: Drone
        # 3: Helicopter
        # 4: Manager
        # 5: Default

        match(type):
            case 0:
                agent_path = "firefighter"
            case 1:
                agent_path = "bulldozer"
            case 2:
                agent_path = "drone"
            case 3:
                agent_path = "helicopter"

        self.memory_buffer = []
        self.proposed_action = None
        self.options = []
        self.past_options = []
        self.action_queue = []
        self.last_action = []
        self.idle_steps = 0

        self.map_range = 0 # Sets at every get_observation()


        # Conversation management
        self.status_messages = []  # Ongoing status phase conversation
        self.timestep_chats = []  # Current timestep's LLM responses for human observation

    def log_chat(self, chat_name: str, messages: List[Tuple[str, str]]):
        chat_string = f"{chat_name}\n" + "-"*20 + "\n\n"
        for source, content in messages:
            chat_string += f"{source}\n-----\n{content}\n-----\n\n"
        chat_string += "-"*20 + "\nEND CHAT\n\n"
        filepath = os.path.join(self.path, f"Agent_{self.id}","chats.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(chat_string)
        file.close()

    def add_message(self, source:str, content:str, time:int):
        self.chat_history.update({f'TIME {time}: {source}': content})

    def status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]], human_feedback: str = None):
        # DEPRECATED: USE ASYNC METHOD
        pass

    async def async_status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async version of status phase for worker agents
        """
        print(f"AGENT_{self.id}: Async Status Phase")
        emit_event(
            self._lobby_id,
            "status_started",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} collecting observations",
        )
        if not self.alive:
            self.status = 0
            self.options = []
            self.past_options = []
            self.action_queue = []
            self.percent_complete = 0
            self.status_messages = []
            self.timestep_chats = []
            self.perception_summary = "DESTROYED"
            self.status_summary = "DESTROYED"
            self.urgent = "DESTROYED"
            self.overview = "DESTROYED"
            return

        # Check if agent is in a helicopter - skip entire status phase if so
        if self.type == 0 and self.extra_variables[2] == 1:
            self.perception_summary = "in a helicopter"
            self.status_summary = "in a helicopter"
            self.urgent = ""
            self.percent_complete = 0.0
            self.timestep_chats = []
            self.status_messages = []
            return

        # Reset timestep chats at the beginning of each status phase
        self.timestep_chats = []

        # Start conversation with context if empty
        system_message = f"""You are AGENT_{self.id}, a {self._get_agent_type_string()} agent within a forest grid world.

Terminology:
- Mission: The overall goal for your team (e.g., "Suppress the fire in sector 7")
- Phase: A major step toward the mission (e.g., "Scout for the fire", "Build firebreaks") 
- Task: Your specific assignment for the current phase (e.g., "Go to coordinates (5,10) and scout for fire")
- Upper team: The team managed by your manager (your parent in the hierarchy)

Example: Your current task is "Go to coordinates (5,10) and scout for fire". Your team's phase is "Coordinate regional suppression". Your team's mission is "Suppress the fire in sector 7". 

Your task: {self.mission}
Your team's phase: {self.upperteam_phase}
Your team's mission: {self.upperteam_mission}

{self._build_announcement_block(global_data)}{f'Past Human Feedback: {self.past_feedback}' if self.past_feedback else ''}

Your job is to analyze observations and provide status updates with structured output using XML tags.
CRITICAL FORMATTING RULE: Every response MUST use the exact XML tags requested. Never omit or rename a tag. Always start your response with the opening tag."""
        
        self.status_messages = [{"role": "system", "content": system_message}]
        
        # Step 1: Generate perception (sequential within agent)
        if self.last_observation:
            user_message = f"""Here are your observations: {self._build_observation_string(agent_states, global_data)}

Create a concise detailed perception summary (<=50 words). The KEY FEATURES section is authoritative — use it as your primary source for fires, civilians, and water locations. You MUST respond using ONLY the following XML tags — no prose, no preamble, no text outside the tags:

<perception>
A detailed summary of what you observe, including your location, surroundings, any fires and civilians from the KEY FEATURES section. Note other agents in your vicinity, and your current status (carrying capacity, etc.). Focus on information relevant to your current task.
</perception>

DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.
Your response MUST begin with <perception> and end with </perception>. Do not write anything before <perception> or after </perception>.

Example response:
<perception>
I am currently at coordinates (5,10) in a medium forest cell. I can see dense forest to the north, a water source to the east at (7,10), and no signs of fire. There are no civilians nearby. I am not carrying any civilians and have 3/5 water remaining. I can see AGENT_2 at coordinates (6,9) to the northeast.
</perception>"""
            
            self.status_messages.append({"role": "user", "content": user_message})
            
            perception_response = await self._async_openai_call(self.status_messages, global_data)
            # Perception log excluded from chats.txt to reduce noise
            self.status_messages.append({"role": "assistant", "content": perception_response})
            
            # Store LLM response for human observation
            self.timestep_chats.append({
                "interaction": "Perception Summary",
                "response": perception_response
            })



            # Parse perception from tags
            perception = self._parse_tag_content(perception_response, "perception")
            self.perception_summary = perception
            emit_event(
                self._lobby_id,
                "perception_done",
                self.id,
                self.name,
                global_data.get("timestep", 0),
                f"{self.name} perception: {perception[:80]}",
                {"perception": perception},
            )

        # Drain slow feedback queue and persist in past_feedback
        slow_feedback_items = []
        if self._slow_status_queue is not None:
            while not self._slow_status_queue.empty():
                slow_feedback_items.append(self._slow_status_queue.get_nowait())

        if slow_feedback_items:
            for msg in slow_feedback_items:
                self.past_feedback.append("[OPERATOR] {}".format(msg))
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="SLOW_FEEDBACK_RECEIVED",
                    details={"messages": slow_feedback_items, "past_feedback_size": len(self.past_feedback)}
                )
            except Exception:
                pass

        # Step 2: Generate status summary with phase context (sequential, depends on step 1)
        idle_note = f"\nYou have been idle for {self.idle_steps} step(s)." if self.idle_steps > 0 else ""
        user_message = f"""Now, given your current task: '{self.mission}'
Your timeline: Past/Completed actions: {', '.join([opt.description for opt in self.past_options]) if self.past_options else 'None'}, Current/Ongoing actions: {self.options[0].description if self.options else 'None'}, Planned future actions: {', '.join([opt.description for opt in self.options[1:3]]) if len(self.options) > 1 else 'None'}{idle_note}

Provide your status summary using the following tags:

<summary>
A concise summary (<=30 words) of your progress and situation, including what you've accomplished and what you're currently doing.
</summary>
<percent_complete>
A number from 0 to 100 representing your estimated percent already complete on your current task ('{self.mission}'). 
</percent_complete>
<urgent>
Any important information for your manager, such as unexpected fires, civilians, or issues. If none, write "None".
</urgent>

Example response:
<summary>
I have moved to coordinates (5,10) and completed the scouting of the northern area. I found no fire in this sector.
</summary>
<percent_complete>
75
</percent_complete>
<urgent>
I have located the missing group of civilians at coordinates (5,10). I suggest we move to the next phase of transporting them.
</urgent>"""
        
        self.status_messages.append({"role": "user", "content": user_message})
        
        status_response = await self._async_openai_call(self.status_messages, global_data)
        # Status log excluded from chats.txt to reduce noise
        self.status_messages.append({"role": "assistant", "content": status_response})
        
        # Store LLM response for human observation
        self.timestep_chats.append({
            "interaction": "Status Summary",
            "response": status_response
        })
        
        
        # Parse status summary from tags
        status_summary = self._parse_tag_content(status_response, "summary")
        percent_complete = self._parse_tag_content(status_response, "percent_complete")
        urgent = self._parse_tag_content(status_response, "urgent")
        
        # Store parsed information
        self.status_summary = status_summary
        self.percent_complete = float(percent_complete) if percent_complete and percent_complete.isdigit() else 0.0
        self.urgent = urgent if urgent != "None" else ""

        emit_event(
            self._lobby_id,
            "options_written",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} status: {status_summary[:80]}",
            {"summary": status_summary, "percent_complete": self.percent_complete},
        )

        # Log status phase event
        try:
            logger = get_master_logger()
            options_list = [opt.description for opt in self.options] if self.options else []
            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="STATUS_PHASE",
                details={
                    "mission": self.mission,
                    "mission_percent": self.percent_complete,
                    "options_list": options_list,
                    "decision": "CONTINUE_TASK" if len(self.options) > 0 else "IDLE",
                    "urgent": self.urgent,
                    "past_feedback": self.past_feedback
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log worker status phase event: {e}")
        
        # Update overview and assess task status
        self._update_overview_with_status()
        
        if len(self.options) == 0:
            self.status = 0  # No more options, become idle
        else:
            self.status = 1  # Still working

    def _parse_tag_content(self, response: str, tag_name: str) -> str:
        """
        Parse content from a specific tag in the response
        """
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)

        response = response.replace("<perceptive>", "<perception>")
        response = response.replace("</perceptive>", "</perception>")
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            content_start = start_index + len(start_tag)
            content = response[content_start:end_index].strip()
            return content
        else:
            print(f"Warning: Could not find {tag_name} tags in response")
            return ""

    def _build_observation_string(self, agent_states: Dict[int, Tuple[int, int]], global_data: dict) -> str:
        """
        Build the observation string for the worker agent
        """
        # Build observation string verbatim (existing logic from worker_agent.py)
        if self.type == 0 and self.extra_variables[2] == 1:
            obs_string = f"""
            You are AGENT_{self.id} and you are within a helicopter. You are unable to perform actions. Your current location is {self.last_position}.
            """
        else:
            grid_features = self._parse_grid_features()
            obs_string = f"""
            You are AGENT_{self.id}, and your current location is {self.last_position} and thus your minimap view will be the range ({self.last_position[0]-self.map_range//2} to {self.last_position[0]+self.map_range//2}, {self.last_position[1]-self.map_range//2} to {self.last_position[1]+self.map_range//2}). The first column is x = {self.last_position[0]-self.map_range//2},  and the first row is  y = {self.last_position[1]-self.map_range//2}. Remember to take this into account in coordinate calculation.

            {grid_features}

            This is your minimap view, with commas separating cells and newlines separating rows:
            \n {self.last_observation} \n 

            Each cell is represented by a character corresponding to the type of terrain:
                0: brush (no trees)
                1: light forest (1 tree)
                2: medium forest (2 trees)
                3: dense forest (3 trees)
                i: Ignited
                f: On Fire
                e: Extinguishing
                x: Fully Extinguished
                w: Water Source Cell (no trees)
                B: building (no trees)
                
            IGNORE ALL "-". Those are unrevealed cells. They will reveal themselves when you get closer to them.

            The cells in single quotations are wet cells. 'C' cells are civilians.
            
            The bolded cell is the current cell you are in. It is a {self.last_current_cell} cell at {self.last_position}. There are other nearby agents at: 

            {self._get_nearby_agents_string(agent_states, global_data)}

            {self._get_extra_variables_string(global_data)}
            """
        
        return obs_string

    def _get_nearby_agents_string(self, agent_states: Dict[int, Tuple[int, int]], global_data: dict) -> str:
        """Get string of nearby agents that share the same parent/team"""
        others = ""
        for a, pos in agent_states.items():
            # Check if agent is nearby and has same parent
            if (a != self.id and 
                abs(pos[0]-self.last_position[0]) < self.map_range and 
                abs(pos[1]-self.last_position[1]) < self.map_range and
                global_data["agents"][a-1].parent == self.parent):
                    others += f"AGENT_{a}: {pos} "
        return others

    def _get_extra_variables_string(self, global_data: dict) -> str:
        """Get string of extra variables"""
        extra_string = ""
        if self.type == 0:
            if self.extra_variables[0] == 0:
                extra_string += "You are NOT carrying a civilian.\n"
            else:
                extra_string += "You are carrying a civilian.\n"
                
        if self.type == 3:
            if self.extra_variables[0] == 0:
                extra_string += "You are NOT carrying any firefighters.\n"
            else:
                extra_string += f"You are carrying {int(self.extra_variables[0])}/5 firefighters.\n"
                
        if "water" in global_data["game_data"]["task_description"]:
            extra_string += f"You currently have {int(self.extra_variables[1])}/5 water"
        
        return extra_string

    def _parse_grid_features(self) -> str:
        """Parse the raw observation grid and extract key features as structured text."""
        grid = self.last_observation
        if not grid:
            return ""

        pos = self.last_position
        half = self.map_range // 2
        x_start, y_start = pos[0] - half, pos[1] - half

        fires = {"i": [], "f": [], "e": []}
        civilians = []
        water_sources = []
        buildings = []
        extinguished = []

        rows = grid.strip().split("\n")
        for row_idx, row in enumerate(rows):
            cells = [c.strip().strip("*") for c in row.split(",") if c.strip()]
            for col_idx, cell in enumerate(cells):
                cx, cy = x_start + col_idx, y_start + row_idx
                if cell == "i":
                    fires["i"].append((cx, cy))
                elif cell == "f":
                    fires["f"].append((cx, cy))
                elif cell == "e":
                    fires["e"].append((cx, cy))
                elif cell == "x":
                    extinguished.append((cx, cy))
                elif cell in ("C", "'C'"):
                    civilians.append((cx, cy))
                elif cell in ("w", "'w'"):
                    water_sources.append((cx, cy))
                elif cell == "B":
                    buildings.append((cx, cy))

        lines = []

        for label, key in [("Ignited", "i"), ("On Fire", "f"), ("Extinguishing", "e")]:
            cells = fires[key]
            if not cells:
                continue
            if len(cells) <= 5:
                coords = ", ".join(f"({x},{y})" for x, y in cells)
                lines.append(f"{label} ({len(cells)}): {coords}")
            else:
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                lines.append(
                    f"{label} ({len(cells)}): "
                    f"region ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})"
                )

        if extinguished:
            if len(extinguished) <= 5:
                coords = ", ".join(f"({x},{y})" for x, y in extinguished)
                lines.append(f"Extinguished ({len(extinguished)}): {coords}")
            else:
                xs = [c[0] for c in extinguished]
                ys = [c[1] for c in extinguished]
                lines.append(
                    f"Extinguished ({len(extinguished)}): "
                    f"region ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})"
                )

        if civilians:
            coords = ", ".join(f"({x},{y})" for x, y in civilians)
            lines.append(f"Civilians ({len(civilians)}): {coords}")

        if water_sources:
            if len(water_sources) <= 5:
                coords = ", ".join(f"({x},{y})" for x, y in water_sources)
                lines.append(f"Water sources ({len(water_sources)}): {coords}")
            else:
                xs = [c[0] for c in water_sources]
                ys = [c[1] for c in water_sources]
                lines.append(
                    f"Water sources ({len(water_sources)}): "
                    f"region ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})"
                )

        if buildings:
            if len(buildings) <= 5:
                coords = ", ".join(f"({x},{y})" for x, y in buildings)
                lines.append(f"Buildings ({len(buildings)}): {coords}")
            else:
                xs = [c[0] for c in buildings]
                ys = [c[1] for c in buildings]
                lines.append(
                    f"Buildings ({len(buildings)}): "
                    f"region ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})"
                )

        if not lines:
            lines.append("No fires, civilians, water, or buildings in view.")

        return "KEY FEATURES IN VIEW:\n" + "\n".join(lines)

    def _update_overview_with_status(self):
        """Update agent overview with current status and parsed information"""

            
        self.overview = str(self.overview_template)
        self.overview = self.overview.replace("LOCATION", str(self.last_position))


    def action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        # DEPRECATED: USE ASYNC METHOD
        pass

    async def async_action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async version of action phase for worker agents
        """
        if not self.alive:
            return
        
        print(f"AGENT_{self.id}: Async Action Phase, Status: {self.status}")

        # Only execute if status is 0 (need new task) or 5 (new mission)
        if self.status == 5:
            emit_event(
                self._lobby_id,
                "action_started",
                self.id,
                self.name,
                global_data.get("timestep", 0),
                f"{self.name} generating action options",
            )
            self.options = []
            self.past_options = []
            self.action_queue = []
            self.last_action = []
            self.memory_buffer = []
            self.percent_complete = 0
            await self._async_generate_options(global_data)
            self.status = 1  # Now working on task
            emit_event(
                self._lobby_id,
                "options_written",
                self.id,
                self.name,
                global_data.get("timestep", 0),
                f"{self.name} options generated",
            )

            # Log action phase event
            try:
                logger = get_master_logger()
                options_list = [opt.description for opt in self.options] if self.options else []
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="ACTION_PHASE",
                    details={
                        "decision": "NEW_TASK",
                        "mission": self.mission,
                        "options_list": options_list,
                        "past_feedback": self.past_feedback
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to log worker action phase event: {e}")

    async def _async_generate_options(self, global_data: dict):
        """
        Async version of generate options using async OpenAI calls
        """
        # Step 1: Generate option sequence (now async)
        option_sequence: OptionSequence = await async_request_options(self, global_data)

        # Step 2: Translate options into structured format (now async)
        options: Options = await async_translate_options(self, option_sequence, global_data)

        # Step 3: Store the translated options
        self.options = options.actions
        self.past_options = []  # Reset past options for new task

        print(f"AGENT_{self.id}: Generated {len(self.options)} options")

        # Log the options for debugging
        for i, option in enumerate(self.options):
            print(f"  Option {i+1}: {option.description}")

    def _generate_options(self, global_data: dict):
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _update_memory(self, action_str: str):
        """Update memory buffer with action"""
        if not hasattr(self, 'memory_buffer'):
            self.memory_buffer = []
            
        self.memory_buffer.append(action_str)
        
        # Keep memory buffer size limited
        max_buffer_size = getattr(self.cfg.llms, 'memory_buffer_size', 10)  # Default to 10 if not specified
        if len(self.memory_buffer) > max_buffer_size:
            # Remove oldest entries to maintain buffer size
            self.memory_buffer = self.memory_buffer[-max_buffer_size:]

    def _get_agent_type_string(self) -> str:
        """Get agent type as string"""
        if self.type == 0:
            return 'Firefighter'
        elif self.type == 1:
            return 'Bulldozer'
        elif self.type == 2:
            return 'Drone'
        elif self.type == 3:
            return 'Helicopter'
        else:
            return 'Unknown'

    def _get_agent_path(self) -> str:
        """Get agent path for prompts"""
        if self.type == 0:
            return 'firefighter-data'
        elif self.type == 1:
            return 'bulldozer-data'
        elif self.type == 2:
            return 'drone-data'
        elif self.type == 3:
            return 'helicopter-data'
        else:
            return 'firefighter-data'  # Default




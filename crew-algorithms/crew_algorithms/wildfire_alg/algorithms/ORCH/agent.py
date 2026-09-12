import os
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
from crew_algorithms.wildfire_alg.algorithms.ORCH.__main__ import Config
from typing import List, Optional, Tuple, Dict, Literal
from .master_logger import get_master_logger
from .event_emitter import emit_event

# Model depth levels for controlling which model to use
# HIGH: For complex reasoning tasks like manager planning for a team
# LOW: For simpler tasks that need speed
ModelDepth = Literal["high", "low"]

def _get_llm_model(cfg) -> str:
    envs = getattr(cfg, "envs", cfg)
    return getattr(envs, "llm_model", "gpt")


def _get_llm_url(cfg):
    envs = getattr(cfg, "envs", cfg)
    return getattr(envs, "llm_url", None)


def _get_model_name(cfg, depth: ModelDepth = "low") -> str:
    model_map = {
        "gpt": {
            "high": "gpt-5.4",
            "low": "gpt-5.4-nano",
        },
        "qwen": {
            "high": "Qwen/Qwen3.6-35B-A3B",
            "low": "Qwen/Qwen3.6-35B-A3B",
        },
        "deepseek": {
            "high": "deepseek-ai/DeepSeek-V4-Pro",
            "low": "deepseek-ai/DeepSeek-V4-Pro",
        },
        "gemma": {
            "high": "google/gemma-4-31B-it",
            "low": "google/gemma-4-31B-it",
        },
        "glm": {
            "high": "zai-org/GLM-5.1-FP8",
            "low": "zai-org/GLM-5.1-FP8",
        },
        "llama": {
            "high": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "low": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        },
        "ernie" :{
            "high": 'baidu/ERNIE-4.5-21B-A3B-PT',
            "low": 'baidu/ERNIE-4.5-21B-A3B-PT',
        },
        "nemotron":{
            "high": 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
            "low": 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',
        },
        "minimax":{
            "high": 'MiniMaxAI/MiniMax-M3-MXFP8',
            "low": "MiniMaxAI/MiniMax-M3-MXFP8",
        },
        "kimi":{
            "high": 'moonshotai/Kimi-K2.6',
            "low": 'moonshotai/Kimi-K2.6',   
        }
    }
    llm_model = _get_llm_model(cfg)
    return model_map.get(llm_model, model_map["gpt"]).get(depth, model_map["gpt"]["low"])

class Agent:
    def __init__(self, id:int, name:str, cfg:"Config", path, api_key, team_name:str, type:int) -> None:
        self.id = id
        self.name = name
        self.path = path
        self.cfg = cfg
        self.api_key = api_key
        self.team_name = team_name
        self.type = type

        self.chat_history = {}  # list of past communications


        self.model_client = None
        self.async_client = None
        self.extra_variables = []

        self.children = []
        self.children_count = [0,0,0,0]
        self.parent = None

        # Enhanced status meanings:
        # 0: Idle (requesting new mission/task)
        # 1: Working on current task/phase (continue with current phase)
        # 2: Move to next phase
        # 3: Add phases
        # 4: Rewrite tasks in current phase
        # 5: New mission received (override other options)
        self.status = 0

        self.observations = ""
        self.overview = ""

        # New phase-based variables
        self.mission = "NO MISSION ASSIGNED"  # Overall goal
        self.current_phase = "NO PHASE ASSIGNED"  # Current phase within mission
        self.phase_history = []  # List of completed phases
        self.future_phases = []  # List of planned phases
        self.phase_progress = 100  # Percentage complete on current phase
        self.phase_completion_condition = ""  # Completion condition for current phase

        
        # Enhanced task tracking
        self.subteam_tasks = {}  # Tasks assigned to subteam members
        self.upperteam_phase = ""  # Current phase of upper team
        self.upperteam_mission = ""  # Mission of upper team
        
        # Conversation management
        self.status_messages = []  # Ongoing status phase conversation
        self.action_messages = []  # Ongoing action phase conversation
        self.timestep_chats = []  # Current timestep's LLM responses for human observation

        self.perception_summary = ""
        self.status_summary = ""
        self.percent_complete = 0.0 
        self.urgent = ""

        self.human = False
        self.past_feedback = []
        self.alive = True

        self.leader = False
        self.time = 0

        # Chat system queues (set from __main__.py before the game loop starts)
        self._slow_status_queue: Optional[asyncio.Queue] = None
        self._fast_feedback_queue: Optional[asyncio.Queue] = None
        self._pending_fast_feedback: list = []  # Consumed items, survives across phase checks
        self._did_replan: bool = False  # Set True when action phase executes a non-continue handler
        self._processed_fast_feedback: list = []  # Original directive texts consumed this timestep
        # Lobby ID for event emission (set from __main__.py)
        self._lobby_id: str = "default"

        match(type):
            case -1:
                agent_path = "manager"
            case 0:
                agent_path = "firefighter"
            case 1:
                agent_path = "bulldozer"
            case 2:
                agent_path = "drone"
            case 3:
                agent_path = "helicopter"

        # Load overview template with error handling
        try:
            overview_path = os.path.join("algorithms", "ORCH", "prompts", "overviews", f"{agent_path}_overview.txt")
            
            if not os.path.exists(overview_path):
                overview_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "overviews", f"{agent_path}_overview.txt")
            
            with open(overview_path, 'r') as file:
                self.overview_template = file.read()        
            file.close()
        except FileNotFoundError:
            print(f"Warning: Overview file not found for {agent_path}, using default template")
            self.overview_template = f"You are AGENT_{self.id} of {self.team_name}. You are a {agent_path} agent."

        self.overview_template=self.overview_template.replace("ID", str(self.id))
        self.overview_template=self.overview_template.replace("TEAMNAME", self.team_name)

        # Load child feedback template with error handling
        try:
            child_feedback_path = os.path.join("algorithms", "ORCH", "prompts", "child_feedback", f"{agent_path}_child_feedback.txt")
            
            if not os.path.exists(child_feedback_path):
                child_feedback_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "child_feedback", f"{agent_path}_child_feedback.txt")
            
            with open(child_feedback_path, 'r') as file:
                child_message = file.read()
            file.close()
        except FileNotFoundError:
            print(f"Warning: Child feedback file not found for {agent_path}, using default message")
            child_message = f"AGENT_{self.id} of {self.team_name} reporting status."
            
        self.child_message = child_message.replace("ID", str(self.id))
        self.child_message = child_message.replace("TEAMNAME", self.team_name)

        if type != -1:
            try:
                planner_path = os.path.join("algorithms", "WILDFIRE_MULTI_ALG", f"{agent_path}-data", "prompts", "manager_prompt.txt")
                
                if not os.path.exists(planner_path):
                    planner_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "WILDFIRE_MULTI_ALG", f"{agent_path}-data", "prompts", "manager_prompt.txt")
                
                with open(planner_path, 'r') as file:
                    self.planner = file.read()
            except FileNotFoundError:
                print(f"Warning: Planner file not found for {agent_path}, using default planner")
                self.planner = f"You are a {agent_path} agent. Plan your actions carefully."
        
        # Load knowledge base for managers
        if type == -1:  # Manager agent
            try:
                knowledge_base_path = os.path.join("algorithms", "ORCH", "prompts", "management", "knowledge_base.json")
                
                if not os.path.exists(knowledge_base_path):
                    knowledge_base_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "management", "knowledge_base.json")
                
                with open(knowledge_base_path, 'r') as file:
                    knowledge_base_data = json.load(file)
                file.close()
                
                # Get current level name from config
                current_level = self.cfg.envs.level if hasattr(self.cfg, 'envs') and hasattr(self.cfg.envs, 'level') else "Cut_Trees_Sparse_small"

                current_level = "general"

                
                # Format the knowledge base as a bulleted list string
                knowledge_parts = []
                
                # Add current level-specific knowledge
                if current_level in knowledge_base_data.get("knowledge_base", {}):
                    level_info = knowledge_base_data["knowledge_base"][current_level]
                    for info in level_info["information"]:
                        knowledge_parts.append(f"- {info}")
                
                self.knowledge_base = "\n".join(knowledge_parts)
            except FileNotFoundError:
                print(f"Warning: Knowledge base file not found, using empty knowledge base")
                self.knowledge_base = ""

        else:
            self.knowledge_base = ""
        
        print("Made Agent "+ str(id))

    def _build_announcement_block(self, global_data: dict) -> str:
        """Build a game event announcement block for LLM prompts."""
        current = global_data.get("active_announcements", [])
        history = global_data.get("announcement_history", [])
        now = global_data.get("timestep", 0)

        sections = []

        # Current-turn announcements — prominent header
        if current:
            items = "\n".join(f"- {a}" for a in current)
            sections.append(f">>> NEW GAME EVENT <<<\n{items}")

        # Past announcements (exclude any that are also current)
        current_set = set(current)
        past_lines = []
        for entry in history:
            if entry["text"] not in current_set:
                turns_ago = now - entry["timestep"]
                past_lines.append(f"- [{turns_ago} turn(s) ago] {entry['text']}")
        if past_lines:
            sections.append("Past Events:\n" + "\n".join(past_lines))

        if not sections:
            return ""
        return "\n\n".join(sections) + "\n\n"

    def _format_feedback(self) -> str:
        """Format past_feedback into distinct sections so the LLM can distinguish
        standing orders from the latest urgent directives."""
        if not self.past_feedback:
            return ""
        standing = [f for f in self.past_feedback if f.startswith("[STANDING ORDER]")]
        urgent = [f for f in self.past_feedback if f.startswith("[URGENT]")]
        operator = [f for f in self.past_feedback if f.startswith("[OPERATOR]")]

        sections = []
        if standing:
            items = "\n".join(f"- {s.replace('[STANDING ORDER] ', '')}" for s in standing)
            sections.append(f"STANDING ORDERS (previous directives, may be overridden by latest directives):\n{items}")
        if operator:
            items = "\n".join(f"- {o.replace('[OPERATOR] ', '')}" for o in operator)
            sections.append(f"OPERATOR GUIDANCE:\n{items}")
        if urgent:
            items = "\n".join(f"- {u.replace('[URGENT] ', '')}" for u in urgent)
            sections.append(f"LATEST DIRECTIVES (highest priority — override any conflicting standing orders):\n{items}")
        return "\n\n".join(sections)

    def log_chat(self, chat_name: str, messages: List[Tuple[str, str]]):
        # Include timestep in the header if provided
        header = chat_name
        chat_string = f"{header}: Time: {self.time}\n" + "="*50 + "\n\n"
        for source, content in messages:
            chat_string += f"{source}\n-----\n{content}\n-----\n\n"
        chat_string += "="*50 + "\nEND CHAT\n\n"
        filepath = os.path.join(self.path, f"Agent_{self.id}","chats.txt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(chat_string)
        file.close()

    def add_message(self, source:str, content:str, time:int):
        self.chat_history.update({f'TIME {time}: {source}': content})

    async def _async_openai_call(self, messages: List[dict], global_data: dict, depth: ModelDepth = "low") -> str:
        """Async wrapper for OpenAI API calls

        Args:
            messages: List of message dicts for the chat completion
            global_data: Global data dict for tracking API usage
            depth: Model depth level - "high" for complex reasoning (manager planning),
                   "low" for fast responses (default)
        """
        if self.async_client is None:
            if _get_llm_model(self.cfg) == "gpt":
                self.async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=3600,
                )
            
            else:
                print(self.api_key)
                self.async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=_get_llm_url(self.cfg),
                    timeout=3600,
                )

        model = _get_model_name(self.cfg, depth)
        if _get_llm_model(self.cfg) == "gpt":
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
            )  
        else:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

        global_data["api_calls"] += 1
        global_data["input_tokens"] += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens

        return response.choices[0].message.content

    async def async_status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async version of status phase with internal sequential steps, external parallelism
        """
        print(f"AGENT_{self.id}: Async Status Phase")
        emit_event(
            self._lobby_id,
            "status_started",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} collecting team observations",
        )

        # Reset timestep chats at the beginning of each status phase
        self.timestep_chats = []
        self._processed_fast_feedback = []
        self.status_messages = []

        system_message = f"""You are AGENT_{self.id}, the Team Manager of {self.team_name}, part of a broader team of embodied agents within a grid world. The map is made up of {self.cfg.envs.map_size} by {self.cfg.envs.map_size} cells/grids with coordinates in the range of [0 to {self.cfg.envs.map_size-1}, 0 to {self.cfg.envs.map_size-1}] with the top left corner of the map being (0,0).

        Terminology:
        - Mission: The overall goal for your team (e.g., "Locate and suppress the fire")
        - Phase: A major step toward the mission (e.g., "Scout for the fire", "Build firebreaks") 
        - Task: The specific assignment for each agent in your team for the current phase
        - Upper team: The team managed by your manager (your parent in the hierarchy)

        Your team's mission: {self.mission}
        Your team's current phase: {self.current_phase}
        Your team's phase progress: {self.phase_progress}%
        Your upper team's phase: {self.upperteam_phase}
        Your upper team's mission: {self.upperteam_mission}

        {self._build_announcement_block(global_data)}{self._format_feedback()}

        Knowledge Base (may or may not be relevant to the current mission/team):
        {self.knowledge_base}

        Your role is to:
        1. Collect and summarize observations from your subteam
        2. Assess progress on current phase and overall mission
        3. Make decisions about phase transitions and task adjustments

        Always respond using the required tag structure and example format.
        CRITICAL FORMATTING RULE: Every response MUST use the exact XML tags requested. Never omit or rename a tag. Always start your response with the opening tag."""

        self.status_messages.append({"role": "system", "content": system_message})
        
        # Update team composition
        self._update_team_composition()
        
        # Step 1: Generate team perception summary (sequential within agent)
        team_context = "Here are your team's observations by Agent: \n\n"
        for worker in self.children:
            if not getattr(worker, 'alive', True):
                continue
            team_context += f"{worker.name}: \n\n{str(worker.perception_summary)}\n\n---\n\n"
        
        user_message = f"""Given your team's observations, summarize your team's collective perception in <=50 words.

            {team_context}

        You MUST respond using ONLY the following XML tags — no prose, no preamble, no text outside the tags. Do not reference any agents by name, but rather refer to them collectively as "my team":

        <perception>
        A concise detailed summary of your team's collective observations. Include what your team has discovered, their overall positions, and any important findings. Do not refer to any agents by name.
        </perception>

        DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.
        Your response MUST begin with <perception> and end with </perception>. Do not write anything before <perception> or after </perception>.

        """
        
        self.status_messages.append({"role": "user", "content": user_message})

        perception_response = await self._async_openai_call(self.status_messages, global_data)
        self.status_messages.append({"role": "assistant", "content": perception_response})
        # Perception/status logs excluded from chats.txt to reduce noise
        
        # Store LLM response for human observation
        self.timestep_chats.append({
            "interaction": "Team Perception Summary",
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

        # Drain slow feedback queue and check fast feedback queue
        slow_feedback_items = []
        if self._slow_status_queue is not None:
            while not self._slow_status_queue.empty():
                slow_feedback_items.append(self._slow_status_queue.get_nowait())

        # Drain fast queue into _pending_fast_feedback (survives for re-run checks)
        if self._fast_feedback_queue is not None:
            while not self._fast_feedback_queue.empty():
                item = self._fast_feedback_queue.get_nowait()
                self._pending_fast_feedback.append(item)
                self._processed_fast_feedback.append(item)

        fast_feedback_items = list(self._pending_fast_feedback)
        is_fast = len(fast_feedback_items) > 0
        self._pending_fast_feedback = []  # Clear after consuming

        # Persist slow feedback in past_feedback so it survives across timesteps
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

        # Persist fast feedback in past_feedback so it survives across timesteps
        if is_fast:
            for msg in fast_feedback_items:
                self.past_feedback.append("[URGENT] {}".format(msg.upper()))
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="FAST_FEEDBACK_RECEIVED",
                    details={"messages": fast_feedback_items, "past_feedback": self.past_feedback}
                )
            except Exception:
                pass

        # Fast feedback gets its own URGENT OPERATOR DIRECTIVE interaction
        if is_fast:
            combined = "\n".join(f"- {m}" for m in fast_feedback_items)
            content = (
                f"URGENT NEW OPERATOR DIRECTIVE — this is the latest directive and overrides any conflicting previous standing orders:\n{combined}\n\n"
                "This comes directly from the human operator and MUST be treated as highest priority. "
                "You MUST factor this directive into your next decision. Ignoring a direct operator command is a serious failure."
            )
            ack = "Understood. I will factor this directive into my next decision."
            emit_event(
                self._lobby_id,
                "fast_feedback_injected",
                self.id,
                self.name,
                global_data.get("timestep", 0),
                f"URGENT: {self.name} processing operator directive",
                {"messages": fast_feedback_items},
            )
            self.status_messages.append({"role": "user", "content": content})
            self.status_messages.append({"role": "assistant", "content": ack})
            self.log_chat("URGENT Operator Directive", [
                ("user", content),
                ("assistant", ack)
            ])
            self.timestep_chats.append({
                "interaction": "URGENT Operator Directive",
                "response": ack
            })

        # Auto-handle no-mission and first-mission scenarios (skip LLM decision)
        if self.mission == "NO MISSION ASSIGNED":
            print(f"AGENT_{self.id}: No mission assigned, skipping decision (idle)")
            self.status_summary = "Awaiting mission assignment"
            self.percent_complete = 0.0
            self.phase_progress = 0.0
            self.status = 0
            return

        needs_initial_plan = (
            self.current_phase == "NO PHASE ASSIGNED"
            and not self.phase_history
        )
        if needs_initial_plan:
            print(f"AGENT_{self.id}: First mission received, auto-planning (skipping decision)")
            self.status_summary = f"Planning mission: {self.mission}"
            self.percent_complete = 0.0
            self.phase_progress = 0.0
            self.status = 5
            return

        # Step 2: Generate team status with phase decisions (sequential within agent, depends on step 1)
        # Create aligned subteam status and task information
        subteam_alignment = []

        # Alert for destroyed agents
        destroyed_agents = [a for a in self.children if not getattr(a, 'alive', True)]
        if destroyed_agents:
            destroyed_names = ", ".join(a.name for a in destroyed_agents)
            subteam_alignment.append(f"ALERT: The following agents have been DESTROYED and are no longer available: {destroyed_names}")
            subteam_alignment.append(f"  You must adapt your plan — these agents cannot execute any tasks.")
            subteam_alignment.append("")

        for agent in self.children:
            agent_name = agent.name
            if not getattr(agent, 'alive', True):
                subteam_alignment.append(f"Agent: {agent_name}")
                subteam_alignment.append(f"  *** DESTROYED — no longer available ***")
                subteam_alignment.append("")
                continue

            agent_task = getattr(agent, 'mission', 'No task assigned')
            agent_status = getattr(agent, 'status_summary', 'No status available')
            agent_percent_complete = getattr(agent, 'percent_complete', 'No percent complete available')
            agent_urgent = getattr(agent, 'urgent', 'None')

            subteam_alignment.append(f"Agent: {agent_name}")
            if hasattr(agent, 'team_name') and agent.team_name != "NULL":
                subteam_alignment.append(f"  Team: {agent.team_name}")
            subteam_alignment.append(f"  Task: {agent_task}")
            subteam_alignment.append(f"  Percent Complete with Task: {agent_percent_complete}")
            subteam_alignment.append(f"  Status: {agent_status}")
            subteam_alignment.append(f"  Urgent: {agent_urgent}")

            if agent.status == 0:
                subteam_alignment.append(f"  State: IDLE ON STANDBY (NO ACTIVE ACTIONS)")
            else:
                subteam_alignment.append(f"  State: EXECUTING ACTIONS")

            agent_past = getattr(agent, 'past_options', [])
            if agent_past:
                recent = ', '.join([opt.description for opt in agent_past[-3:]])
                subteam_alignment.append(f"  Recent Actions: {recent}")

            agent_idle = getattr(agent, 'idle_steps', 0)
            if agent_idle > 0:
                subteam_alignment.append(f"  Idle For: {agent_idle} step(s)")

            subteam_alignment.append("")

        subteam_info = "\n".join(subteam_alignment)
        
        # Decision prompt logic — build decision options conditionally
        if not self.future_phases:
            decision_options = f"""CONTINUE_PHASE, ADD_PHASES, REWRITE_TASKS, NEW_MISSION"""
            decision_descriptions = f"""CONTINUE_PHASE: If the team is on track to complete the current phase ('{self.current_phase}').
            ADD_PHASES: If the team has FULLY completed ALL planned phases ('{self.current_phase}'), but the mission ('{self.mission}') is not fully complete. The status of the current phase should be 100% complete. You may always add new phases later.
            REWRITE_TASKS: If the team needs new/reorganized tasks for the current phase ('{self.current_phase}') to be completed. Rewrite the tasks only if the current tasks are insufficient or the team is not progressing towards the completion of the phase. You may always rewrite tasks later.
            NEW_MISSION: If the team has FULLY completed the current mission ('{self.mission}'), regardless if all phases are necessarily complete. Request a new mission. You may always request a new mission later."""
            example_decision = "ADD_PHASES"
            example_summary = "The team has completed all planned phases but the fire is not fully contained. Additional work is needed."
            example_pct = "100"
        else:
            decision_options = f"""CONTINUE_PHASE, NEXT_PHASE, REWRITE_TASKS, ADD_PHASES, NEW_MISSION"""
            decision_descriptions = f"""CONTINUE_PHASE: If the team is on track to complete the current phase ('{self.current_phase}').
            NEXT_PHASE: If the team has FULLY completed the current phase and the next phase is needed ('{self.future_phases[0]}'). The phase completion condition should already be met. You may always move to the next phase later.
            REWRITE_TASKS: If the team needs new/reorganized tasks for the current phase ('{self.current_phase}') to be completed. Rewrite the tasks only if the current tasks are insufficient or the team is not progressing towards the completion of the phase. You may always rewrite tasks later.
            ADD_PHASES: If the team needs to perform a new phase immediately. Your current phase ('{self.current_phase}') will be moved until this phase is completed.
            NEW_MISSION: If the team has FULLY completed the current mission ('{self.mission}'), regardless if all phases are necessarily complete. Request a new mission. You may always request a new mission later.

            You MUST request a new mission if you have no mission assigned."""
            example_decision = "CONTINUE_PHASE"
            example_summary = "The team has scouted 80% of the area and found a small fire in the northern sector. All agents are progressing as planned."
            example_pct = "80"

        user_message = f"""Given your current mission: '{self.mission}'

            Your timeline: Past Phases: {self.phase_history}, Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

            Your current phase's completion condition: {self.phase_completion_condition}

            {self._build_announcement_block(global_data)}{self._format_feedback()}
            Your subteam's current status and tasks:
            {subteam_info}

            Provide:

            1. A status summary using the following tags:
            <summary>
            A concise summary of your team's progress and situation, including what has been accomplished and current status. Do not cite specific agents, only the team as a whole.
            </summary>
            <phase_percent_complete>
            A number from 0 to 100 representing your team's estimated percent complete on the current phase ('{self.current_phase}').
            </phase_percent_complete>
            <mission_percent_complete>
            A number from 0 to 100 representing your team's estimated percent complete on the current mission ('{self.mission}').
            </mission_percent_complete>
            <urgent>
            Any urgent information for your upper team, such as unexpected fires, civilians, or other issues. If none, write "None".
            </urgent>

            2. A decision using the following tag:
            <reasoning>
            Concise reasoning for your decision.
            </reasoning>
            <decision>
            One of: {decision_options}
            </decision>

            {decision_descriptions}

            Example response:
            <summary>
            {example_summary}
            </summary>
            <phase_percent_complete>
            {example_pct}
            </phase_percent_complete>
            <mission_percent_complete>
            20
            </mission_percent_complete>
            <urgent>
            None
            </urgent>
            <reasoning>
            The team has completed all planned phases but the fire is not fully contained. Additional work is needed.
            </reasoning>
            <decision>
            {example_decision}
            </decision>"""
        
        self.status_messages.append({"role": "user", "content": user_message})

        status_response = await self._async_openai_call(self.status_messages, global_data, depth="high")
        self.status_messages.append({"role": "assistant", "content": status_response})
        # Status log excluded from chats.txt to reduce noise

        # Store LLM response for human observation
        self.timestep_chats.append({
            "interaction": "Team Status Summary",
            "response": status_response
        })

        # Parse status summary and decision from tags
        status_summary = self._parse_tag_content(status_response, "summary")
        percent_complete = self._parse_tag_content(status_response, "mission_percent_complete")
        phase_complete = self._parse_tag_content(status_response, "phase_percent_complete")
        urgent = self._parse_tag_content(status_response, "urgent")
        decision = self._parse_tag_content(status_response, "decision")
        
        # Store parsed information
        self.status_summary = status_summary
        self.percent_complete = float(percent_complete) if percent_complete and percent_complete.isdigit() else 0.0
        self.phase_progress = float(phase_complete) if phase_complete and phase_complete.isdigit() else 0.0
        self.urgent = urgent if urgent != "None" else ""
        

        # If the decision is NEW_MISSION and this agent is the leader, reprompt for a new decision
        if decision == "NEW_MISSION" and getattr(self, "leader", False) and self.mission != "NO MISSION ASSIGNED":
            # The leader should not request a new mission unless the mission is fully complete
            # Reprompt for a new decision
            reprompt_message = (
                f"""The Mission '{self.mission}' is NOT fully complete. 
                Please reconsider and respond with an appropriate decision (e.g., CONTINUE_PHASE, NEXT_PHASE, etc.).
                
                
                <decision>
                new decision
                </decision>
                """
            )
            self.status_messages.append({"role": "user", "content": reprompt_message})
            status_response = await self._async_openai_call(self.status_messages, global_data, depth="high")
            self.status_messages.append({"role": "assistant", "content": status_response})
            # Reprompted status log excluded from chats.txt to reduce noise
            # Parse new decision
            decision = self._parse_tag_content(status_response, "decision")

        # Step 3: Confirm decision if it's not CONTINUE_PHASE (sequential, depends on step 2)
        if decision != "CONTINUE_PHASE":
            self.log_chat("Team Status Summary", [("user", user_message), ("assistant", status_response)])
            decision = await self._async_confirm_decision(decision, global_data)
        
        # Log status phase event
        try:
            logger = get_master_logger()
            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="STATUS_PHASE",
                details={
                    "mission": self.mission,
                    "mission_percent": self.percent_complete,
                    "phase": self.current_phase,
                    "phase_percent": self.phase_progress,
                    "decision": decision,
                    "urgent": self.urgent,
                    "past_feedback": self.past_feedback
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log status phase event: {e}")

        emit_event(
            self._lobby_id,
            "decision_made",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} decided: {decision}",
            {"decision": decision},
        )

        # Parse phase decisions and update status
        # past_feedback is NOT cleared here — it persists until popped after action phase commit
        if decision == "NEW_MISSION":
            self.status = 0
        elif decision == "CONTINUE_PHASE":
            self.status = 1
        elif decision == "NEXT_PHASE":
            self.status = 2
        elif decision in ("ADD_PHASES", "ADD_PHASE"):
            self.status = 3
        elif decision == "REWRITE_TASKS":
            self.status = 4
        else:
            self.status = 1

    def status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]], human_feedback: str = None):
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _parse_tag_content(self, response: str, tag_name: str) -> str:
        """
        Parse content from a specific tag in the response
        """
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        
        start_index = response.find(start_tag)
        end_index = response.find(end_tag)
        
        if tag_name == "decision":
            if end_index == -1:
               response += f"\n\n</{tag_name}>"  # Append closing tag if missing
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

    async def _async_confirm_decision(self, decision: str, global_data: dict) -> str:
        """
        Async version of confirm decision
        """
        warning_message = ""
        if decision == "NEXT_PHASE":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions for this phase."
        elif decision == "ADD_PHASES":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions for this phase."
        elif decision == "REWRITE_TASKS":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions for this phase."
        elif decision == "NEW_MISSION":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions and phases for this mission."
        
        confirmation_message = f"""
        {self._build_announcement_block(global_data)}{self._format_feedback()}
        {warning_message}

        IMPORTANT: Standing orders and human operator feedback ALWAYS take priority. You must NOT override or contradict human directives. If your decision conflicts with any standing order or operator feedback, you MUST CANCEL.

        Please confirm your decision using the following tag:
        <reasoning>
        Any reasoning for your decision.
        </reasoning>
        <confirmation>
        CONFIRM or CANCEL
        </confirmation>

        CONFIRM: Proceed with the decision to {decision}
        CANCEL: Continue with the current phase instead

        Example response:
        <reasoning>
        The team is nearly complete with the current mission, but a few agents need to finish their last actions.
        </reasoning>
        <confirmation>
        CANCEL
        </confirmation>"""

        self.status_messages.append({"role": "user", "content": confirmation_message})

        confirmation_response = await self._async_openai_call(self.status_messages, global_data)
        self.status_messages.append({"role": "assistant", "content": confirmation_response})
        
        # Parse confirmation
        confirmation = self._parse_tag_content(confirmation_response, "confirmation")
        
        self.log_chat("Decision Confirmation", [("user", confirmation_message), ("assistant", confirmation_response + f"{confirmation == 'CONFIRM'}")])


        if confirmation == "CONFIRM":
            print(f"AGENT_{self.id}: Confirmed decision to {decision}")
            return decision
        else:
            print(f"AGENT_{self.id}: Cancelled decision, continuing with current phase")
            return "CONTINUE_PHASE"
        

    def _confirm_decision(self, decision: str, global_data: dict) -> str:
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _update_team_composition(self):
        """Update team composition string and overview"""
        self.children_count = [0, 0, 0, 0]  # Reset before re-counting
        composition_string = ""
        worker_map = ["Firefighter", "Bulldozer", "Drone", "Helicopter"]

        direct_children = [0,0,0,0]
        for a in self.children:
            if not getattr(a, 'alive', True):
                continue
            self.children_count = [x + y for x, y in zip(self.children_count, a.children_count)]
            if hasattr(a, 'type') and a.type >= 0 and a.type < 4:
                direct_children[a.type] += 1
            else:
                composition_string += f"\n\n\n- {a.name} who manages {a.team_name} with:"
                subteam_children_count = a.children_count
                for c in range(len(subteam_children_count)):
                    if subteam_children_count[c] > 0:
                        composition_string += f"\n\t-{str(subteam_children_count[c])} {worker_map[c]} Agents"

        for c in range(len(direct_children)):
            if direct_children[c] > 0:
                composition_string += f"\n\n\n- {str(direct_children[c])} {worker_map[c]} Agents"

        for c in range(len(self.children_count)):
            if self.children_count[c] > 0:
                
                des_path = os.path.join('algorithms', 'ORCH', 'prompts', 'descriptions', f'{worker_map[c].lower()}_description.txt')
                if not os.path.exists(des_path):
                    des_path = os.path.join('crew_algorithms', "wildfire_alg", 'algorithms', 'ORCH', 'prompts', 'descriptions', f'{worker_map[c].lower()}_description.txt')
                
                with open(des_path, 'r') as file:
                    description = file.read()
                file.close()
                composition_string += f"\n\n{description}"


        self.overview = str(self.overview_template)

        self.overview = self.overview.replace("COMPOSITION", composition_string)

    def _drain_fast_feedback(self):
        """Drain fast feedback queue into _pending_fast_feedback and past_feedback."""
        if self._fast_feedback_queue and not self._fast_feedback_queue.empty():
            count = 0
            while not self._fast_feedback_queue.empty():
                item = self._fast_feedback_queue.get_nowait()
                self._pending_fast_feedback.append(item)
                self._processed_fast_feedback.append(item)
                self.past_feedback.append("[URGENT] {}".format(item.upper()))
                count += 1
            print(f"[FAST FEEDBACK] {self.name}: Drained {count} item(s) from fast queue")
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=0,
                    agent_id=str(self.id),
                    event_type="FAST_FEEDBACK_DRAINED",
                    details={"count": count, "past_feedback": self.past_feedback}
                )
            except Exception:
                pass
            return True
        return False

    async def _execute_action_handler(self, global_data: dict):
        """Execute the appropriate action handler based on self.status."""
        if self.status == 2:
            await self._async_handle_phase_transition(global_data)
        elif self.status == 3:
            await self._async_add_phases_to_mission(global_data)
            await self._async_handle_phase_transition(global_data)
        elif self.status == 4:
            await self._async_handle_task_rewrite(global_data)
        elif self.status == 5:
            await self._async_handle_new_mission(global_data)

    async def async_action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async action phase with checkpoint-based fast feedback handling.

        Before committing changes to children, checks the fast feedback queue.
        If new feedback arrived during the action handler, restores children state
        and re-runs status + action with the new feedback incorporated.
        """
        self._did_replan = False
        print(f"AGENT_{self.id}: Async Action Phase, Status: {self.status}")
        emit_event(
            self._lobby_id,
            "action_started",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} beginning action phase (status={self.status})",
        )

        self.log_chat("Confirming Decision", [("system", self.status)])
        self.action_messages = []

        # Step 1: Pre-handler drain (ALL statuses) — check if feedback arrived
        # between status phase and action phase
        if self._drain_fast_feedback():
            print(f"[CHECKPOINT] {self.name}: Feedback arrived before action (status={self.status}), re-running status")
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="CHECKPOINT_RERUN",
                    details={"trigger": "pre_handler", "status": self.status}
                )
            except Exception:
                pass
            saved_processed = list(self._processed_fast_feedback)
            await self.async_status_phase(global_data, agent_states)
            self._processed_fast_feedback = saved_processed + self._processed_fast_feedback

        # Step 2: CONTINUE/IDLE early exit (with conversion + report)
        if self.status in (0, 1):
            # Convert URGENT to standing orders even on CONTINUE path
            urgent_items = [f for f in self.past_feedback if f.startswith("[URGENT]")]
            if urgent_items:
                print(
                    f"[CHECKPOINT] {self.name}: Converting "
                    f"{len(urgent_items)} URGENT items to standing orders (CONTINUE path)"
                )
                self.past_feedback = [
                    f.replace("[URGENT] ", "[STANDING ORDER] ")
                    if f.startswith("[URGENT]")
                    else f
                    for f in self.past_feedback
                ]
                try:
                    logger = get_master_logger()
                    logger.log_event(
                        timestep=global_data.get("timestep", 0),
                        agent_id=str(self.id),
                        event_type="FEEDBACK_CONVERTED",
                        details={"converted_count": len(urgent_items), "past_feedback": self.past_feedback}
                    )
                except Exception:
                    pass
            # Generate report even on CONTINUE path
            if self._processed_fast_feedback:
                await self._generate_fast_feedback_report(global_data)
            for child in self.children:
                child.upperteam_phase = self.current_phase
                child.upperteam_mission = self.mission
            return

        # Step 3: Non-CONTINUE — CHECKPOINT: save children state before handler mutates them
        checkpoint = {c.id: (c.mission, c.status, c.percent_complete) for c in self.children}
        saved_manager = {
            'status': self.status,
            'current_phase': self.current_phase,
            'future_phases': list(self.future_phases) if self.future_phases else [],
            'phase_history': list(self.phase_history) if self.phase_history else [],
            'phase_progress': self.phase_progress,
            'phase_completion_condition': self.phase_completion_condition,
        }

        # Execute handler (mutates children)
        await self._execute_action_handler(global_data)
        self._did_replan = True

        # CHECK QUEUE before committing
        if self._drain_fast_feedback():
            print(f"[CHECKPOINT] {self.name}: Feedback arrived during action phase, restoring and re-running")
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="CHECKPOINT_RERUN",
                    details={"trigger": "action_phase", "status": self.status}
                )
            except Exception:
                pass
            self._did_replan = False  # Reset — final decision may differ
            saved_processed = list(self._processed_fast_feedback)
            # RESTORE checkpoint
            for child in self.children:
                if child.id in checkpoint:
                    child.mission, child.status, child.percent_complete = checkpoint[child.id]
            for k, v in saved_manager.items():
                setattr(self, k, v)

            # Re-run status (will see new URGENT items in past_feedback)
            await self.async_status_phase(global_data, agent_states)
            # Restore items wiped by status phase clear
            self._processed_fast_feedback = saved_processed + self._processed_fast_feedback

            # Re-execute if needed
            if self.status not in (0, 1):
                await self._execute_action_handler(global_data)
                self._did_replan = True

        # Convert URGENT to persistent standing orders instead of removing
        urgent_items = [f for f in self.past_feedback if f.startswith("[URGENT]")]
        if urgent_items:
            print(
                f"[CHECKPOINT] {self.name}: Converting "
                f"{len(urgent_items)} URGENT items to standing orders"
            )
            self.past_feedback = [
                f.replace("[URGENT] ", "[STANDING ORDER] ")
                if f.startswith("[URGENT]")
                else f
                for f in self.past_feedback
            ]
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="FEEDBACK_CONVERTED",
                    details={"converted_count": len(urgent_items), "past_feedback": self.past_feedback}
                )
            except Exception:
                pass

        # Generate fast feedback report using actual post-decision state
        if self._processed_fast_feedback:
            await self._generate_fast_feedback_report(global_data)

        # Propagate to children
        for child in self.children:
            child.upperteam_phase = self.current_phase
            child.upperteam_mission = self.mission

    async def _generate_fast_feedback_report(self, global_data: dict):
        """Generate a report on how the manager responded to fast feedback,
        using the actual post-decision state (not a stale snapshot)."""
        try:
            directives = "\n".join(f"- {d}" for d in self._processed_fast_feedback)

            children_context = ""
            if self.children:
                lines = []
                for c in self.children:
                    lines.append(
                        f"  - {c.name}: mission={getattr(c, 'mission', 'N/A')}, "
                        f"status={getattr(c, 'status_summary', 'N/A')}"
                    )
                children_context = "\n  Children assignments:\n" + "\n".join(lines)

            # Extract planning conversation excerpts so the review can see
            # what reasoning happened (not just the final state)
            planning_context = ""
            if self.action_messages:
                excerpts = []
                for msg in self.action_messages:
                    if msg.get("role") == "assistant":
                        text_excerpt = msg["content"][:500]
                        if len(msg["content"]) > 500:
                            text_excerpt += "..."
                        excerpts.append(text_excerpt)
                if excerpts:
                    # Last 2 assistant messages from planning
                    planning_context = (
                        "\n\nPlanning conversation excerpts:\n"
                        + "\n---\n".join(excerpts[-2:])
                    )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI manager in a wildfire simulation. You have received "
                        "an urgent human directive and are adjusting your plan. Write a first-person summary (2-3 sentences) "
                        "of what you plan to do in response. Use 'I' and 'my team'. "
                        "Describe the new plan and any changes you are making, referencing specifics from the planning conversation if available."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original directive(s):\n{directives}\n\n"
                        f"Manager's ACTUAL updated state after processing:\n"
                        f"  Status summary: {self.status_summary}\n"
                        f"  Current phase: {self.current_phase}\n"
                        f"  Phase progress: {self.phase_progress}%\n"
                        f"  Mission: {self.mission}"
                        f"{children_context}"
                        f"{planning_context}\n\n"
                        "Describe what you plan to do in response to the directive and what changes you are making to your team's approach, in first person."
                    ),
                },
            ]
            text = await self._async_openai_call(messages, global_data, depth="high")
            if not text:
                text = "I've received the directive and I'm adjusting my team's plan to address it."

            import tempfile as _tmp
            import datetime as _dt
            chat_out = os.path.join(
                _tmp.gettempdir(), f"chat_out_{self._lobby_id}.jsonl"
            )
            record = {
                "in_reply_to": f"fast_feedback_report_t{self.time}",
                "type": "fast_feedback_report",
                "content": text,
                "target_agent_id": self.id,
                "timestamp": _dt.datetime.now().isoformat(),
            }
            with open(chat_out, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.log_chat("Fast Feedback Report", [
                ("user", f"Directives: {directives}"),
                ("assistant", text)
            ])
            self.timestep_chats.append({
                "interaction": "Fast Feedback Report",
                "response": text
            })
            print(f"[FAST FEEDBACK REPORT] t={self.time} agent {self.id}: {text[:80]}")
            try:
                logger = get_master_logger()
                logger.log_event(
                    timestep=global_data.get("timestep", 0),
                    agent_id=str(self.id),
                    event_type="FAST_FEEDBACK_REPORT",
                    details={"directives": self._processed_fast_feedback, "report": text}
                )
            except Exception:
                pass
        except Exception as e:
            print(f"[FAST FEEDBACK REPORT] Failed at t={self.time} for agent {self.id}: {e}")

    def action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _handle_phase_transition(self, global_data: dict):
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _handle_task_rewrite(self, global_data: dict):
        # DEPRECATED: USE ASYNC METHOD
        pass

    def _handle_new_mission(self, global_data: dict):
        # DEPRECATED: USE ASYNC METHOD
        pass
    
    def _parse_phases_from_response(self, response: str) -> List[str]:
        """Parse phases from OpenAI response"""
        # Simple parsing - look for numbered or bulleted phases
        lines = response.split('\n')
        phases = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering/bullets and clean up
                phase = line.lstrip('0123456789.-* ').strip()
                if phase:
                    phases.append(phase)
        return phases

    def _assign_tasks_from_plan(self, plan: str, global_data: dict):
        """Assign tasks to team members from plan"""
        # Parse plan to extract agent-task pairs
        assigned_tasks = {}
        
        for agent in self.children:
            # Look for agent name in plan using tag structure
            start_tag = f"<{agent.name}>"
            end_tag = f"</{agent.name}>"
            
            start_index = plan.find(start_tag)
            end_index = plan.find(end_tag)
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                content_start = start_index + len(start_tag)
                task = plan[content_start:end_index].strip()
                assigned_tasks[agent.name] = task
            else:
                assigned_tasks[agent.name] = "Stand by and wait for further instructions"
                
        # Parse completion condition
        completion_condition = self._parse_tag_content(plan, "completion_condition")
        if completion_condition:
            self.phase_completion_condition = completion_condition
            print(f"Phase completion condition: {completion_condition}")
        
        # Assign tasks to agents (skip destroyed)
        for agent in self.children:
            if not getattr(agent, 'alive', True):
                continue
            task = assigned_tasks.get(agent.name, "Stand by and wait for further instructions")
            agent.mission = task
            agent.status = 5
            agent.percent_complete = 0
            print(f"Assigned task to {agent.name}: {task}")

    def _update_phase_and_assign_tasks(self, plan: str, global_data: dict):
        """Update to next phase and assign tasks"""
        # Move to next phase
        if self.future_phases:
            self.phase_history.append(self.current_phase)
            self.current_phase = self.future_phases.pop(0)
        
        # Assign tasks from plan
        self._assign_tasks_from_plan(plan, global_data)

    def _build_feedback_summary(self, feedback: Dict[str, str]) -> str:
        """Build summary of feedback for plan refinement"""
        summary = ""
        for agent_name, agent_feedback in feedback.items():
            if "YES" not in agent_feedback:
                summary += f"{agent_name}'s Feedback: {agent_feedback}\n\n"
        return summary

    def _collect_team_feedback(self, plan: str, global_data: dict) -> Dict[str, str]:
        # DEPRECATED: USE ASYNC METHOD
        pass

    async def _async_collect_team_feedback(self, plan: str, global_data: dict) -> Dict[str, str]:
        """
        Async version of team feedback collection - runs all child feedback in parallel
        """
        feedback_tasks = []
        
        for agent in [a for a in self.children if a.alive]:
            feedback_tasks.append(self._async_collect_child_feedback(agent, plan, global_data))
        
        # Run all feedback collection in parallel
        feedback_results = await asyncio.gather(*feedback_tasks)
        
        # Combine results into feedback dictionary
        feedback = {}
        for i, agent in enumerate([a for a in self.children if a.alive]):
            feedback[agent.name] = feedback_results[i]
        
        return feedback

    async def _async_collect_child_feedback(self, agent, plan: str, global_data: dict) -> str:
        """
        Collect feedback from one child asynchronously
        """
        # Load the appropriate child feedback prompt for this agent type
        if agent.type == -1:  # Manager agent
            agent_type = "manager"
        else:
            agent_type_map = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter"}
            agent_type = agent_type_map.get(agent.type, "firefighter")
        
        try:
            child_feedback_path = os.path.join("algorithms", "ORCH", "prompts", "child_feedback", f"{agent_type}_child_feedback.txt")
            
            if not os.path.exists(child_feedback_path):
                child_feedback_path = os.path.join("crew_algorithms", "wildfire_alg", "algorithms", "ORCH", "prompts", "child_feedback", f"{agent_type}_child_feedback.txt")
            
            with open(child_feedback_path, 'r') as file:
                child_system_prompt = file.read()
            file.close()
        except FileNotFoundError:
            print(f"Warning: Child feedback file not found for {agent_type}, using default")
            if agent_type == "manager":
                child_system_prompt = f"You are AGENT_{agent.id}, a team manager. Provide feedback on team plans."
            else:
                child_system_prompt = f"You are AGENT_{agent.id}, a {agent_type} agent. Provide feedback on team plans."
        
        # Replace placeholders in the child feedback prompt
        child_system_prompt = child_system_prompt.replace("AGENT_ID", f"AGENT_{agent.id}")
        child_system_prompt = child_system_prompt.replace("TEAMNAME", agent.team_name if hasattr(agent, 'team_name') and agent.team_name else "Unknown Team")
        
        # Handle location/position information
        if hasattr(agent, 'last_position') and agent.last_position:
            location_str = str(agent.last_position)
        elif hasattr(agent, 'children') and agent.children:
            # For managers, use team composition instead of single position
            location_str = f"managing team of {len(agent.children)} agents"
        else:
            location_str = "unknown"
        child_system_prompt = child_system_prompt.replace("LOCATION", location_str)
        
        # Handle observations
        if hasattr(agent, 'observations') and agent.observations:
            obs_str = agent.observations
        elif hasattr(agent, 'children') and agent.children:
            # For managers, collect team observations
            team_obs = []
            for child in agent.children:
                if hasattr(child, 'observations') and child.observations:
                    team_obs.append(f"{child.name}: {child.observations}")
            obs_str = "\n".join(team_obs) if team_obs else "No team observations available"
        else:
            obs_str = "No observations available"
        child_system_prompt = child_system_prompt.replace("OBS", obs_str)
        
        # Handle team composition for managers
        if agent_type == "manager" and hasattr(agent, 'children') and agent.children:
            composition = []
            for child in agent.children:
                child_type_map = {0: "Firefighter", 1: "Bulldozer", 2: "Drone", 3: "Helicopter", -1: "Manager"}
                child_type = child_type_map.get(child.type, "Unknown")
                composition.append(f"- {child.name} ({child_type})")
            composition_str = "\n".join(composition)
            child_system_prompt = child_system_prompt.replace("COMPOSITION", composition_str)
        else:
            child_system_prompt = child_system_prompt.replace("COMPOSITION", "No team members")
        
        # Build context line conditionally (horizontal managers have no phases)
        if self.current_phase is not None:
            context_line = f"The team's current phase is: {self.current_phase}, which is part of the broader plan: {self.mission}. This is the timeline of the plan: Past Phases: {self.phase_history}, New Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}"
            plan_label = f"specifically for the current phase ('{self.current_phase}')"
        else:
            context_line = f"The team's current mission is: {self.mission}"
            plan_label = "for the current mission"

        # Create feedback prompt for each agent
        if agent_type == "manager":
            feedback_message = f"""
            You are a member of a team structure {[i.name for i in self.children]}, including yourself: {agent.name}.
            {context_line}

            Your observations:
            {agent.perception_summary}

            Knowledge Base (may or may not be relevant to the current mission/team):
            {self.knowledge_base}

            Here is the proposed plan for the team by your Team Manager {plan_label}:

            {plan}

            As a subteam manager, examine your OWN subteam's designated role in the plan.
            Given your subteam's observations, status, and capabilities, determine if this plan is effective for your subteam and consistent with your subteam's capabilities.
            DO NOT be picky, just ensure that the plan works and aligns with your skills without obvious contingencies.

            If it satisfactory, respond with 'YES'
            If it instead requires adjustment for your subteam's role, give CONCISE feedback directed towards the Team Manager.
            """
        else:
            feedback_message = f"""
            You are a member of a team structure {[i.name for i in self.children]}, including yourself: {agent.name}.
            {context_line}

            Your observations:
            {agent.perception_summary}

            Knowledge Base (may or may not be relevant to the current mission/team):
            {self.knowledge_base}

            Here is the proposed plan for the team by your Team Manager {plan_label}:

            {plan}

            Specifically, examine your OWN designated role in the plan. Do not concern yourself with the role of your team members.
            Given your observations, status, and capabilities, determine if this plan is effective for you and consistent with your capabilities.

            If it satisfactory, respond with 'YES'
            If it instead requires adjustment for your role, give concise feedback directed towards the Team Manager.
            """
        
        messages = [
            {"role": "system", "content": child_system_prompt},
            {"role": "user", "content": feedback_message}
        ]
        
        agent_feedback = await self._async_openai_call(messages, global_data)
        self.log_chat(f"Agent_{agent.id} Feedback", [("user", feedback_message), ("assistant", agent_feedback)])
        
        return agent_feedback

    def _build_team_summary(self, detailed=False):
        """Build team summary. detailed=True adds sub-manager perception/status (for fresh conversations)."""
        worker_map = {0: "Firefighter", 1: "Bulldozer", 2: "Drone", 3: "Helicopter"}
        roster_lines = []
        seen_types = set()

        for agent in self.children:
            if not getattr(agent, 'alive', True):
                continue

            if agent.type == -1:  # Sub-manager
                comp_parts = []
                for idx, count in enumerate(getattr(agent, 'children_count', [])):
                    if count > 0:
                        comp_parts.append(f"{count} {worker_map.get(idx, 'Unknown')}s")
                        seen_types.add(worker_map.get(idx, 'Unknown'))
                comp_str = ", ".join(comp_parts) if comp_parts else "no agents"
                roster_lines.append(f"- {agent.name} manages {agent.team_name} ({comp_str})")

                if detailed:
                    perc = getattr(agent, 'perception_summary', None)
                    if perc:
                        roster_lines.append(f"    Perception: {perc}")
                    status = getattr(agent, 'status_summary', None)
                    if status:
                        pct = getattr(agent, 'percent_complete', None)
                        status_line = f"    Status: {status}"
                        if pct is not None:
                            status_line += f" ({pct}% complete)"
                        roster_lines.append(status_line)
            else:  # Worker
                agent_type = agent._get_agent_type_string() if hasattr(agent, '_get_agent_type_string') else f"Type {agent.type}"
                pos = getattr(agent, 'last_position', 'unknown')
                roster_lines.append(f"- {agent.name} ({agent_type}) at {pos}")
                seen_types.add(agent_type)

        # Type capabilities once per worker type
        type_descriptions = []
        for agent_type in seen_types:
            filepath = os.path.join('algorithms', 'ORCH', 'prompts', 'team_overviews', f'{agent_type.lower()}.txt')
            try:
                with open(filepath, 'r') as f:
                    desc = f.read().strip()
                type_descriptions.append(f"{agent_type}:\n{desc}")
            except FileNotFoundError:
                pass

        return "Agent Roster:\n" + "\n".join(roster_lines) + "\n\nAgent Type Capabilities:\n\n" + "\n\n".join(type_descriptions)

    def _get_agent_type_string(self) -> str:
        """Get agent type as string"""
        if self.type == -1:
            return 'Manager'
        elif self.type == 0:
            return 'Firefighter'
        elif self.type == 1:
            return 'Bulldozer'
        elif self.type == 2:
            return 'Drone'
        elif self.type == 3:
            return 'Helicopter'
        else:
            return 'Unknown'

    def _add_phases_to_mission(self, global_data: dict):
        # DEPRECATED: USE ASYNC METHOD
        pass

    async def _async_add_phases_to_mission(self, global_data: dict):
        """Async version of add phases to mission"""
        self.phase_progress=0
        team_summary = self._build_team_summary()

        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        user_message = f"""
        Now, generate additional phase(s) to continue the mission. Phases should be concise, simple, and built directly on the exact abilities of the team.

        Your team composition and team abilities:
        {team_summary}
        {feedback_section}
        List new phases using the following tags:
        <phases>
        1. ...
        2. ...
        </phases>
        Only include necessary phases. DO NOT include other lines/info in between, it should be exactly n lines.
        """
        self.action_messages.append({"role": "user", "content": user_message})

        phases_response = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Adding Phases", [("user", user_message), ("assistant", phases_response)])
        self.action_messages.append({"role": "assistant", "content": phases_response})

        self.timestep_chats.append({
            "interaction": "Adding Phases",
            "response": phases_response
        })
        
        phases_content = self._parse_tag_content(phases_response, "phases")
        new_phases = self._parse_phases_from_response(phases_content)
        self.future_phases.extend(new_phases)
        
        # Log add phases event
        try:
            logger = get_master_logger()
            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="ACTION_PHASE",
                details={
                    "decision": "ADD_PHASES",
                    "mission": self.mission,
                    "phases": [self.current_phase] + self.future_phases,
                    "new_phases_added": new_phases,
                    "past_feedback": self.past_feedback
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log add phases event: {e}")

    async def _async_handle_phase_transition(self, global_data: dict):
        """Async version of phase transition"""
        # Continue from status phase conversation
        self.phase_history.append(self.current_phase)
        try:
            self.current_phase = self.future_phases.pop(0)
        except IndexError:
            print(f"Warning: No future phases available for transition at agent {self.id}. Keeping current phase.")
            
        self.phase_progress=0
        self.phase_completion_condition = ""  # Reset completion condition for new phase

        team_summary = self._build_team_summary()

        agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        announcement_section = self._build_announcement_block(global_data)

        user_message = f"""Great, now let's move to the next phase.

        Timeline: Past Phases: {self.phase_history}, New Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}
        {feedback_section}
        {announcement_section}
        Your team composition and team abilities:
        {team_summary}

        Break down the next phase into tasks for each team member and define a completion condition using the following tags:

        {agent_tags}

        <completion_condition>
        A short, specific condition that defines when this phase is complete. Be concrete and measurable.
        </completion_condition>

        Keep each task concise — one or two short sentences. Be specific with coordinates but skip unnecessary detail, such as communication, staying alert, reporting observations, etc. However, the completion of all tasks SHOULD result in the full completion of the phase, so make them complete.

        Example response:
        <AGENT_1>
        Scout northeast region around (45,12) for fire outbreaks.
        </AGENT_1>
        <AGENT_2>
        Extinguish fire at southern border along y=400.
        </AGENT_2>
        <AGENT_3>
        Cut all trees at (30,15), (31,15), (32,15).
        </AGENT_3>
        <completion_condition>
        Northeast scouted, southern fire extinguished, trees at specified coordinates cut.
        </completion_condition>

        ONLY ASSIGN ONE TASK PER AGENT.
        """

        self.action_messages = self.status_messages.copy()  # Continue conversation
        self.action_messages.append({"role": "user", "content": user_message})

        plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Plan", [("user", user_message), ("assistant", plan)])
        self.action_messages.append({"role": "assistant", "content": plan})

        self.timestep_chats.append({
            "interaction": "Next Phase Plan Draft",
            "response": plan
        })
        
        # Collect feedback and refine
        
        # Refine plan based on feedback
        consensus = False
        max_iterations = 0
        iteration = 0

        while not consensus and iteration < max_iterations:
            feedback = await self._async_collect_team_feedback(plan, global_data)

            if feedback and not all("YES" in f for f in feedback.values()):
                user_message = f"""Here is feedback from your team:
        {self._build_feedback_summary(feedback)}

        Revise your plan accordingly using the same tag structure."""
                
                self.action_messages.append({"role": "user", "content": user_message})

                plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
                self.log_chat("Revise Plan", [("user", user_message), ("assistant", plan)])
                self.action_messages.append({"role": "assistant", "content": plan})

                feedback = await self._async_collect_team_feedback(plan, global_data)
            else:
                consensus = True
            iteration += 1

        self.timestep_chats.append({
            "interaction": "Final Plan",
            "response": plan
        })

        # Update phase and assign tasks
        self._assign_tasks_from_plan(plan, global_data)
        self.status = 1

        emit_event(
            self._lobby_id,
            "plan_written",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} wrote task assignments for phase transition",
            {"phase": self.current_phase},
        )
        emit_event(
            self._lobby_id,
            "phase_transition",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} transitioned to phase: {self.current_phase}",
            {"current_phase": self.current_phase, "future_phases": self.future_phases},
        )

        # Log action phase event
        try:
            logger = get_master_logger()
            task_designations = {}
            for agent in self.children:
                task_designations[agent.name] = getattr(agent, 'mission', 'No task assigned')

            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="ACTION_PHASE",
                details={
                    "decision": "PHASE_TRANSITION",
                    "mission": self.mission,
                    "phases": [self.current_phase] + self.future_phases,
                    "task_designations": task_designations,
                    "past_feedback": self.past_feedback
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log action phase event: {e}")

    async def _async_handle_task_rewrite(self, global_data: dict):
        """Async version of task rewrite"""
        self.phase_progress = 0

        team_summary = self._build_team_summary()

        agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        announcement_section = self._build_announcement_block(global_data)

        user_message = f"""Great, now rewrite tasks for the current phase: '{self.current_phase}'
        Timeline: Past Phases: {self.phase_history}, Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}
        {feedback_section}
        {announcement_section}
        Your team composition and team abilities:
        {team_summary}


        Explain what changed and why a new plan for this phase is necessary, then break down the current phase into new tasks and update the completion condition using the following tags:

        <explanation>
        Explain what changed and why a new plan for this phase is necessary.
        </explanation>
        {agent_tags}
        <completion_condition>
        A short, specific condition that defines when the current phase is complete. Be concrete and measurable.
        </completion_condition>

        Keep each task concise — one or two short sentences. Be specific with coordinates but skip unnecessary detail, such as communication, staying alert, reporting observations, etc. However, the completion of all tasks SHOULD result in the full completion of the phase, so make them complete.

        Example response:
        <explanation>
        Fire spotted in northeast. Reassigning tasks to respond.
        </explanation>

        <AGENT_1>
        Scout northeast region around (45,12) for fire outbreaks.
        </AGENT_1>
        <AGENT_2>
        Extinguish fire at southern border along y=400.
        </AGENT_2>
        <AGENT_3>
        Cut all trees at (30,15), (31,15), (32,15).
        </AGENT_3>
        <completion_condition>
        Northeast scouted, southern fire extinguished.
        </completion_condition>

        ONLY ASSIGN ONE TASK PER AGENT.
        """
        
        self.action_messages = self.status_messages.copy()  # Continue conversation
        self.action_messages.append({"role": "user", "content": user_message})

        plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Replan Phase", [("user", user_message), ("assistant", plan)])
        self.action_messages.append({"role": "assistant", "content": plan})

        self.timestep_chats.append({
            "interaction": "Replan Phase",
            "response": plan
        })
        
        # Collect feedback and refine
        
        # Refine plan based on feedback
        consensus = False
        max_iterations = 0
        iteration = 0
        
        while not consensus and iteration < max_iterations:
            feedback = await self._async_collect_team_feedback(plan, global_data)

            if feedback and not all("YES" in f for f in feedback.values()):
                user_message = f"""Here is feedback from your team:
        {self._build_feedback_summary(feedback)}

        Revise your plan accordingly using the same tag structure."""
                
                self.action_messages.append({"role": "user", "content": user_message})

                plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
                self.log_chat("Revise Plan", [("user", user_message), ("assistant", plan)])
                self.action_messages.append({"role": "assistant", "content": plan})

                feedback = await self._async_collect_team_feedback(plan, global_data)
            else:
                consensus = True
            iteration += 1

        # Assign new tasks
        self._assign_tasks_from_plan(plan, global_data)
        self.status = 1

        # Log task rewrite event
        try:
            logger = get_master_logger()
            task_designations = {}
            for agent in self.children:
                task_designations[agent.name] = getattr(agent, 'mission', 'No task assigned')
            
            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="ACTION_PHASE",
                details={
                    "decision": "REWRITE_TASKS",
                    "mission": self.mission,
                    "phases": [self.current_phase] + self.future_phases,
                    "task_designations": task_designations,
                    "past_feedback": self.past_feedback
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log task rewrite event: {e}")

    async def _async_handle_new_mission(self, global_data: dict):
        """Async version of new mission handling"""
        self.percent_complete=0
        self.phase_progress = 0
        self.phase_history = []
        self.future_phases = []
        self.current_phase = None
        self.phase_completion_condition = ""  # Reset completion condition for new mission

        team_summary = self._build_team_summary(detailed=True)

        system_message = f"""You are AGENT_{self.id}, the Team Manager of {self.team_name}: {[a.name for a in self.children if getattr(a, 'alive', True)]}, part of a broader team of embodied agents within a grid world. The map is made up of {self.cfg.envs.map_size} by {self.cfg.envs.map_size} cells/grids with coordinates in the range of [0 to {self.cfg.envs.map_size-1}, 0 to {self.cfg.envs.map_size-1}] with the top left corner of the map being (0,0).

        Your team composition and team abilities:
        {team_summary}

        Knowledge Base (may or may not be relevant to the current mission/team):
        {self.knowledge_base}

        Terminology:
        - Mission: The overall goal for your team (e.g., "Locate and suppress the fire")
        - Phase: A major step toward the mission (e.g., "Scout for the fire", "Build firebreaks around the fire") 
        - Task: The specific assignment for each agent in your team for the current phase
        - Upper team: The team managed by your manager (your parent in the hierarchy)

        Your job is to break down new missions into phases and tasks.
        First break the mission into phases, then break the first phase into tasks."""
                    
        self.action_messages = [{"role": "system", "content": system_message}]
                
        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        user_message = f"""Your mission: '{self.mission}'
        Upper team phase: {self.upperteam_phase}
        Upper team mission: {self.upperteam_mission}
        {feedback_section}
        First, break your mission into phases using the following tags:

        <phases>
        List each phase on a separate line, numbered or with bullet points.
        </phases>

        Example response:
        <phases>
        1. Scout for the fire
        2. Move to the fire and suppress it
        3. Check if the fire is fully contained
        </phases>

        Phases should be concise, simple, and built directly on the exact abilities of the team.
        You may only need one/a few phases. Do not include unnecessary phases, or phases that are outside of agent capabilities. However, look at specifically how many agents you have and what needs to be done for the mission to be fully complete. You may have to repeat groups of phases to complete the whole mission.
        DO NOT BREAK DOWN ANY PHASES INTO TASKS YET. KEEP THEM CONCISE.
        """
        
        self.action_messages.append({"role": "user", "content": user_message})

        phases_response = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Creating Phases", [("system", system_message), ("user", user_message), ("assistant", phases_response)])
        self.action_messages.append({"role": "assistant", "content": phases_response})

        self.timestep_chats.append({
            "interaction": "Creating Phases",
            "response": phases_response
        })
        
        # Parse phases and set current phase
        phases_content = self._parse_tag_content(phases_response, "phases")
        self.future_phases = self._parse_phases_from_response(phases_content)
        if self.future_phases:
            self.current_phase = self.future_phases.pop(0)
        
        agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

        # Now break down first phase into tasks
        user_message = f"""Now break down the first phase: '{self.current_phase}' into tasks for each team member and define a completion condition using the following tags:

        {agent_tags}
        <completion_condition>
        A short, specific condition that defines when this first phase is complete. Be concrete and measurable.
        </completion_condition>



        Consider each agent's capabilities, current position, and status when assigning tasks.

        Example response:
        <AGENT_1>
        Cut all trees at (30,15), (31,15), (32,15).
        </AGENT_1>
        <AGENT_2>
        Scout northeast region around (45,12) for fire outbreaks.
        </AGENT_2>
        <AGENT_3>
        Extinguish fire at southern border along y=400.
        </AGENT_3>
        <completion_condition>
        Trees cut at specified coordinates, northeast scouted.
        </completion_condition>

        Keep each task concise — one or two short sentences. Be specific with coordinates but skip unnecessary detail, such as communication, staying alert, reporting observations, etc. However, the completion of all tasks SHOULD result in the full completion of the phase, so make them complete.

        ONLY ASSIGN ONE TASK PER AGENT.
        """

        self.action_messages.append({"role": "user", "content": user_message})

        plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Creating Tasks", [("user", user_message), ("assistant", plan)])
        self.action_messages.append({"role": "assistant", "content": plan})
        
        self.timestep_chats.append({
            "interaction": "Creating Tasks",
            "response": plan
        })
        # Collect feedback and refine

        
        # Refine plan based on feedback
        consensus = False
        max_iterations = 0
        iteration = 0
        
        while not consensus and iteration < max_iterations:
            feedback = await self._async_collect_team_feedback(plan, global_data)

            if feedback and not all("YES" in f for f in feedback.values()):
                user_message = f"""Here is feedback from your team:
                    {self._build_feedback_summary(feedback)}
                    Revise your plan accordingly using the same tag structure."""
                
                self.action_messages.append({"role": "user", "content": user_message})

                plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
                self.log_chat("Revise Tasks", [("user", user_message), ("assistant", plan)])
                self.action_messages.append({"role": "assistant", "content": plan})
            else:
                consensus = True
            iteration += 1

        # Assign tasks and update status
        self._assign_tasks_from_plan(plan, global_data)
        self.status = 1

        emit_event(
            self._lobby_id,
            "plan_written",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} wrote initial task assignments for new mission",
            {"mission": self.mission, "phase": self.current_phase},
        )

        # Log new mission event
        try:
            logger = get_master_logger()
            task_designations = {}
            for agent in self.children:
                task_designations[agent.name] = getattr(agent, 'mission', 'No task assigned')

            logger.log_event(
                timestep=global_data.get("timestep", 0),
                agent_id=str(self.id),
                event_type="ACTION_PHASE",
                details={
                    "decision": "NEW_MISSION",
                    "mission": self.mission,
                    "phases": [self.current_phase] + self.future_phases,
                    "task_designations": task_designations,
                    "past_feedback": self.past_feedback
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log new mission event: {e}")

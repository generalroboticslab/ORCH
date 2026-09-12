import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from crew_algorithms.wildfire_alg.algorithms.ORCH.__main__ import Config
from typing import List, Tuple, Dict
from crew_algorithms.wildfire_alg.algorithms.ORCH.agent import Agent
from .master_logger import get_master_logger
from .event_emitter import emit_event


class HorizontalManagerAgent(Agent):
    """
    Horizontal Manager Agent - manages a team without phase-based planning.

    Unlike the regular Manager Agent which breaks missions into temporal phases,
    the Horizontal Manager only distributes tasks across its subteam in parallel.
    It has no concept of phases - only missions and tasks.
    """

    def __init__(self, id: int, name: str, cfg: Config, path, api_key, team_name: str) -> None:
        super().__init__(id=id, name=name, cfg=cfg, path=path, api_key=api_key, team_name=team_name, type=-1)

        # Override phase-related variables to disable phase logic
        self.current_phase = None
        self.phase_history = []
        self.future_phases = []
        self.phase_progress = 0
        self.phase_completion_condition = ""

        # Task assignment history tracking
        self.task_history = []  # List of task assignment records
        # Each record: {"timestep": int, "mission": str, "reasoning": str, "tasks": {agent_name: task_str}}

        print(f"Made Horizontal Manager Agent {id}")

    async def async_status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async status phase without phase logic - only tracks mission progress
        """
        print(f"AGENT_{self.id}: Horizontal Manager Async Status Phase")
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
        - Task: The specific assignment for each agent in your team
        - Upper team: The team managed by your manager (your parent in the hierarchy)

        Your team's mission: {self.mission}
        Your upper team's mission: {self.upperteam_mission}

        {self._build_announcement_block(global_data)}{self._format_feedback()}

        Knowledge Base (may or may not be relevant to the current mission/team):
        {self.knowledge_base}

        Your role is to:
        1. Collect and summarize observations from your subteam
        2. Assess progress on the current mission
        3. Make decisions about task adjustments

        Always respond using the required tag structure and example format.
        CRITICAL FORMATTING RULE: Every response MUST use the exact XML tags requested. Never omit or rename a tag. Always start your response with the opening tag."""

        self.status_messages.append({"role": "system", "content": system_message})

        # Update team composition
        self._update_team_composition()

        # Step 1: Generate team perception summary
        team_context = "Here are your team's observations by Agent: \n\n"
        for worker in self.children:
            if not getattr(worker, 'alive', True):
                continue
            team_context += f"{worker.name}: \n\n{str(worker.perception_summary)}\n\n---\n\n"

        user_message = f"""Given your team's observations, summarize your team's collective perception in <=50 words.

            {team_context}

        You MUST respond using ONLY the following XML tags — no prose, no preamble, no text outside the tags. Do not reference any agents by name, but rather refer to them collectively as "my team":

        <perception>
        A detailed summary of your team's collective observations. Include what your team has discovered, their overall positions, and any important findings. Do not refer to any agents by name.
        </perception>

        DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.
        Your response MUST begin with <perception> and end with </perception>. Do not write anything before <perception> or after </perception>.

        """

        self.status_messages.append({"role": "user", "content": user_message})

        perception_response = await self._async_openai_call(self.status_messages, global_data)
        self.status_messages.append({"role": "assistant", "content": perception_response})
        # Perception log excluded from chats.txt to reduce noise

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
        # Horizontal managers have no phases, so fast feedback options are restricted
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
            print(f"[FAST FEEDBACK] {self.name}: Injecting {len(fast_feedback_items)} urgent message(s) into status phase")

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
            self.status = 0
            return

        if not self.task_history:
            print(f"AGENT_{self.id}: First mission received, auto-planning (skipping decision)")
            self.status_summary = f"Planning mission: {self.mission}"
            self.percent_complete = 0.0
            self.status = 5
            return

        # Step 2: Generate team status with task decisions (no phase logic)
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

        # Decision prompt without phase logic
        user_message = f"""Given your current mission: '{self.mission}'

            {self._build_announcement_block(global_data)}{self._format_feedback()}
            Your subteam's current status and tasks:
            {subteam_info}

            Provide:

            1. A status summary using the following tags:
            <summary>
            A concise summary of your team's progress and situation, including what has been accomplished and current status. Do not cite specific agents, only the team as a whole.
            </summary>
            <mission_percent_complete>
            A number from 0 to 100 representing your team's estimated percent complete on the current mission ('{self.mission}').
            </mission_percent_complete>
            <urgent>
            Any urgent information for your upper team, such as unexpected fires, civilians, or other issues. If none, write "None".
            </urgent>

            2. A decision using the following tag:
            <reasoning>
            Any reasoning for your decision.
            </reasoning>
            <decision>
            One of: CONTINUE_MISSION, REWRITE_TASKS, NEW_MISSION
            </decision>

            CONTINUE_MISSION: If the team is on track to complete the current mission ('{self.mission}').
            REWRITE_TASKS: If the team needs new/reorganized tasks for the current phase ('{self.current_phase}') to be completed. Rewrite the tasks only if the current tasks are insufficient or the team is not progressing towards the completion of the phase. You may always rewrite tasks later.
            NEW_MISSION: If the team has FULLY completed the current mission ('{self.mission}'). Request a new mission. You may always request a new mission later.

            Example response:
            <summary>
            The team has covered 80% of the search area and found a small fire in the northern sector. All agents are progressing as planned.
            </summary>
            <mission_percent_complete>
            80
            </mission_percent_complete>
            <urgent>
            None
            </urgent>
            <reasoning>
            The team is making steady progress toward the mission goal.
            </reasoning>
            <decision>
            CONTINUE_MISSION
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
        urgent = self._parse_tag_content(status_response, "urgent")
        decision = self._parse_tag_content(status_response, "decision")

        # Store parsed information
        self.status_summary = status_summary
        try:
            self.percent_complete = float(percent_complete) if percent_complete and percent_complete.isdigit() else 0.0 
        except ValueError:
            self.percent_complete = 0.0
        self.urgent = urgent if urgent != "None" else ""

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
                    "decision": decision,
                    "urgent": self.urgent,
                    "past_feedback": self.past_feedback
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log status phase event: {e}")


        # If the decision is NEW_MISSION and this agent is the leader, reprompt for a new decision
        if decision == "NEW_MISSION" and getattr(self, "leader", False) and self.mission != "NO MISSION ASSIGNED":
            # The leader should not request a new mission unless the mission is fully complete
            reprompt_message = (
                f"""The Mission '{self.mission}' is NOT fully complete.
                Please reconsider and respond with an appropriate decision (e.g., CONTINUE_MISSION, REWRITE_TASKS).


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

        # Step 3: Confirm decision if it's not CONTINUE_MISSION
        if decision != "CONTINUE_MISSION":
            self.log_chat("Team Status Summary", [("user", user_message), ("assistant", status_response)])
            decision = await self._async_confirm_decision_horizontal(decision, global_data)

        emit_event(
            self._lobby_id,
            "decision_made",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} decided: {decision}",
            {"decision": decision},
        )

        # Update status based on decision (no phase transitions)
        # past_feedback is NOT cleared here — it persists until popped after action phase commit
        if decision == "NEW_MISSION":
            self.status = 0
        elif decision == "CONTINUE_MISSION":
            self.status = 1
        elif decision == "REWRITE_TASKS":
            self.status = 4
        else:
            self.status = 1

    async def _async_confirm_decision_horizontal(self, decision: str, global_data: dict) -> str:
        """
        Async version of confirm decision for horizontal manager (no phase logic)
        """
        warning_message = ""
        if decision == "REWRITE_TASKS":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions for this mission."
        elif decision == "NEW_MISSION":
            warning_message = "WARNING: This decision will overwrite all ongoing/queued actions for this mission."

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
        CANCEL: Continue with the current mission instead

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
        self.log_chat("Decision Confirmation", [("user", confirmation_message), ("assistant", confirmation_response)])

        # Parse confirmation
        confirmation = self._parse_tag_content(confirmation_response, "confirmation")

        if confirmation == "CONFIRM":
            print(f"AGENT_{self.id}: Confirmed decision to {decision}")
            return decision
        else:
            print(f"AGENT_{self.id}: Cancelled decision, continuing with current mission")
            return "CONTINUE_MISSION"

    def _record_task_assignment(self, global_data: dict, reasoning: str = ""):
        """
        Records the current task assignments to history with reasoning.

        Args:
            global_data: Global data containing timestep info
            reasoning: The reasoning for this task assignment/rewrite
        """
        tasks = {}
        for agent in self.children:
            tasks[agent.name] = getattr(agent, 'mission', 'No task assigned')

        record = {
            "timestep": global_data.get("timestep", 0),
            "mission": self.mission,
            "reasoning": reasoning,
            "tasks": tasks
        }

        self.task_history.append(record)
        print(f"AGENT_{self.id}: Recorded task assignment #{len(self.task_history)} at timestep {record['timestep']}")

    def _format_task_history(self) -> str:
        """
        Formats the task history into a readable string for LLM context.

        Returns:
            Formatted string showing the history of task assignments
        """
        if not self.task_history:
            return "No previous task assignments."

        formatted_history = []
        for i, record in enumerate(self.task_history, 1):
            formatted_history.append(f"=== Task Assignment #{i} (Timestep {record['timestep']}) ===")
            formatted_history.append(f"Mission: {record['mission']}")
            if record['reasoning']:
                formatted_history.append(f"Reasoning: {record['reasoning']}")
            formatted_history.append("Task Assignments:")
            for agent_name, task in record['tasks'].items():
                formatted_history.append(f"  - {agent_name}: {task}")
            formatted_history.append("")  # blank line between records

        return "\n".join(formatted_history)

    async def async_action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
        """
        Async action phase with horizontal task distribution only (no phases)
        """
        self._did_replan = False
        print(f"AGENT_{self.id}: Horizontal Manager Async Action Phase")

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
                child.upperteam_mission = self.mission
            return

        # Step 3: Non-CONTINUE — CHECKPOINT: save children state before handler mutates them
        checkpoint = {c.id: (c.mission, c.status, c.percent_complete) for c in self.children}
        saved_status = self.status

        # Execute handler (mutates children)
        if self.status == 4:
            await self._async_handle_task_rewrite_horizontal(global_data)
        elif self.status == 5:
            await self._async_handle_new_mission_horizontal(global_data)
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
            self.status = saved_status

            # Re-run status (will see new URGENT items in past_feedback)
            await self.async_status_phase(global_data, agent_states)
            # Restore items wiped by status phase clear
            self._processed_fast_feedback = saved_processed + self._processed_fast_feedback

            # Re-execute if needed
            if self.status == 4:
                await self._async_handle_task_rewrite_horizontal(global_data)
                self._did_replan = True
            elif self.status == 5:
                await self._async_handle_new_mission_horizontal(global_data)
                self._did_replan = True

        # Feedback was acted on — convert URGENT items to standing orders
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

        for child in self.children:
            child.upperteam_mission = self.mission

    async def _async_handle_task_rewrite_horizontal(self, global_data: dict):
        """Async version of task rewrite without phase logic"""

        team_summary = self._build_team_summary()

        agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

        # Include task history in the prompt
        history_section = ""
        if self.task_history:
            history_section = f"""
        TASK ASSIGNMENT HISTORY:
        {self._format_task_history()}

        The above shows all previous task assignments for this mission, including the reasoning for each change.
        """

        # Include human feedback so LLM knows WHY it's rewriting
        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        user_message = f"""Great, now rewrite tasks for the current mission: '{self.mission}'

        Your team composition and team abilities:
        {team_summary}

        {history_section}
        {feedback_section}
        Explain what changed and why a new plan for this mission is necessary, then break down the mission into new tasks using the following tags:

        <explanation>
        Explain what changed and why a new plan for this mission is necessary.
        </explanation>
        {agent_tags}

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

        ONLY ASSIGN ONE TASK PER AGENT.
        """

        self.action_messages = self.status_messages.copy()  # Continue conversation
        self.action_messages.append({"role": "user", "content": user_message})

        plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Replan Mission", [("user", user_message), ("assistant", plan)])
        self.action_messages.append({"role": "assistant", "content": plan})

        self.timestep_chats.append({
            "interaction": "Replan Mission",
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

        # Extract explanation/reasoning for history
        explanation = self._parse_tag_content(plan, "explanation")
        if not explanation:
            explanation = "Task rewrite requested"

        # Assign new tasks
        self._assign_tasks_from_plan(plan, global_data)

        # Record task assignment to history
        self._record_task_assignment(global_data, reasoning=explanation)

        self.status = 1

        emit_event(
            self._lobby_id,
            "plan_written",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} rewrote task assignments",
            {"mission": self.mission},
        )

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
                    "task_designations": task_designations,
                    "past_feedback": self.past_feedback
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log task rewrite event: {e}")

    async def _async_handle_new_mission_horizontal(self, global_data: dict):
        """Async version of new mission handling without phase logic"""

        self.percent_complete = 0

        # Clear task history when starting a new mission
        self.task_history = []

        team_summary = self._build_team_summary(detailed=True)

        system_message = f"""You are AGENT_{self.id}, the Team Manager of {self.team_name}: {[a.name for a in self.children if getattr(a, 'alive', True)]}, part of a broader team of embodied agents within a grid world. The map is made up of {self.cfg.envs.map_size} by {self.cfg.envs.map_size} cells/grids with coordinates in the range of [0 to {self.cfg.envs.map_size-1}, 0 to {self.cfg.envs.map_size-1}] with the top left corner of the map being (0,0).

        Your team composition and team abilities:
        {team_summary}

        Knowledge Base (may or may not be relevant to the current mission/team):
        {self.knowledge_base}

        Terminology:
        - Mission: The overall goal for your team (e.g., "Locate and suppress the fire")
        - Task: The specific assignment for each agent in your team

        Your job is to break down new missions into tasks for your team members.
        """

        self.action_messages = [{"role": "system", "content": system_message}]

        agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

        feedback_section = ""
        if self.past_feedback:
            feedback_section = f"""
        HUMAN OPERATOR FEEDBACK TO INCORPORATE:
        {self._format_feedback()}
        """

        # Break down mission into tasks
        user_message = f"""Your mission: '{self.mission}'
        Upper team mission: {self.upperteam_mission}
        {feedback_section}
        Break down your mission into tasks for each team member using the following tags:

        {agent_tags}

        Consider each agent's capabilities, current position, and status when assigning tasks.

        Keep each task concise — one or two short sentences. Be specific with coordinates but skip unnecessary detail, such as communication, staying alert, reporting observations, etc. However, the completion of all tasks SHOULD result in the full completion of the phase, so make them complete.

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

        ONLY ASSIGN ONE TASK PER AGENT.

        Try to divide up the entire mission into this one set of tasks. You may not have another chance to replan. A task can be multiple actions as long as it is exactly one line.
        """


        self.action_messages.append({"role": "user", "content": user_message})

        plan = await self._async_openai_call(self.action_messages, global_data, depth="high")
        self.log_chat("Creating Tasks", [("system", system_message), ("user", user_message), ("assistant", plan)])
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

                feedback = await self._async_collect_team_feedback(plan, global_data)
            else:
                consensus = True
            iteration += 1

        # Assign tasks and update status
        self._assign_tasks_from_plan(plan, global_data)

        # Record initial task assignment to history for new mission
        self._record_task_assignment(global_data, reasoning=f"Initial task assignments for new mission: {self.mission}")

        self.status = 1

        emit_event(
            self._lobby_id,
            "plan_written",
            self.id,
            self.name,
            global_data.get("timestep", 0),
            f"{self.name} wrote initial task assignments for new mission",
            {"mission": self.mission},
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
                    "task_designations": task_designations,
                    "past_feedback": self.past_feedback
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log new mission event: {e}")

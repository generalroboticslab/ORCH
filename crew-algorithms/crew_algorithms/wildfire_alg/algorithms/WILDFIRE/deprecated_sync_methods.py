"""
DEPRECATED SYNC METHODS
=======================
These methods have been replaced by async equivalents.
Kept for reference only - DO NOT USE.

Last updated: 2026-02-16

This file contains deprecated synchronous methods from:
- agent.py (Agent class)
- worker_agent.py (WorkerAgent class)
- utils.py

All code below is commented out to prevent accidental use.
"""

'''
# ============================================================================
# DEPRECATED METHODS FROM agent.py (Agent class)
# ============================================================================

# -----------------------------------------------------------------------------
# Agent.status_phase - DEPRECATED (use async_status_phase instead)
# Original location: agent.py lines 541-895
# -----------------------------------------------------------------------------

def status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]], human_feedback: str = None):
    """
     Enhanced status phase with phase-based team coordination
     """
     print(f"AGENT_{self.id}: Status Phase")

    # Reset timestep chats at the beginning of each status phase
    self.timestep_chats = []

    # Start conversation with team context if empty

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

    {'Critical Human Feedback for this Phase' +  str(self.past_feedback) if self.past_feedback else ""}


    Knowledge Base (may or may not be relevant to the current mission/team):
    {self.knowledge_base}

    Your role is to:
    1. Collect and summarize observations from your subteam
    2. Assess progress on current phase and overall mission
    3. Make decisions about phase transitions and task adjustments

    Always respond using the required tag structure and example format."""

    self.status_messages.append({"role": "system", "content": system_message})

    # Update team composition
    self._update_team_composition()

    # Interaction 1: Generate team perception summary
    team_context = "Here are your team's observations by Agent: \n\n"
    for worker in self.children:
        team_context += f"{worker.name}: \n\n{str(worker.perception_summary)}\n\n---\n\n"

    user_message = f"""Given your team's observations, Summarize your team's collective perception.

        {team_context}

    Summarize your team's collective perception using the following tags. Do not reference any agents by name, but rather refer to them collectively as "my team":

    <perception>
    A detailed summary of your team's collective observations,  Include what your team has discovered, their overall positions, and any important findings. Do not refer to any agents by name.
    </perception>

    DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.

    """

    self.status_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=self.status_messages,

    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    perception_response = response.choices[0].message.content
    self.status_messages.append({"role": "assistant", "content": perception_response})
    self.log_chat("Perception Summary", [("system", system_message), ("user", user_message), ("assistant", perception_response)])

    # Store LLM response for human observation
    self.timestep_chats.append({
        "interaction": "Team Perception Summary",
        "response": perception_response
    })

    # Parse perception from tags
    perception = self._parse_tag_content(perception_response, "perception")
    self.perception_summary = perception
    client.close()

    # Add human feedback to conversation if provided
    if human_feedback:
        feedback_message = f"""IMPORTANT: Your human observer has just provided the following critical feedback: "{human_feedback}"

        Include this feedback in your status summary."""
        self.status_messages.append({"role": "user", "content": feedback_message})

        # Simple acknowledgment from assistant
        ack_response = "I understand the critical human feedback and will incorporate it into my decision-making and status summary, even if it overrides previous decisions."
        self.status_messages.append({"role": "assistant", "content": ack_response})

        self.log_chat("Human Feedback", [("user", feedback_message), ("assistant", ack_response)])
        self.past_feedback.append(human_feedback)
    # Interaction 2: Generate team status with phase decisions
    # Create aligned subteam status and task information
    subteam_alignment = []
    for agent in self.children:
        agent_name = agent.name
        agent_task = getattr(agent, 'mission', 'No task assigned')
        agent_status = getattr(agent, 'status_summary', 'No status available')
        agent_percent_complete = getattr(agent, 'percent_complete', 'No percent complete available')


        agent_urgent = getattr(agent, 'urgent', 'None')

        subteam_alignment.append(f"Agent: {agent_name}")
        subteam_alignment.append(f"  Task: {agent_task}")
        subteam_alignment.append(f"  Percent Complete with Task: {agent_percent_complete}")
        subteam_alignment.append(f"  Status: {agent_status}")
        subteam_alignment.append(f"  Urgent: {agent_urgent}")

        if agent.status == 0:
            subteam_alignment.append(f"  State: IDLE ON STANDBY (NO ACTIVE ACTIONS)")
        else:
            subteam_alignment.append(f"  State: EXECUTING ACTIONS")

        subteam_alignment.append("")

    subteam_info = "\n".join(subteam_alignment)

    # --- MODIFIED PROMPT LOGIC FOR END OF PHASES ---
    if not self.future_phases:
        # No future phases, so offer ADD_PHASES instead of NEXT_PHASE
        user_message = f"""Given your current mission: '{self.mission}'

    Your timeline: Past Phases: {self.phase_history}, Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

    Your current phase's completion condition: {self.phase_completion_condition}

    Your subteam's current status and tasks:
    {subteam_info}

    Provide:

    1. A status summary using the following tags:
    <summary>
    A concise summary of your team's progress and situation, including what has been accomplished and current status. Do not cite specific agents, only the team as a whole.
    </summary>
    <phase_percent_complete>
    A number from 0 to 100 representing your team's estimated percent complete on the current phase.
    </phase_percent_complete>
    <mission_percent_complete>
    A number from 0 to 100 representing your team's estimated percent complete on the current mission.
    </mission_percent_complete>
    <urgent>
    Any urgent suggestions or issues for your upper team. If none, write "None".
    </urgent>

    2. A decision using the following tag:
    <reasoning>
    Any reasoning for your decision.
    </reasoning>
    <decision>
    One of: CONTINUE_PHASE, ADD_PHASE, REWRITE_TASKS, NEW_MISSION
    </decision>

    CONTINUE_PHASE: If the team is on track to complete the current phase ('{self.current_phase}').
    ADD_PHASE: If the team has FULLY completed ALL planned phases ('{self.current_phase}'), but the mission ('{self.mission}') is not fully complete. The status of the current phase should be 100% complete. You may always add new phases later.
    REWRITE_TASKS: If the team needs new/reorganized tasks for the current phase ('{self.current_phase}'). Rewrite the tasks. You may always rewrite tasks later.
    NEW_MISSION: If the team has FULLY completed the current mission ('{self.mission}'), regardless if all phases are necessarily complete. Request a new mission. You may always request a new mission later.

    Example response:
    <summary>
    The team has completed all planned phases but the fire is not fully contained. Additional work is needed.
    </summary>
    <phase_percent_complete>
    100
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
    ADD_PHASES
    </decision>"""


    else:
        user_message = f"""Given your current mission: '{self.mission}'

    Your timeline: Past Phases: {self.phase_history}, Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

    Your current phase's completion condition: {self.phase_completion_condition}

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
    Any reasoning for your decision.
    </reasoning>
    <decision>
    One of: CONTINUE_PHASE, NEXT_PHASE, REWRITE_TASKS,NEW_MISSION
    </decision>

    CONTINUE_PHASE: If the team is on track to complete the current phase ('{self.current_phase}').
    NEXT_PHASE: If the team has FULLY completed the current phase and the next phase is needed ('{self.future_phases[0]}'). The phase completion condition should already be met. You may always move to the next phase later.
    REWRITE_TASKS: If the team needs new/reorganized tasks for the current phase ('{self.current_phase}'). Rewrite the tasks only if the current tasks are insufficient. You may always rewrite tasks later.
    NEW_MISSION: If the team has FULLY completed the current mission, regardless if all phases are necessarily complete. Request a new mission. You may always request a new mission later.

    Example response:
    <summary>
    The team has scouted 80% of the area and found a small fire in the northern sector. All agents are progressing as planned.
    </summary>
    <phase_percent_complete>
    80
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
    CONTINUE_PHASE
    </decision>"""

    self.status_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=self.status_messages,

    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    status_response = response.choices[0].message.content
    self.status_messages.append({"role": "assistant", "content": status_response})
    self.log_chat("Status Summary", [("user", user_message), ("assistant", status_response)])

    # Store LLM response for human observation
    self.timestep_chats.append({
        "interaction": "Team Status Summary",
        "response": status_response
    })

    client.close()

    # Parse status summary and decision from tags
    status_summary = self._parse_tag_content(status_response, "summary")
    percent_complete = self._parse_tag_content(status_response, "mission_percent_complete")
    phase_complete = self._parse_tag_content(status_response, "phase_percent_complete")
    urgent = self._parse_tag_content(status_response, "urgent")
    decision = self._parse_tag_content(status_response, "decision")

    # Store parsed information
    self.status_summary = status_summary
    self.percent_complete    = float(percent_complete) if percent_complete and percent_complete.isdigit() else 0.0
    self.phase_progress = float(phase_complete) if phase_complete and phase_complete.isdigit() else 0.0
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
                "phase": self.current_phase,
                "phase_percent": self.phase_progress,
                "decision": decision,
                "urgent": self.urgent
            },
            phase=self.current_phase
        )
    except Exception as e:
        print(f"Warning: Failed to log status phase event: {e}")

    # Confirm decision if it's not CONTINUE_PHASE
    if decision != "CONTINUE_PHASE":
        decision = self._confirm_decision(decision, global_data)





    # Parse phase decisions and update status
    if not self.future_phases:
        if decision == "ADD_PHASE":
            self.status = 3  # Move to next phase (add phases)
            self.past_feedback = []
        elif decision == "REWRITE_TASKS":
            self.status = 4
            self.past_feedback = []
        elif decision == "CONTINUE_PHASE":
            self.status = 1
        elif decision == "NEW_MISSION":
            self.status = 0
            self.past_feedback = []
        else:
            self.status = 1
    else:
        if decision == "NEXT_PHASE":
            self.status = 3  # Move to next phase
            self.past_feedback = []
        elif decision == "REWRITE_TASKS":
            self.status = 4  # Rewrite tasks in current phase
            self.past_feedback = []
        elif decision == "CONTINUE_PHASE":
            self.status = 1  # Continue with current phase
        elif decision == "NEW_MISSION":
            self.status = 0  # Request new mission
            self.past_feedback = []
        else:
            self.status = 1  # Default to continue


# -----------------------------------------------------------------------------
# Agent._confirm_decision - DEPRECATED (use _async_confirm_decision instead)
# Original location: agent.py lines 968-1033
# -----------------------------------------------------------------------------

def _confirm_decision(self, decision: str, global_data: dict) -> str:
    """
    Confirm a non-continue phase decision with the agent, noting that ongoing phases/actions will be overwritten
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

    {warning_message}

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

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=self.status_messages,

    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    confirmation_response = response.choices[0].message.content
    self.status_messages.append({"role": "assistant", "content": confirmation_response})
    self.log_chat("Decision Confirmation", [("user", confirmation_message), ("assistant", confirmation_response)])
    client.close()

    # Parse confirmation
    confirmation = self._parse_tag_content(confirmation_response, "confirmation")

    if confirmation == "CONFIRM":
        print(f"AGENT_{self.id}: Confirmed decision to {decision}")
        return decision
    else:
        print(f"AGENT_{self.id}: Cancelled decision, continuing with current phase")
        return "CONTINUE_PHASE"


# -----------------------------------------------------------------------------
# Agent.action_phase - DEPRECATED (use async_action_phase instead)
# Original location: agent.py lines 1098-1128
# -----------------------------------------------------------------------------

def action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
    """
    Enhanced action phase with phase-based planning
    """
    print(f"AGENT_{self.id}: Action Phase")

    self.action_messages = []
    # Handle different scenarios based on status
    if self.status == 0:
        # Request new mission - handled by main loop
        return
    elif self.status == 1:
        # Continue with current phase - no action needed
        return
    elif self.status == 3:
        # Move to next phase or add phases if none left
        if not self.future_phases:
            self._add_phases_to_mission(global_data)
        self._handle_phase_transition(global_data)
    elif self.status == 4:
        # Rewrite tasks in current phase
        self._handle_task_rewrite(global_data)
    elif self.status == 5:
        # New mission received
        self._handle_new_mission(global_data)

    for child in self.children:
        child.upperteam_phase = self.current_phase
        child.upperteam_mission = self.mission


# -----------------------------------------------------------------------------
# Agent._handle_phase_transition - DEPRECATED (use _async_handle_phase_transition instead)
# Original location: agent.py lines 1130-1283
# -----------------------------------------------------------------------------

def _handle_phase_transition(self, global_data: dict):
    """Move to next phase and break it down into tasks"""
    # Continue from status phase conversation
    self.phase_history.append(self.current_phase)
    self.current_phase = self.future_phases.pop(0)
    self.phase_progress=0
    self.phase_completion_condition = ""  # Reset completion condition for new phase


    team_info = []
    for agent in self.children:
        agent_name = agent.name
        agent_type = agent._get_agent_type_string() if hasattr(agent, '_get_agent_type_string') else f"Type {agent.type}"
        agent_overview = getattr(agent, 'overview', 'No overview available')


        # Format agent information
        agent_info = f"Agent: {agent_name} ({agent_type})"
        agent_info += f"\n  Overview: {agent_overview}"
        agent_info += "\n"

        team_info.append(agent_info)

    team_summary = "\n".join(team_info)

    agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

    user_message = f"""Great, now let's move to the next phase.

    Timeline: Past Phases: {self.phase_history}, New Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

    Your team composition and team abilities:
    {team_summary}


    Break down the next phase into tasks for each team member and define a completion condition using the following tags:

    {agent_tags}

    <completion_condition>
    A short, specific condition that defines when this phase is complete. Be concrete and measurable.
    </completion_condition>

    Example response:
    <AGENT_1>
    Move to the northeast region around (x,y) and scout for new fire outbreaks.
    </AGENT_1>
    <AGENT_2>
    Send your team to extingush the fire at the southern border along y=400.
    </AGENT_2>
    <AGENT_3>
    Cut all trees at (x,y), (x,y)...
    </AGENT_3>
    <completion_condition>
    Northeast region has been scouted, fire at southern border has been extinguished, and all trees at (x,y), (x,y)... have been cut.
    </completion_condition>

    ONLY ASSIGN ONE TASK PER AGENT.
    """

    self.action_messages = self.status_messages.copy()  # Continue conversation
    self.action_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=self.action_messages,

    )


    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    plan = response.choices[0].message.content
    self.log_chat("Plan", [("user", user_message), ("assistant", plan)])
    self.action_messages.append({"role": "assistant", "content": plan})
    client.close()

    self.timestep_chats.append({
        "interaction": "Next Phase Plan Draft",
        "response": plan
    })
    # Collect feedback and refine
    feedback = self._collect_team_feedback(plan, global_data)

    # Refine plan based on feedback
    consensus = False
    max_iterations = 0
    iteration = 0

    while not consensus and iteration < max_iterations:
        if feedback and not all("YES" in f for f in feedback.values()):
            user_message = f"""Here is feedback from your team:
    {self._build_feedback_summary(feedback)}

    Revise your plan accordingly using the same tag structure."""

            self.action_messages.append({"role": "user", "content": user_message})

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.action_messages,

            )
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens

            plan = response.choices[0].message.content
            self.log_chat("Revise Plan", [("user", user_message), ("assistant", plan)])
            self.action_messages.append({"role": "assistant", "content": plan})
            client.close()

            feedback = self._collect_team_feedback(plan, global_data)
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
                "task_designations": task_designations
            },
            phase=self.current_phase
        )
    except Exception as e:
        print(f"Warning: Failed to log action phase event: {e}")


# -----------------------------------------------------------------------------
# Agent._handle_task_rewrite - DEPRECATED (use _async_handle_task_rewrite instead)
# Original location: agent.py lines 1285-1433
# -----------------------------------------------------------------------------

def _handle_task_rewrite(self, global_data: dict):
    """Rewrite tasks in current phase"""

    self.phase_progress = 0

    team_info = []
    for agent in self.children:
        agent_name = agent.name
        agent_type = agent._get_agent_type_string() if hasattr(agent, '_get_agent_type_string') else f"Type {agent.type}"
        agent_overview = getattr(agent, 'overview', 'No overview available')


        # Format agent information
        agent_info = f"Agent: {agent_name} ({agent_type})"
        agent_info += f"\n  Overview: {agent_overview}"
        agent_info += "\n"

        team_info.append(agent_info)

    team_summary = "\n".join(team_info)

    agent_tags = '\n'.join([f'<{a_name}>\nTask for this agent.\n</{a_name}>' for a_name in [a.name for a in self.children if a.alive]])

    user_message = f"""Great, now rewrite tasks for the current phase: '{self.current_phase}'
    Timeline: Past Phases: {self.phase_history}, Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

    Your team composition and team abilities:
    {team_summary}


    Explain what changed and why a new plan for this phase is necessary, then break down the current phase into new tasks and update the completion condition using the following tags:

    <explanation>
    Explain what changed and why a new plan for this phase is necessary.
    </explanation>
    {agent_tags}
    <completion_condition>
    A short, specific condition that defines when this first phase is complete. Be concrete and measurable.
    </completion_condition>

    Example response:
    <explanation>
    The fire has been unexpectantly spotted in the northeast. We need to reassign tasks to the agents.
    </explanation>

    <AGENT_1>
    Move to the northeast region around (x,y) and scout for new fire outbreaks.
    </AGENT_1>
    <AGENT_2>
    Send your team to extingush the fire at the southern border along y=400.
    </AGENT_2>
    <AGENT_3>
    Cut all trees at (x,y), (x,y)...
    </AGENT_3>
    <completion_condition>
    Northeast region has been scouted and fire at southern border has been extinguished.
    </completion_condition>

    ONLY ASSIGN ONE TASK PER AGENT.
    """

    self.action_messages = self.status_messages.copy()  # Continue conversation
    self.action_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=self.action_messages,

    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    plan = response.choices[0].message.content
    self.log_chat("Replan Phase", [("user", user_message), ("assistant", plan)])
    self.action_messages.append({"role": "assistant", "content": plan})
    client.close()

    self.timestep_chats.append({
        "interaction": "Replan Phase",
        "response": plan
    })

    # Collect feedback and refine
    feedback = self._collect_team_feedback(plan, global_data)

            # Refine plan based on feedback
    consensus = False
    max_iterations = 0
    iteration = 0

    while not consensus and iteration < max_iterations:
        if feedback and not all("YES" in f for f in feedback.values()):
            user_message = f"""Here is feedback from your team:
    {self._build_feedback_summary(feedback)}

    Revise your plan accordingly using the same tag structure."""

            self.action_messages.append({"role": "user", "content": user_message})

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.action_messages,

            )
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens

            plan = response.choices[0].message.content
            self.log_chat("Revise Plan", [("user", user_message), ("assistant", plan)])
            self.action_messages.append({"role": "assistant", "content": plan})
            client.close()

            feedback = self._collect_team_feedback(plan, global_data)
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
                "task_designations": task_designations
            },
            phase=self.current_phase
        )
    except Exception as e:
        print(f"Warning: Failed to log task rewrite event: {e}")


# -----------------------------------------------------------------------------
# Agent._handle_new_mission - DEPRECATED (use _async_handle_new_mission instead)
# Original location: agent.py lines 1435-1647
# -----------------------------------------------------------------------------

def _handle_new_mission(self, global_data: dict):
        """Handle new mission by breaking it into phases and tasks"""
        
        self.percent_complete=0
        self.phase_progress = 0
        self.phase_history = []
        self.future_phases = []
        self.current_phase = None
        self.phase_completion_condition = ""  # Reset completion condition for new mission

        # Compile detailed team information by agent
        team_info = []
        for agent in self.children:
            agent_name = agent.name
            agent_type = agent._get_agent_type_string() if hasattr(agent, '_get_agent_type_string') else f"Type {agent.type}"
            agent_overview = getattr(agent, 'overview', 'No overview available')

            
            # Format agent information
            agent_info = f"Agent: {agent_name} ({agent_type})"
            agent_info += f"\n  Overview: {agent_overview}"
            agent_info += "\n"
            
            team_info.append(agent_info)
        
        team_summary = "\n".join(team_info)

        system_message = f"""You are AGENT_{self.id}, the Team Manager of {self.team_name}: {[a.name for a in self.children]}, part of a broader team of embodied agents within a grid world. The map is made up of {self.cfg.envs.map_size} by {self.cfg.envs.map_size} cells/grids with coordinates in the range of [0 to {self.cfg.envs.map_size-1}, 0 to {self.cfg.envs.map_size-1}] with the top left corner of the map being (0,0).

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
                
        user_message = f"""Your mission: '{self.mission}'
        Upper team phase: {self.upperteam_phase}
        Upper team mission: {self.upperteam_mission}

        First, break your mission into phases using the following tags:

        <phases>
        List each phase on a separate line, numbered or with bullet points.
        </phases>

        Example response:
        <phases>
        1. Scout for the fire
        2. Move to the fire and suppress it
        3. Check if the fire is fully contained and fill in gaps

        </phases>

        You may only need one/a few phases. Do not include unnecessary phases, or phases that are outside of agent capabilities.
        DO NOT BREAK DOWN ANY PHASES INTO TASKS YET. KEEP THEM HIGH LEVEL AND CONCISE.
        """
                
        self.action_messages.append({"role": "user", "content": user_message})
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.action_messages,
            
        )
        global_data["api_calls"] += 1
        global_data["input_tokens"] += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens
        
        phases_response = response.choices[0].message.content
        self.log_chat("Creating Phases", [("system", system_message), ("user", user_message), ("assistant", phases_response)])
        self.action_messages.append({"role": "assistant", "content": phases_response})
        client.close()

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
        Cut all trees at (x,y), (x,y)...
        </AGENT_1>
        <AGENT_2>
        Scout the northeast region around (x,y) for new fire outbreaks.
        </AGENT_2>
        <AGENT_3>
        Send your team to extingush the fire at the southern border along y=400.
        </AGENT_3>
        <completion_condition>
        All trees at specified coordinates have been cut and northeast region has been fully scouted.
        </completion_condition>

        ONLY ASSIGN ONE TASK PER AGENT.
        """


        
        self.action_messages.append({"role": "user", "content": user_message})

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=self.action_messages,
            
        )
        global_data["api_calls"] += 1
        global_data["input_tokens"] += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens

        plan = response.choices[0].message.content
        self.log_chat("Creating Tasks", [("user", user_message), ("assistant", plan)])
        self.action_messages.append({"role": "assistant", "content": plan})
        client.close()
        
        self.timestep_chats.append({
            "interaction": "Creating Tasks",
            "response": plan
        })
        # Collect feedback and refine
        feedback = self._collect_team_feedback(plan, global_data)
        
        # Refine plan based on feedback
        consensus = False
        max_iterations = 0
        iteration = 0
        
        while not consensus and iteration < max_iterations:
            if feedback and not all("YES" in f for f in feedback.values()):
                user_message = f"""Here is feedback from your team:
        {self._build_feedback_summary(feedback)}

        Revise your plan accordingly using the same tag structure."""
                
                self.action_messages.append({"role": "user", "content": user_message})
                
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=self.action_messages,
                    
                )
                global_data["api_calls"] += 1
                global_data["input_tokens"] += response.usage.prompt_tokens
                global_data["output_tokens"] += response.usage.completion_tokens
                
                plan = response.choices[0].message.content
                self.log_chat("Revise Tasks", [("user", user_message), ("assistant", plan)])
                self.action_messages.append({"role": "assistant", "content": plan})
                client.close()
                
                feedback = self._collect_team_feedback(plan, global_data)
            else:
                consensus = True
            iteration += 1
        
        # Assign tasks and update status
        self._assign_tasks_from_plan(plan, global_data)
        self.status = 1
        
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
                    "task_designations": task_designations
                },
                phase=self.current_phase
            )
        except Exception as e:
            print(f"Warning: Failed to log new mission event: {e}")
        

# -----------------------------------------------------------------------------
# Agent._collect_team_feedback - DEPRECATED (use _async_collect_team_feedback instead)
# Original location: agent.py lines 1715-1846
# -----------------------------------------------------------------------------

def _collect_team_feedback(self, plan: str, global_data: dict) -> Dict[str, str]:
        """Collect feedback from team members on the proposed plan"""
        
        feedback = {}
        
        
        for agent in [a for a in self.children if a.alive]:
            # Load the appropriate child feedback prompt for this agent type
            if agent.type == -1:  # Manager agent
                agent_type = "manager"
            else:
                agent_type_map = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter"}
                agent_type = agent_type_map.get(agent.type, "firefighter")
            
            try:
                child_feedback_path = os.path.join("algorithms", "WILDFIRE", "prompts", "child_feedback", f"{agent_type}_child_feedback.txt")
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
            
            # Create feedback prompt for each agent
            if agent_type == "manager":
                feedback_message = f"""
                You are a member of a team structure {[i.name for i in self.children]}, including yourself: {agent.name}.
                The team's current phase is: {self.current_phase}, which is part of the broader plan: {self.mission}. This is the timeline of the plan: Past Phases: {self.phase_history}, New Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}


                Your observations:
                {agent.perception_summary}

                Knowledge Base (may or may not be relevant to the current mission/team):
                {self.knowledge_base}

                Here is the proposed plan for the team by your Team Manager specifically for the current phase ('{self.current_phase}'):

                {plan}

                As a subteam manager, examine your OWN subteam's designated role in the plan. Consider your subteam's capabilities and current status.
                Given your subteam's observations, status, and capabilities, determine if this plan is effective for you and your subteam and consistent with your subteam's capabilities. 

                If it satisfactory, respond with 'YES'
                If it instead requires adjustment for your subteam's role, give concise feedback directed towards the Team Manager.
                """
            else:
                feedback_message = f"""
                You are a member of a team structure {[i.name for i in self.children]}, including yourself: {agent.name}.
                The team's current phase is: {self.current_phase}, which is part of the broader plan: {self.mission}. This is the timeline of the plan: Past Phases: {self.phase_history}, New Current Phase: '{self.current_phase}', Future Phases: {self.future_phases}

                Your observations:
                {agent.perception_summary}

                Knowledge Base (may or may not be relevant to the current mission/team):
                {self.knowledge_base}

                Here is the proposed plan for the team by your Team Manager specifically for the current phase ('{self.current_phase}'):

                {plan}

                Specifically, examine your OWN designated role in the plan. Do not concern yourself with the role of your team members.
                Given your observations, status, and capabilities, determine if this plan is effective for you and consistent with your capabilities. 

                If it satisfactory, respond with 'YES'
                If it instead requires adjustment for your role, give concise feedback directed towards the Team Manager.
                """
            
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": child_system_prompt},
                    {"role": "user", "content": feedback_message}
                ],
                
            )
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens

            agent_feedback = response.choices[0].message.content
            feedback[agent.name] = agent_feedback
            self.log_chat(f"Agent_{agent.id} Feedback", [("user", feedback_message), ("assistant", agent_feedback)])
            client.close()
        
        return feedback
        



# -----------------------------------------------------------------------------
# Agent._add_phases_to_mission - DEPRECATED (use _async_add_phases_to_mission instead)
# Original location: agent.py lines 2001-2072
# -----------------------------------------------------------------------------

def _add_phases_to_mission(self, global_data: dict):
    """Prompt the agent to generate new phases for the current mission if all phases are complete."""
    team_info = []
    self.phase_progress=0

    for agent in self.children:
        agent_name = agent.name
        agent_type = agent._get_agent_type_string() if hasattr(agent, '_get_agent_type_string') else f"Type {agent.type}"
        agent_overview = getattr(agent, 'overview', 'No overview available')
        agent_info = f"Agent: {agent_name} ({agent_type})"
        agent_info += f"\n  Overview: {agent_overview}\n"
        team_info.append(agent_info)
    team_summary = "\n".join(team_info)

    user_message = f"""You have completed all planned phases for your mission: '{self.mission}'.
    Now, generate additional phases to continue the mission.
    Your team composition and team abilities:
    {team_summary}

    List new phases using the following tags:
    <phases>
    1. ...
    2. ...
    </phases>
    Only include necessary phases.
    """
    self.action_messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=self.action_messages,

    )
    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    phases_response = response.choices[0].message.content
    self.log_chat("Adding Phases", [("user", user_message), ("assistant", phases_response)])
    self.action_messages.append({"role": "assistant", "content": phases_response})
    client.close()

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
                "new_phases_added": new_phases
            },
            phase=self.current_phase
        )
    except Exception as e:
        print(f"Warning: Failed to log add phases event: {e}")


# ============================================================================
# DEPRECATED METHODS FROM worker_agent.py (WorkerAgent class)
# ============================================================================

# -----------------------------------------------------------------------------
# WorkerAgent.status_phase - DEPRECATED (use async_status_phase instead)
# Original location: worker_agent.py lines 77-276
# -----------------------------------------------------------------------------

def status_phasedef status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]], human_feedback: str = None):
        """
        Enhanced status phase with structured output parsing
        """
        print(f"AGENT_{self.id}: Status Phase")
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

Your job is to analyze observations and provide status updates with structured output using tags."""
        
        self.status_messages = [{"role": "system", "content": system_message}]
        
        # Interaction 1: Generate perception
        if self.last_observation:
            user_message = f"""Here are your observations: {self._build_observation_string(agent_states, global_data)}

Create a detailed perception summary (<=100 words) using the following tags:

<perception>
A detailed summary of what you observe, including your location, surroundings, any fires, civilians, or important features you can see, terrain types, other agents in your vicinity, and your current status (carrying capacity, etc.). Focus on information relevant to your current task.
</perception>


DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.

Example response:
<perception>
I am currently at coordinates (5,10) in a medium forest cell. I can see dense forest to the north, a water source to the east at (7,10), and no signs of fire in my immediate vicinity. There are no civilians nearby. I am not carrying any civilians and have 3/5 water remaining. I can see AGENT_2 at coordinates (6,9) to the northeast.
</perception>"""
            
            self.status_messages.append({"role": "user", "content": user_message})
            
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-5-2025-08-07",
                messages=self.status_messages,
                
            )
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens
            
            perception_response = response.choices[0].message.content
            self.log_chat("Perception Summary", [("system", system_message), ("user", user_message), ("assistant", perception_response)])
            self.status_messages.append({"role": "assistant", "content": perception_response})
            
            # Store LLM response for human observation
            self.timestep_chats.append({
                "interaction": "Perception Summary",
                "response": perception_response
            })
            
            # Parse perception from tags
            perception = self._parse_tag_content(perception_response, "perception")
            self.perception_summary = perception
            client.close()
        
        # Add human feedback to conversation if provided
        if human_feedback:
            feedback_message = f"""Your human observer has provided the following critical feedback: "{human_feedback}"
            
            Include this feedback in your status summary."""
            self.status_messages.append({"role": "user", "content": feedback_message})
            
            # Simple acknowledgment from assistant
            ack_response = "I understand the human feedback and will incorporate it into my status assessment."
            self.status_messages.append({"role": "assistant", "content": ack_response})
            
            self.log_chat("Human Feedback", [("user", feedback_message), ("assistant", ack_response)])
        
        # Interaction 2: Generate status summary with phase context
        user_message = f"""Now, given your current task: '{self.mission}'
Your timeline: Past/Completed actions: {self.past_options if self.past_options else 'None'}, Current/Ongoing actions: {self.options[0].description if self.options else 'None'}, Planned future actions: {self.options[1:3] if len(self.options) > 1 else 'None'}

Provide your status summary using the following tags:

<summary>
A concise summary of your progress and situation, including what you've accomplished and what you're currently doing.
</summary>
<percent_complete>
A number from 0 to 100 representing your estimated percent already complete on your current task ('{self.mission}'). 
</percent_complete>
<urgent>
Any important information for your manager, such as unexpected fires, civilians, or issues. If none, write "None".
</urgent>

Example response:
<summary>
I have moved to coordinates (5,10) and completed the scouting of the northern area. I found no fire in this sector and am ready to proceed to the next assigned location.
</summary>
<percent_complete>
75
</percent_complete>
<urgent>
I have located the missing group of civilians at coordinates (5,10). I suggest we move to the next phase of transporting them.
</urgent>"""
        
        self.status_messages.append({"role": "user", "content": user_message})
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=self.status_messages,
            
        )
        global_data["api_calls"] += 1
        global_data["input_tokens"] += response.usage.prompt_tokens
        global_data["output_tokens"] += response.usage.completion_tokens
        
        status_response = response.choices[0].message.content
        self.log_chat("Status Summary", [("user", user_message), ("assistant", status_response)])
        self.status_messages.append({"role": "assistant", "content": status_response})
        
        # Store LLM response for human observation
        self.timestep_chats.append({
            "interaction": "Status Summary",
            "response": status_response
        })
        
        client.close()
        
        # Parse status summary from tags
        status_summary = self._parse_tag_content(status_response, "summary")
        percent_complete = self._parse_tag_content(status_response, "percent_complete")
        urgent = self._parse_tag_content(status_response, "urgent")
        
        # Store parsed information
        self.status_summary = status_summary
        self.percent_complete = float(percent_complete) if percent_complete and percent_complete.isdigit() else 0.0
        self.urgent = urgent if urgent != "None" else ""
        
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
                    "urgent": self.urgent
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

    async def async_status_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]], human_feedback: str = None):
        """
        Async version of status phase for worker agents
        """
        print(f"AGENT_{self.id}: Async Status Phase")
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

Your job is to analyze observations and provide status updates with structured output using tags."""
        
        self.status_messages = [{"role": "system", "content": system_message}]
        
        # Step 1: Generate perception (sequential within agent)
        if self.last_observation:
            user_message = f"""Here are your observations: {self._build_observation_string(agent_states, global_data)}

Create a detailed perception summary (<=100 words) using the following tags:

<perception>
A detailed summary of what you observe, including your location, surroundings, any fires and civilians. Count the spaces/characters to describe all fires and civilians if any in exact coordinates. Also note other agents in your vicinity, and your current status (carrying capacity, etc.). Focus on information relevant to your current task.
</perception>


DO NOT INCLUDE ANYTHING ABOUT YOUR MISSION/TASK YET. ONLY PROVIDE A PERCEPTION SUMMARY.

Example response:
<perception>
I am currently at coordinates (5,10) in a medium forest cell. I can see dense forest to the north, a water source to the east at (7,10), and no signs of fire in my immediate vicinity. There are no civilians nearby. I am not carrying any civilians and have 3/5 water remaining. I can see AGENT_2 at coordinates (6,9) to the northeast.
</perception>"""
            
            self.status_messages.append({"role": "user", "content": user_message})
            
            perception_response = await self._async_openai_call(self.status_messages, global_data)
            self.log_chat("Perception Summary", [("system", system_message), ("user", user_message), ("assistant", perception_response)])
            self.status_messages.append({"role": "assistant", "content": perception_response})
            
            # Store LLM response for human observation
            self.timestep_chats.append({
                "interaction": "Perception Summary",
                "response": perception_response
            })



            # Parse perception from tags
            perception = self._parse_tag_content(perception_response, "perception")
            self.perception_summary = perception
        
        # Add human feedback to conversation if provided
        if human_feedback:
            feedback_message = f"""Your human observer has provided the following critical feedback: "{human_feedback}"
            
            Include this feedback in your status summary."""
            self.status_messages.append({"role": "user", "content": feedback_message})
            
            # Simple acknowledgment from assistant
            ack_response = "I understand the human feedback and will incorporate it into my status assessment."
            self.status_messages.append({"role": "assistant", "content": ack_response})
            
            self.log_chat("Human Feedback", [("user", feedback_message), ("assistant", ack_response)])
        
        # Step 2: Generate status summary with phase context (sequential, depends on step 1)
        user_message = f"""Now, given your current task: '{self.mission}'
Your timeline: Past/Completed actions: {self.past_options if self.past_options else 'None'}, Current/Ongoing actions: {self.options[0].description if self.options else 'None'}, Planned future actions: {self.options[1:3] if len(self.options) > 1 else 'None'}

Provide your status summary using the following tags:

<summary>
A concise summary of your progress and situation, including what you've accomplished and what you're currently doing.
</summary>
<percent_complete>
A number from 0 to 100 representing your estimated percent already complete on your current task ('{self.mission}'). 
</percent_complete>
<urgent>
Any important information for your manager, such as unexpected fires, civilians, or issues. If none, write "None".
</urgent>

Example response:
<summary>
I have moved to coordinates (5,10) and completed the scouting of the northern area. I found no fire in this sector and am ready to proceed to the next assigned location.
</summary>
<percent_complete>
75
</percent_complete>
<urgent>
I have located the missing group of civilians at coordinates (5,10). I suggest we move to the next phase of transporting them.
</urgent>"""
        
        self.status_messages.append({"role": "user", "content": user_message})
        
        status_response = await self._async_openai_call(self.status_messages, global_data)
        self.log_chat("Status Summary", [("user", user_message), ("assistant", status_response)])
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
                    "urgent": self.urgent
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


# -----------------------------------------------------------------------------
# WorkerAgent.action_phase - DEPRECATED (use async_action_phase instead)
# Original location: worker_agent.py lines 553-588
# -----------------------------------------------------------------------------

def action_phase(self, global_data: dict, agent_states: Dict[int, Tuple[int, int]]):
    """
    Enhanced action phase with mission/phase awareness
    """
    if not self.alive:
        return

    print(f"AGENT_{self.id}: Action Phase, Status: {self.status}")

    # Only execute if status is 0 (need new task) or 5 (new mission)
    if self.status == 5:
        self.options = []
        self.past_options = []
        self.action_queue = []
        self.last_action = []
        self.memory_buffer = []
        self.percent_complete = 0
        self._generate_options(global_data)
        self.status = 1  # Now working on task

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
                    "options_list": options_list
                }
            )
        except Exception as e:
            print(f"Warning: Failed to log worker action phase event: {e}")


# -----------------------------------------------------------------------------
# WorkerAgent._generate_options - DEPRECATED (use _async_generate_options instead)
# Original location: worker_agent.py lines 647-669
# -----------------------------------------------------------------------------

def _generate_options(self, global_data: dict):
    """
    Generate action options using the option libraries from wildfire_alg
    """
    # Step 1: Generate option sequence (untranslated option strings)
    option_sequence: OptionSequence = request_options(self, global_data)

    # Step 2: Translate options into structured format
    options: Options = translate_options(self, option_sequence, global_data)

    # Step 3: Store the translated options
    self.options = options.actions
    self.past_options = []  # Reset past options for new task

    print(f"AGENT_{self.id}: Generated {len(self.options)} options")

    # Log the options for debugging
    for i, option in enumerate(self.options):
        print(f"  Option {i+1}: {option.description}")


# ============================================================================
# DEPRECATED METHODS FROM utils.py
# ============================================================================

# -----------------------------------------------------------------------------
# request_options - DEPRECATED (use async_request_options instead)
# Original location: utils.py lines 399-540
# -----------------------------------------------------------------------------

def request_options(agent, global_data: dict) -> OptionSequence:
    """
    Generate action options for an agent based on their current task and observations.
    
    Args:
        agent: The agent object containing task, observations, and configuration
        global_data: Global data containing API tracking information
        
    Returns:
        OptionSequence: A sequence of action descriptions and reasoning
    """
    

    
    # Load manager prompt for option generation

    agent_type_string = "Unknown"
    if hasattr(agent, 'type'):
        type_map = {0: "firefighter", 1: "bulldozer", 2: "drone", 3: "helicopter", -1: "manager"}
        agent_type_string = type_map.get(agent.type, "Unknown")
    
    
    try:
        prompt_path = os.path.join("algorithms", "WILDFIRE", "prompts", "planner", f"{agent_type_string}_planner.txt")
        with open(prompt_path, 'r') as file:
            prompt_content = file.read()
    except FileNotFoundError:
        # Fallback prompt if file not found
        map_range = getattr(agent, 'map_range', 5)  # Default map range
        last_position = getattr(agent, 'last_position', (0, 0))  # Default position
        observations = getattr(agent, 'observations', "No observations available")
        current_cell = getattr(agent, 'last_current_cell', "unknown")
        current_task = getattr(agent, 'mission', "No task assigned")
        
        prompt_content = f"""You are a {agent_type_string} agent within a forest cell grid world. 
        The map is made up of {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} cells/grids with coordinates in the range of (0 to {agent.cfg.envs.map_size-1}, 0 to {agent.cfg.envs.map_size-1}) inclusive with the top left corner of the map being (0,0).
        
        You will receive observations of the map through the form of a 2d array of codes, where each cell is represented by an character corresponding to the type of terrain:
            0: brush (no trees)
            1: light forest (1 tree)
            2: medium forest (2 trees)
            3: dense forest (3 trees)
            i: Ignited
            f: On Fire
            e: Extinguishing
            x: Fully Extinguished
        
        You are only given information in a small square centered around you, representing a limited view/minimap. Your current location is {last_position} and thus your observations will be the range ({last_position[0]-map_range} to {last_position[0]+map_range}, {last_position[1]-map_range} to {last_position[1]+map_range}). Here is the mini-map view of the environment:
        
        {observations}
        
        IGNORE ALL "-". Those are unrevealed cells. They will reveal themselves when you get closer to them.
        
        The cell with the "*" is the current cell you are in. It is a {current_cell} cell.
        
        Now your task is the following:
        
        {current_task}
        
        Your job is to break down your high-level task, if needed, into a sequence of actions that you can perform. You may make as many or as few actions as you need.

        Respond using the following tag format:

        <actions>
        1. First action description
        2. Second action description
        </actions>
        <reasonings>
        1. Reasoning for first action
        2. Reasoning for second action
        </reasonings>

        Both lists should have the same number of items."""

    # Replace placeholders in prompt
    prompt_content = prompt_content.replace("MAPSIZE-1", str(agent.cfg.envs.map_size-1))
    prompt_content = prompt_content.replace("MAPSIZE", str(agent.cfg.envs.map_size))
    prompt_content = prompt_content.replace("POSITION", str(getattr(agent, 'last_position', (0, 0))))
    prompt_content = prompt_content.replace("MAPRANGE", f"({getattr(agent, 'last_position', (0, 0))[0]-getattr(agent, 'map_range', 5)} to {getattr(agent, 'last_position', (0, 0))[0]+getattr(agent, 'map_range', 5)}, {getattr(agent, 'last_position', (0, 0))[1]-getattr(agent, 'map_range', 5)} to {getattr(agent, 'last_position', (0, 0))[1]+getattr(agent, 'map_range', 5)})")
    prompt_content = prompt_content.replace("OBS", getattr(agent, 'perception_summary', "No observations available"))
    prompt_content = prompt_content.replace("CURRCELL", getattr(agent, 'last_current_cell', "unknown"))
    prompt_content = prompt_content.replace("TASK", getattr(agent, 'mission', "No task assigned"))
    prompt_content = prompt_content.replace("OBSRANGE", str(getattr(agent, 'map_range', 5)))

    # Call OpenAI to generate options
    client = OpenAI(api_key=agent.api_key)

    option_sequence = None
    max_retries = 3
    retry_count = 0

    system_message = """You are a highly trained agent within a grid forest world. Your job is to break down a task into smaller actions to be performed by the agent.

Respond using the following tag format:

<actions>
1. First action description
2. Second action description
</actions>
<reasonings>
1. Reasoning for first action
2. Reasoning for second action
</reasonings>

Both lists should have the same number of items."""

    while not option_sequence and retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0,
                max_tokens=4096,
            )

            response_content = response.choices[0].message.content
            option_sequence = _parse_option_sequence_from_tags(response_content)

            # Validate parsed result
            if not option_sequence.actions:
                raise ValueError("No actions parsed from response")

            agent.log_chat("Generate Options", [("user", prompt_content), ("assistant", response_content)])
            global_data["api_calls"] += 1
            global_data["input_tokens"] += response.usage.prompt_tokens
            global_data["output_tokens"] += response.usage.completion_tokens

        except Exception as e:
            print(f"Error generating options for agent {agent.id}, retry {retry_count + 1}: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                # Create a default option sequence
                option_sequence = OptionSequence(
                    actions=["Move to current position and wait for instructions"],
                    reasonings=["Default action due to generation failure"]
                )

    client.close()
    return option_sequence


# -----------------------------------------------------------------------------
# translate_options - DEPRECATED (use async_translate_options instead)
# Original location: utils.py lines 686-779
# -----------------------------------------------------------------------------

def translate_optionsdef translate_options(agent, option_sequence: OptionSequence, global_data: dict) -> Options:
    """
    Translate option sequence into structured Option objects.

    Args:
        agent: The agent object
        option_sequence: The raw option sequence from request_options
        global_data: Global data containing API tracking information

    Returns:
        Options: Structured options for the agent
    """

    # Determine agent type string
    if agent.type == 0:
        type_string = 'firefighter'
    elif agent.type == 1:
        type_string = 'bulldozer'
    elif agent.type == 2:
        type_string = 'drone'
    else:
        type_string = 'helicopter'

    system_message = f"""You are the controller of a highly trained {type_string} agent within a grid forest world.
Your job is to convert a sequence of string actions into a structured format for robotic control.

Respond using the following tag format for each action:

<options>
<option>
<type>action_type_number</type>
<param_1>first_parameter</param_1>
<param_2>second_parameter</param_2>
<description>description of the action</description>
</option>
</options>

Include one <option> block for each action."""

    # Load translator prompt
    prompt_path = os.path.join("algorithms", "WILDFIRE", "prompts", "translator", f"{type_string}_translator.txt")
    optionstring = "\n".join(option_sequence.actions)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().replace("ACTIONS", str(optionstring))
    except FileNotFoundError:
        print(f"Warning: Prompt file not found for {type_string}, using default")
        prompt = f""

    # Add tag format instructions to the prompt
    tag_instructions = """

Respond using the following tag format for each action:

<options>
<option>
<type>action_type_number</type>
<param_1>first_parameter</param_1>
<param_2>second_parameter</param_2>
<description>description of the action</description>
</option>
</options>

Include one <option> block for each action."""
    prompt = prompt + tag_instructions

    # Call OpenAI
    client = OpenAI(api_key=agent.api_key)
    response = client.chat.completions.create(
        model='gpt-4.1-nano',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
    )

    global_data["api_calls"] += 1
    global_data["input_tokens"] += response.usage.prompt_tokens
    global_data["output_tokens"] += response.usage.completion_tokens

    response_content = response.choices[0].message.content
    agent.log_chat("Translate Options", [("user", prompt), ("assistant", response_content)])

    options = _parse_options_from_tags(response_content)

    # Fallback: if no options parsed, create default
    if not options.actions:
        print(f"Warning: No options parsed for agent {agent.id}, creating default")
        options = Options(actions=[Option(type=0, param_1=0, param_2=0, description="Wait for instructions")])

    client.close()
    return options



'''

"""
Agent snapshot dataclass for the ORCH algorithm.

Provides frozen snapshots of agent state that are safe to read concurrently
from the chat worker without risk of reading mid-mutation data.

Snapshots are written after each asyncio.gather() barrier in the main loop,
so they are always coherent.
"""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class AgentSnapshot:
    agent_id: int
    agent_type: str  # "manager", "worker", "horizontal_manager"
    team_name: str
    agent_name: str
    is_leader: bool
    mission: str
    current_phase: Optional[str]
    phase_history: List[str]
    future_phases: List[str]
    phase_progress: int
    status: int
    perception_summary: str
    status_summary: str
    past_options: List[Any]  # last 5 from self.past_options
    memory_buffer: List[str]  # last 10 from self.memory_buffer (workers only)
    children_summaries: List[dict]  # list of child status dicts for managers
    timestep: int
    snapshot_time: float


def take_snapshot(agent: Any, timestep: int) -> AgentSnapshot:
    """
    Build a snapshot from a live agent object.

    Only reads stable fields that are set once per phase and never mutated
    mid-phase. Safe to call after asyncio.gather() barriers.

    Args:
        agent: A live Agent, WorkerAgent, or HorizontalManagerAgent instance
        timestep: Current game timestep

    Returns:
        An AgentSnapshot with all stable fields copied
    """
    # Determine agent_type by class name
    class_name = type(agent).__name__
    if class_name == "HorizontalManagerAgent":
        agent_type = "horizontal_manager"
    elif class_name == "WorkerAgent":
        agent_type = "worker"
    else:
        agent_type = "manager"

    # Build children summaries for manager agents
    children_summaries = []
    if hasattr(agent, "children"):
        for child in agent.children:
            children_summaries.append(
                {
                    "name": getattr(child, "name", ""),
                    "mission": getattr(child, "mission", ""),
                    "status_summary": getattr(child, "status_summary", ""),
                    "percent_complete": getattr(child, "percent_complete", 0.0),
                    "urgent": getattr(child, "urgent", ""),
                }
            )

    # Collect past_options (last 5)
    raw_past_options = getattr(agent, "past_options", [])
    past_options = list(raw_past_options[-5:]) if raw_past_options else []

    # Collect memory_buffer (last 10, workers only)
    raw_memory = getattr(agent, "memory_buffer", [])
    memory_buffer = list(raw_memory[-10:]) if raw_memory else []

    return AgentSnapshot(
        agent_id=agent.id,
        agent_type=agent_type,
        team_name=getattr(agent, "team_name", ""),
        agent_name=agent.name,
        is_leader=getattr(agent, "leader", False),
        mission=getattr(agent, "mission", ""),
        current_phase=getattr(agent, "current_phase", None),
        phase_history=list(getattr(agent, "phase_history", [])),
        future_phases=list(getattr(agent, "future_phases", [])),
        phase_progress=int(getattr(agent, "phase_progress", 0)),
        status=getattr(agent, "status", 0),
        perception_summary=getattr(agent, "perception_summary", ""),
        status_summary=getattr(agent, "status_summary", ""),
        past_options=past_options,
        memory_buffer=memory_buffer,
        children_summaries=children_summaries,
        timestep=timestep,
        snapshot_time=time.time(),
    )

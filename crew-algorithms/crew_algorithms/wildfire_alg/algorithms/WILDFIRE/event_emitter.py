"""
Event emitter for WILDFIRE algorithm activity events.

Appends structured events to /tmp/events_{lobby_id}.jsonl with sequential IDs.
Thread-safe per-lobby sequence counter using asyncio locks.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, Dict

# Global state per lobby
_event_seq: Dict[str, int] = {}
_event_locks: Dict[str, asyncio.Lock] = {}

# Valid event types
EVENT_TYPES = {
    "status_started",
    "perception_done",
    "slow_feedback_applied",
    "fast_feedback_injected",
    "decision_made",
    "action_started",
    "plan_written",
    "phase_transition",
    "options_written",
    "fast_feedback_rerun",
    "children_cancelled",
    "question_answered",
    "announcement",
    "agent_destroyed",
}


def _get_lock(lobby_id: str) -> asyncio.Lock:
    """Get or create a per-lobby asyncio lock."""
    if lobby_id not in _event_locks:
        _event_locks[lobby_id] = asyncio.Lock()
    return _event_locks[lobby_id]


def emit_event(
    lobby_id: str,
    event_type: str,
    agent_id: int,
    agent_name: str,
    timestep: int,
    detail: str,
    data: Optional[dict] = None,
) -> None:
    """
    Append event to /tmp/events_{lobby_id}.jsonl.

    Uses a synchronous increment (safe from a single asyncio task context).
    For concurrent access, use emit_event_async instead.

    Args:
        lobby_id: The game lobby identifier
        event_type: One of the defined event type strings
        agent_id: Numeric agent ID
        agent_name: Human-readable agent name
        timestep: Current game timestep
        detail: Human-readable description of the event
        data: Optional extra structured data dict
    """
    if lobby_id not in _event_seq:
        _event_seq[lobby_id] = 0

    _event_seq[lobby_id] += 1
    seq = _event_seq[lobby_id]

    event = {
        "seq": seq,
        "event_type": event_type,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "timestep": timestep,
        "timestamp": datetime.now().isoformat(),
        "detail": detail,
        "data": data or {},
    }

    events_file = os.path.join(tempfile.gettempdir(), f"events_{lobby_id}.jsonl")
    try:
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Warning: Failed to write event to {events_file}: {e}")


async def emit_event_async(
    lobby_id: str,
    event_type: str,
    agent_id: int,
    agent_name: str,
    timestep: int,
    detail: str,
    data: Optional[dict] = None,
) -> None:
    """
    Async version of emit_event with lock-protected seq increment.

    Use this when multiple concurrent tasks may emit events for the same lobby.
    """
    lock = _get_lock(lobby_id)
    async with lock:
        emit_event(
            lobby_id, event_type, agent_id, agent_name, timestep, detail, data
        )

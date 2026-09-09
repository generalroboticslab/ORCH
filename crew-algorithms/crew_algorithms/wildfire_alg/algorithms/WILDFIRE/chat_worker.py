"""
Chat worker for the WILDFIRE algorithm.

Async task that runs concurrently with the game loop.
Polls /tmp/chat_in_{lobby_id}.jsonl and processes human messages by:
  - Classifying them as question / slow_feedback / fast_feedback
  - Routing appropriately to agent queues or generating direct answers
  - Writing responses to /tmp/chat_out_{lobby_id}.jsonl
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Literal

from .agent_snapshot import AgentSnapshot
from .event_emitter import emit_event


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM outputs
# ---------------------------------------------------------------------------


class MessageClassification(BaseModel):
    type: Literal["question", "slow_feedback", "fast_feedback"]
    reasoning: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_new_messages(chat_in: str, seen_ids: set) -> list:
    """Read all unprocessed lines from the chat_in file."""
    messages = []
    try:
        with open(chat_in, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_id = msg.get("id")
                    if msg_id and msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        messages.append(msg)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[ChatWorker] Error reading chat_in: {e}")
    return messages


def _write_response(
    chat_out: str,
    msg_id: str,
    response_type: str,
    content: str,
    target_agent_id: int,
) -> None:
    """Append a response record to chat_out JSONL."""
    record = {
        "in_reply_to": msg_id,
        "type": response_type,
        "content": content,
        "target_agent_id": target_agent_id,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(chat_out, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[ChatWorker] Error writing to chat_out: {e}")


def _log_chat(output_path: str, entry: dict) -> None:
    """Append a structured log entry to {output_path}/chat_worker/chats.txt.

    Creates the directory if it does not already exist.  Each call appends a
    single JSON line so the file is easy to stream-process later.

    Args:
        output_path: The base results directory for this run (e.g. the ``path``
            variable in ``__main__.py``).
        entry: Arbitrary dict that will be serialised as a JSON line.
    """
    try:
        log_dir = Path(output_path) / "chat_worker"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "chats.txt"
        with open(log_file, "a") as _f:
            _f.write(json.dumps(entry) + "\n")
    except Exception as _e:
        print(f"[ChatWorker] Failed to write chat log: {_e}")


def _build_snapshot_context(snapshot: Optional[AgentSnapshot]) -> str:
    """Format a snapshot into a readable context string for LLM prompts."""
    if snapshot is None:
        return "No agent snapshot available."

    lines = [
        f"Agent: {snapshot.agent_name} (ID: {snapshot.agent_id}, type: {snapshot.agent_type})",
        f"Team: {snapshot.team_name}",
        f"Is team leader: {snapshot.is_leader}",
        f"Mission: {snapshot.mission}",
        f"Current phase: {snapshot.current_phase}",
        f"Phase history: {snapshot.phase_history}",
        f"Future phases: {snapshot.future_phases}",
        f"Phase progress: {snapshot.phase_progress}%",
        f"Status code: {snapshot.status}",
        f"Perception summary: {snapshot.perception_summary}",
        f"Status summary: {snapshot.status_summary}",
        f"Recent actions: {[str(o) for o in snapshot.past_options]}",
    ]
    if snapshot.memory_buffer:
        lines.append(f"Memory buffer (recent): {snapshot.memory_buffer[-5:]}")
    if snapshot.children_summaries:
        lines.append("Children:")
        for c in snapshot.children_summaries:
            lines.append(
                f"  - {c['name']}: {c['status_summary']} "
                f"({c['percent_complete']}% complete, urgent: {c['urgent']})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call helpers (use AsyncOpenAI directly — same pattern as agents)
# ---------------------------------------------------------------------------

MODEL = "gpt-5.4"
MODEL_LIGHT = "gpt-5.4-nano"


MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_LIGHT = "Qwen/Qwen3.6-35B-A3B"

llm_url = "http://localhost:8000/v1"


async def _classify_message(
    content: str,
    snapshot: Optional[AgentSnapshot],
    phase_tracker: dict,
    client: AsyncOpenAI,
) -> MessageClassification:
    """
    Classify a human message as question / slow_feedback / fast_feedback.
    Returns MessageClassification with .type and .reasoning.
    Falls back to slow_feedback on parse error.
    """
    agent_context = _build_snapshot_context(snapshot)
    is_leader = snapshot.is_leader if snapshot else False
    current_phase = phase_tracker.get("current_phase", "unknown")

    system = (
        "You are a classifier for human messages directed at AI agents in a wildfire "
        "management simulation. Classify the message as one of:\n"
        "- question: The human is asking for information about the agent's state, "
        "  plans, or observations.\n"
        "- slow_feedback: Minor suggestions, tips, or observations that do not "
        "  require changing the current plan (e.g. 'good job', 'be careful').\n"
        "- fast_feedback: The human wants to change priorities, redirect agents, "
        "  assign new targets, alter strategy, or give any directive that requires "
        "  replanning. This includes orders like 'go to X', 'focus on Y', "
        "  'stop doing Z', 'prioritize A over B'. Only valid for the root team "
        "  leader (is_leader=True). For non-leaders, treat as slow_feedback.\n\n"
        "When in doubt between slow_feedback and fast_feedback, prefer fast_feedback.\n\n"
        "Respond ONLY with valid JSON matching this schema:\n"
        '{"type": "question"|"slow_feedback"|"fast_feedback", "reasoning": "..."}'
    )
    user = (
        f"Current game phase: {current_phase}\n"
        f"Target agent context:\n{agent_context}\n\n"
        f"Human message: \"{content.upper()}\"\n\n"
        "Classify this message. If the target is not the root leader, fast_feedback "
        "is not valid — use slow_feedback instead."
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "classification",
                    "schema": MessageClassification.model_json_schema(),
                },
            },
        )
        raw = response.choices[0].message.content
        classification = MessageClassification.model_validate_json(raw)
        # Enforce: non-leaders cannot receive fast_feedback
        if classification.type == "fast_feedback" and not is_leader:
            classification.type = "slow_feedback"
            classification.reasoning += " (downgraded: target is not root leader)"
        return classification
    except Exception as e:
        print(f"[ChatWorker] Classification failed: {e}, defaulting to slow_feedback")
        return MessageClassification(
            type="slow_feedback", reasoning=f"Classification error: {e}"
        )


async def _answer_question(
    content: str,
    snapshot: Optional[AgentSnapshot],
    client: AsyncOpenAI,
) -> str:
    """Generate a direct answer to a question from the snapshot context."""
    agent_context = _build_snapshot_context(snapshot)

    system = (
        "You are a helpful assistant answering questions from a human operator about "
        "an AI agent in a wildfire management simulation. Use the agent state below "
        "to answer the question accurately and concisely (2-4 sentences)."
    )
    user = (
        f"Agent state:\n{agent_context}\n\n"
        f"Human question: \"{content}\""
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or "I could not generate an answer."
    except Exception as e:
        print(f"[ChatWorker] Answer generation failed: {e}")
        return f"Sorry, I could not generate an answer at this time: {e}"


async def _generate_preview(
    content: str,
    snapshot: Optional[AgentSnapshot],
    client: AsyncOpenAI,
) -> str:
    """Generate a brief preview of how feedback will be applied."""
    agent_context = _build_snapshot_context(snapshot)

    system = (
        "You are a helpful assistant in a wildfire management simulation. "
        "A human has sent feedback to an AI agent. Generate a brief (1-2 sentence) "
        "preview confirming receipt and predicting how the feedback will influence "
        "the agent's next decision based on its current state."
    )
    user = (
        f"Agent state:\n{agent_context}\n\n"
        f"Human feedback: \"{content}\""
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or (
            "Received. Your feedback will be incorporated at the next decision cycle."
        )
    except Exception as e:
        print(f"[ChatWorker] Preview generation failed: {e}")
        return "Received. Your feedback will be incorporated at the next decision cycle."


async def _generate_fast_feedback_response(
    content: str,
    snapshot: Optional[AgentSnapshot],
    client: AsyncOpenAI,
) -> str:
    """Generate response after fast feedback has been processed."""
    if snapshot is None:
        return "Urgent directive received and will be processed."

    system = (
        "You are a helpful assistant summarizing an AI manager's response to an "
        "urgent human directive in a wildfire simulation. Be concise (2-3 sentences)."
    )
    user = (
        f"The manager just processed an urgent directive.\n"
        f"Original directive: \"{content}\"\n\n"
        f"Manager's updated state:\n"
        f"  Status summary: {snapshot.status_summary}\n"
        f"  Current phase: {snapshot.current_phase}\n"
        f"  Phase progress: {snapshot.phase_progress}%\n"
        f"  Mission: {snapshot.mission}\n\n"
        "Summarize the manager's response to the urgent directive for the human operator."
    )

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or (
            "Urgent directive processed. Manager has updated its plan accordingly."
        )
    except Exception as e:
        print(f"[ChatWorker] Fast feedback response generation failed: {e}")
        return "Urgent directive processed. Manager has updated its plan accordingly."


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


async def _handle_message(
    msg: dict,
    lobby_id: str,
    agent_snapshots: Dict[int, AgentSnapshot],
    slow_status_queues: Dict[int, asyncio.Queue],
    fast_feedback_queues: Dict[int, asyncio.Queue],
    phase_tracker: dict,
    chat_out: str,
    client: AsyncOpenAI,
    output_path: str,
) -> None:
    """Classify and route a single incoming human message."""
    target_id = msg.get("target_agent_id")
    msg_id = msg.get("id", "")
    content = msg.get("content", "")

    snapshot = agent_snapshots.get(target_id) if target_id is not None else None
    timestep = phase_tracker.get("timestep", 0)

    classification = await _classify_message(
        content, snapshot, phase_tracker, client
    )
    msg_type = classification.type
    print(
        f"[CHAT] Message classified as '{msg_type}' for agent {target_id}: "
        f'"{content[:60]}" (reason: {classification.reasoning[:80]})'
    )

    response_content: str = ""

    if msg_type == "question":
        answer = await _answer_question(content, snapshot, client)
        response_content = answer
        _write_response(chat_out, msg_id, "question_answer", answer, target_id)
        emit_event(
            lobby_id,
            "question_answered",
            agent_id=target_id or -1,
            agent_name=snapshot.agent_name if snapshot else "unknown",
            timestep=timestep,
            detail=f"Answered question: {content[:60]}",
        )

    elif msg_type == "slow_feedback":
        if target_id is not None and target_id in slow_status_queues:
            await slow_status_queues[target_id].put(content)
        preview = await _generate_preview(content, snapshot, client)
        response_content = preview
        _write_response(chat_out, msg_id, "slow_feedback_preview", preview, target_id)

    elif msg_type == "fast_feedback":
        is_leader = snapshot.is_leader if snapshot else False
        if is_leader and target_id is not None and target_id in fast_feedback_queues:
            await fast_feedback_queues[target_id].put(content)
            print(
                f"[CHAT] Fast feedback QUEUED for agent {target_id} "
                f"(queue size now: {fast_feedback_queues[target_id].qsize()})"
            )
            # Send immediate ack only — the real report is generated in
            # __main__.py after the agent actually processes the directive
            # and snapshots are updated with the new state.
            response_content = "Thanks for the feedback! This looks important \u2014 I'm going to factor this into my planning and may adjust the team's approach."
            _write_response(
                chat_out,
                msg_id,
                "fast_feedback_queued",
                response_content,
                target_id,
            )
        else:
            # Downgrade to slow feedback
            if target_id is not None and target_id in slow_status_queues:
                await slow_status_queues[target_id].put(content)
            preview = await _generate_preview(content, snapshot, client)
            response_content = preview
            _write_response(
                chat_out, msg_id, "slow_feedback_preview", preview, target_id
            )

    # Persist a structured log entry for this message.
    _log_chat(
        output_path,
        {
            "timestamp": datetime.now().isoformat(),
            "timestep": timestep,
            "msg_id": msg_id,
            "target_agent_id": target_id,
            "content": content,
            "classification_type": msg_type,
            "classification_reasoning": classification.reasoning,
            "response_content": response_content,
        },
    )


# ---------------------------------------------------------------------------
# Main chat worker coroutine
# ---------------------------------------------------------------------------


async def run_chat_worker(
    lobby_id: str,
    agent_snapshots: Dict[int, AgentSnapshot],
    slow_status_queues: Dict[int, asyncio.Queue],
    fast_feedback_queues: Dict[int, asyncio.Queue],
    phase_tracker: dict,
    stop_event: asyncio.Event,
    api_key: str,
    cfg: object,
    output_path: str = "",
) -> None:
    """
    Run concurrently with game loop. Polls chat_in file and processes messages.

    Args:
        lobby_id: Game lobby identifier
        agent_snapshots: Shared dict of latest AgentSnapshot per agent_id
        slow_status_queues: Per-agent queues drained at status phase injection
        fast_feedback_queues: Per-manager queues for urgent fast feedback
        phase_tracker: Shared dict with current_phase, timestep, manager_decision_done
        stop_event: Set to signal this worker to stop
        api_key: OpenAI API key
        cfg: Hydra config object
        output_path: Base results directory for this run; chat logs are written to
            ``{output_path}/chat_worker/chats.txt``.  Defaults to empty string
            (logging disabled when no path is provided).
    """
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=3600,
    )
    seen_ids: set = set()

    chat_in = os.path.join(tempfile.gettempdir(), f"chat_in_{lobby_id}.jsonl")
    chat_out = os.path.join(tempfile.gettempdir(), f"chat_out_{lobby_id}.jsonl")

    print(f"[ChatWorker] Started for lobby {lobby_id}")

    while not stop_event.is_set():
        await asyncio.sleep(0.5)

        if not os.path.exists(chat_in):
            continue

        new_messages = _read_new_messages(chat_in, seen_ids)
        for msg in new_messages:
            asyncio.create_task(
                _handle_message(
                    msg,
                    lobby_id,
                    agent_snapshots,
                    slow_status_queues,
                    fast_feedback_queues,
                    phase_tracker,
                    chat_out,
                    client,
                    output_path,
                )
            )

    print(f"[ChatWorker] Stopped for lobby {lobby_id}")

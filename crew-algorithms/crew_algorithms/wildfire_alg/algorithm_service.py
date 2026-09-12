"""
Algorithm Service for ORCH
Runs the ORCH algorithm and provides HTTP API for communication
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import threading
import time
import json
import os
import sys
import subprocess
import signal
import tempfile
import random
import base64
from datetime import datetime

# Add crew_algorithms to path - support both Docker and host environments
script_dir = os.path.dirname(os.path.abspath(__file__))
crew_algorithms_root = os.path.dirname(os.path.dirname(script_dir))  # Go up to crew_algorithms
sys.path.insert(0, crew_algorithms_root)
print(f"Added to Python path: {crew_algorithms_root}")

app = FastAPI()

# Global state for managing game sessions
active_games: Dict[str, Dict[str, Any]] = {}
game_processes: Dict[str, subprocess.Popen] = {}
game_threads: Dict[str, threading.Thread] = {}


# Pydantic models
class StartGameRequest(BaseModel):
    lobby_id: str
    lobby_config: Dict[str, Any]

class SubmitActionsBatchRequest(BaseModel):
    lobby_id: str
    actions: Dict[str, List[int]]  # role -> action
    timestep: int = 0

class SendMessageRequest(BaseModel):
    lobby_id: str
    chat_id: str
    sender_id: int
    message: str
    timestamp: Optional[str] = None

class CreateChatsRequest(BaseModel):
    lobby_id: str
    chats: Dict[str, List[int]]  # chat_id -> list of participant agent_ids

class SubmitFeedbackRequest(BaseModel):
    lobby_id: str
    agent_id: int
    feedback: str
    timestep: int
    player_name: str


class ChatMessageRequest(BaseModel):
    lobby_id: str
    player_name: str
    target_agent_id: int
    content: str
    message_id: str  # UUID from frontend

class GameStatus(BaseModel):
    lobby_id: str
    status: str
    active: bool

# Game state storage
game_observations: Dict[str, Dict[str, Any]] = {}  # lobby_id -> {role -> observations}
game_chats: Dict[str, Dict[str, Dict[str, Any]]] = {}  # lobby_id -> {chat_id -> {participants: [agent_ids], messages: [...]}}


@app.post("/start_game")
async def start_game(request: StartGameRequest):
    """Start an ORCH game session"""
    try:
        if request.lobby_id in active_games:
            raise HTTPException(status_code=400, detail="Game already running")
        
        # Initialize game session
        active_games[request.lobby_id] = {
            "status": "starting",
            "lobby_config": request.lobby_config,
            "active": True,
            "started_at": datetime.now()
        }
        
        # Initialize game state
        game_observations[request.lobby_id] = {}
        game_chats[request.lobby_id] = {}
        
        # Start ORCH process in background thread
        thread = threading.Thread(
            target=run_wildfire_game,
            args=(request.lobby_id, request.lobby_config),
            daemon=True
        )
        game_threads[request.lobby_id] = thread
        thread.start()
        
        active_games[request.lobby_id]["status"] = "running"
        return {"status": "success", "message": "Game started"}
        
    except Exception as e:
        if request.lobby_id in active_games:
            active_games[request.lobby_id]["status"] = "error"
            active_games[request.lobby_id]["active"] = False
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")

def run_wildfire_game(lobby_id: str, lobby_config: Dict[str, Any]):
    """Run the ORCH algorithm in a separate process"""
    try:
        print(f"[DEBUG] Lobby config: {lobby_config}")
        # Update status
        active_games[lobby_id]["status"] = "running"
        
        # Prepare team_config override string if present
        team_config_override = None
        if "team_config" in lobby_config:
            # Convert dict to OmegaConf/Hydra override string
            tc = lobby_config["team_config"]
            # Format: {humans: [1, 2, 3], managers: {4: [1, 2, 3]}}
            def dict_to_omegaconf(d):
                if isinstance(d, dict):
                    items = []
                    for k, v in d.items():
                        items.append(f"{k}: {dict_to_omegaconf(v)}")
                    return "{" + ", ".join(items) + "}"
                elif isinstance(d, list):
                    return "[" + ", ".join(dict_to_omegaconf(x) for x in d) + "]"
                elif isinstance(d, str):
                    return f'"{d}"'
                else:
                    return str(d)
            team_config_override = f"++envs.team_config={dict_to_omegaconf(tc)}"
        
        # Determine working directory and paths
        work_dir = os.path.dirname(os.path.abspath(__file__))  # Current wildfire_alg directory
        python_path = crew_algorithms_root
        
        # Get collaboration mode from lobby config
        collaboration_mode = lobby_config.get('collaboration_mode', 'human_control')
        print(f"[DEBUG] Collaboration mode: {collaboration_mode}")
        
        cmd = [
            sys.executable, "-m", "crew_algorithms.wildfire_alg.algorithms.ORCH",
            f"envs.level={lobby_config.get('level', 'level1')}",
            f"envs.seed={lobby_config.get('seed', 2351)}",
            f"envs.collaboration_mode={collaboration_mode}",
            f"envs.lobby_id={lobby_id}"
        ]
        if team_config_override:
            cmd.append(team_config_override)
        
        print(f"Starting ORCH for lobby {lobby_id}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Working directory: {work_dir}")
        print(f"Python path: {python_path}")
        if team_config_override:
            print(f"Team config override: {team_config_override}")
        
        # Start the process in its own session so os.killpg can kill Unity too
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            env=dict(os.environ, PYTHONPATH=python_path),
            preexec_fn=os.setsid
        )
        
        game_processes[lobby_id] = process
        
        # Stream output in real-time
        def stream_output(stream, label):
            for line in iter(stream.readline, ''):
                print(f"{label}: {line.rstrip()}")
        
        # Start threads to capture output
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "STDOUT"))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "STDERR"))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        print(f"ORCH for lobby {lobby_id} completed with return code: {return_code}")
        
        # Update status
        if lobby_id in active_games:
            active_games[lobby_id]["status"] = "finished"
            active_games[lobby_id]["active"] = False
        
        # Clean up
        if lobby_id in game_processes:
            del game_processes[lobby_id]
        
    except Exception as e:
        print(f"Error running ORCH for lobby {lobby_id}: {e}")
        if lobby_id in active_games:
            active_games[lobby_id]["status"] = "error"
            active_games[lobby_id]["active"] = False

@app.post("/submit_actions_batch")
async def submit_actions_batch(request: SubmitActionsBatchRequest):
    """Submit batch of human actions to the algorithm"""
    print(f"[DEBUG] Received actions for lobby {request.lobby_id} timestep {request.timestep}: {request.actions}")
    try:
        if request.lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        if not active_games[request.lobby_id]["active"]:
            raise HTTPException(status_code=400, detail="Game not active")
        timestep = request.timestep
        actions_file = os.path.join(tempfile.gettempdir(), f"actions_{request.lobby_id}_{timestep}.json")
        try:
            with open(actions_file, "w") as f:
                json.dump(request.actions, f)
            print(f"[DEBUG] Wrote actions to {actions_file}: {request.actions}")
        except Exception as e:
            print(f"[DEBUG] Failed to write actions file: {e}")
        active_games[request.lobby_id]["last_action_time"] = datetime.now()
        print(f"Received actions for lobby {request.lobby_id} timestep {timestep}: {request.actions}")
        return {"status": "success", "message": "Actions received and stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit actions: {str(e)}")

@app.post("/submit_feedback")
async def submit_feedback(request: SubmitFeedbackRequest):
    """
    Submit human feedback for an agent.

    DEPRECATED: Use POST /send_chat instead.
    Kept for backward compatibility — writes to chat_in_{lobby_id}.jsonl using a
    generated message ID so the new chat worker can process it.
    """
    print(
        f"[DEBUG] (deprecated) Received feedback for lobby {request.lobby_id} "
        f"agent {request.agent_id} timestep {request.timestep}: {request.feedback}"
    )
    try:
        if request.lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        if not active_games[request.lobby_id]["active"]:
            raise HTTPException(status_code=400, detail="Game not active")

        import uuid as _uuid
        msg = {
            "id": str(_uuid.uuid4()),
            "target_agent_id": request.agent_id,
            "player_name": request.player_name,
            "content": request.feedback,
            "timestamp": datetime.now().isoformat(),
        }
        chat_in = os.path.join(tempfile.gettempdir(), f"chat_in_{request.lobby_id}.jsonl")
        try:
            with open(chat_in, "a") as f:
                f.write(json.dumps(msg) + "\n")
            print(f"[DEBUG] Appended (deprecated feedback) to {chat_in}: {msg}")
        except Exception as e:
            print(f"[DEBUG] Failed to write chat_in file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to write feedback: {e}")

        return {"status": "success", "message": "Feedback received and stored (via chat system)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.post("/send_chat")
async def send_chat(request: ChatMessageRequest):
    """Write a human chat message to /tmp/chat_in_{lobby_id}.jsonl for the chat worker."""
    try:
        if request.lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        if not active_games[request.lobby_id]["active"]:
            raise HTTPException(status_code=400, detail="Game not active")

        chat_in = os.path.join(
            tempfile.gettempdir(), f"chat_in_{request.lobby_id}.jsonl"
        )
        msg = {
            "id": request.message_id,
            "target_agent_id": request.target_agent_id,
            "player_name": request.player_name,
            "content": request.content,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(chat_in, "a") as f:
                f.write(json.dumps(msg) + "\n")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write message: {e}")

        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send chat: {str(e)}")


@app.get("/get_chat")
async def get_chat(lobby_id: str, since_id: str = ""):
    """
    Return chat responses from /tmp/chat_out_{lobby_id}.jsonl.

    since_id: ISO timestamp string; return only responses with timestamp > since_id.
    If omitted, returns all responses.
    """
    chat_out = os.path.join(tempfile.gettempdir(), f"chat_out_{lobby_id}.jsonl")
    responses = []
    if os.path.exists(chat_out):
        try:
            with open(chat_out, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if not since_id or r.get("timestamp", "") > since_id:
                            responses.append(r)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[DEBUG] Error reading chat_out: {e}")
    return {"responses": responses}


@app.get("/events")
async def get_events(lobby_id: str, since_seq: int = 0):
    """
    Return events from /tmp/events_{lobby_id}.jsonl with seq > since_seq.
    """
    events_file = os.path.join(tempfile.gettempdir(), f"events_{lobby_id}.jsonl")
    events = []
    if os.path.exists(events_file):
        try:
            with open(events_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        if e.get("seq", 0) > since_seq:
                            events.append(e)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[DEBUG] Error reading events file: {e}")
    latest = events[-1]["seq"] if events else since_seq
    return {"events": events, "latest_seq": latest}


@app.get("/phase_status")
async def get_phase_status(lobby_id: str):
    """
    Return current phase status for a lobby from /tmp/phase_status_{lobby_id}.json.
    """
    status_file = os.path.join(tempfile.gettempdir(), f"phase_status_{lobby_id}.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DEBUG] Error reading phase_status: {e}")
    return {"current_phase": "idle", "timestep": 0}

@app.get("/observations_batch/{lobby_id}")
async def get_observations_batch(lobby_id: str):
    """Get observations for all human players"""
    try:
        if lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        if not active_games[lobby_id]["active"]:
            return {"error": "Game not active"}
        
        # Try to read observations from file
        obs_file = os.path.join(tempfile.gettempdir(), f"observations_{lobby_id}.json")
        try:
            if os.path.exists(obs_file):
                with open(obs_file, "r") as f:
                    observations = json.load(f)
                
                # Process observations to convert image paths to base64
                #print(f"[DEBUG] Observations: {observations}")
                
                processed_observations = {}
                
                def find_latest_image(base_path, agent_name, image_type):
                    """Find the latest image (minimap or POV) for an agent"""
                    try:
                        # Convert relative path to absolute path from algorithm service perspective
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        if not os.path.isabs(base_path):
                            # If base_path is relative, make it relative to algorithm service directory
                            abs_base_path = os.path.join(script_dir, base_path)
                        else:
                            abs_base_path = base_path
                            
                        agent_image_dir = os.path.join(abs_base_path, agent_name, image_type)
                        #print(f"[DEBUG] Searching for {image_type} at {agent_image_dir}")
                        if os.path.exists(agent_image_dir):
                            #print(f"[DEBUG] {image_type} path exists")
                            # Find all PNG files and get the one with highest capture number
                            image_files = [f for f in os.listdir(agent_image_dir) if f.endswith('.png')]
                            if image_files:
                                # Sort by capture number (assuming format like capture_N.png)
                                def extract_capture_number(filename):
                                    try:
                                        import re
                                        match = re.search(r'capture_(\d+)\.png', filename)
                                        return int(match.group(1)) if match else 0
                                    except:
                                        return 0
                                
                                latest_image = max(image_files, key=extract_capture_number)
                                #print(f"[DEBUG] Found {image_type} at time {latest_image}")
                                return os.path.join(agent_image_dir, latest_image)
                        else:
                            print(f"[DEBUG] {image_type} path does not exist")
                    except Exception as e:
                        print(f"[DEBUG] Error finding {image_type} for {agent_name}: {e}")
                    return None

                def concatenate_minimap_pov(base_path, agent_name):
                    """Concatenate minimap and POV horizontally for an agent"""
                    try:
                        # Unity saves to Agent_{id}/Minimap/ and Agent_{id}/POV/
                        unity_agent_name = agent_name.replace("AGENT_", "Agent_")
                        minimap_path = find_latest_image(base_path, unity_agent_name, "Minimap")
                        pov_path = find_latest_image(base_path, unity_agent_name, "POV")
                        
                        if minimap_path and pov_path and os.path.exists(minimap_path) and os.path.exists(pov_path):
                            from PIL import Image
                            
                            # Load both images
                            minimap_img = Image.open(minimap_path)
                            pov_img = Image.open(pov_path)
                            
                            # Concatenate horizontally (minimap + POV)
                            total_width = minimap_img.width + pov_img.width
                            max_height = max(minimap_img.height, pov_img.height)
                            
                            combined = Image.new('RGB', (total_width, max_height))
                            combined.paste(minimap_img, (0, 0))
                            combined.paste(pov_img, (minimap_img.width, 0))
                            
                            # Close original images
                            minimap_img.close()
                            pov_img.close()
                            
                            #print(f"[DEBUG] Successfully combined minimap+POV for {agent_name}")
                            return combined
                        else:
                            print(f"[DEBUG] Missing images for {agent_name} - minimap: {bool(minimap_path)}, POV: {bool(pov_path)}")
                    except Exception as e:
                        print(f"[DEBUG] Error combining images for {agent_name}: {e}")
                    return None
                    
                
                for agent_id, obs_data in observations.items():
                    processed_obs = obs_data.copy()
                    
                    # Handle manager agents - create new layout: accumulative map + children grid
                    # Use leaf_worker_ids for all leaf workers (handles multi-level hierarchies)
                    # Only show accumulative map for top-level manager
                    if obs_data.get('type') == -1 and 'leaf_worker_ids' in obs_data:
                        #print(f"[DEBUG] Processing manager agent {agent_id} with leaf workers: {obs_data['leaf_worker_ids']}")
                        try:
                            base_path = obs_data.get('base_path')
                            is_top_level = obs_data.get('is_top_level', False)

                            if base_path:
                                from PIL import Image
                                import io
                                import math

                                # 1. Load accumulative map (left side) - ONLY for top-level manager
                                accumulative_img = None
                                if is_top_level:
                                    try:
                                        # Convert relative path to absolute path
                                        script_dir = os.path.dirname(os.path.abspath(__file__))
                                        if not os.path.isabs(base_path):
                                            abs_base_path = os.path.join(script_dir, base_path)
                                        else:
                                            abs_base_path = base_path

                                        accumulative_dir = os.path.join(abs_base_path, "Server_Accumulative")
                                        if os.path.exists(accumulative_dir):
                                            # Find latest capture file
                                            image_files = [f for f in os.listdir(accumulative_dir) if f.startswith('capture_') and f.endswith('.png')]
                                            if image_files:
                                                def extract_capture_number(filename):
                                                    try:
                                                        import re
                                                        match = re.search(r'capture_(\d+)\.png', filename)
                                                        return int(match.group(1)) if match else 0
                                                    except:
                                                        return 0

                                                latest_file = max(image_files, key=extract_capture_number)
                                                accumulative_path = os.path.join(accumulative_dir, latest_file)
                                                accumulative_img = Image.open(accumulative_path)
                                                #print(f"[DEBUG] Loaded accumulative map: {latest_file}")
                                    except Exception as e:
                                        print(f"[DEBUG] Error loading accumulative map: {e}")

                                # 2. Collect leaf worker minimap+POV images (all workers in subtree, not just direct children)
                                children_combined_images = []
                                for worker_id in obs_data['leaf_worker_ids']:
                                    combined_img = concatenate_minimap_pov(base_path, f"AGENT_{worker_id}")
                                    if combined_img:
                                        children_combined_images.append(combined_img)
                                        #print(f"[DEBUG] Got combined image for worker {worker_id}")
                                    else:
                                        print(f"[DEBUG] No combined image for worker {worker_id}")
                                
                                # 3. Create children grid (right side) - prioritize vertical layout
                                children_grid_img = None
                                if children_combined_images:
                                    num_children = len(children_combined_images)
                                    
                                    # Calculate grid dimensions prioritizing vertical but avoiding extreme ratios
                                    # Find the most balanced grid that still prefers vertical layout
                                    best_cols = 1
                                    best_score = float('inf')
                                    
                                    for cols in range(1, num_children + 1):
                                        rows = math.ceil(num_children / cols)
                                        
                                        # Skip if too horizontal (more cols than rows)
                                        if cols > rows:
                                            continue
                                            
                                        # Calculate score: prefer fewer columns but penalize extreme ratios
                                        aspect_ratio = rows / cols  # How many times taller than wide
                                        
                                        # Penalize extreme aspect ratios (too tall and thin)
                                        if aspect_ratio > 3:  # More than 3:1 ratio is too extreme
                                            penalty = aspect_ratio * 2
                                        else:
                                            penalty = 1
                                        
                                        # Score: prefer fewer columns with reasonable aspect ratio
                                        score = cols * penalty
                                        
                                        if score < best_score:
                                            best_score = score
                                            best_cols = cols
                                    
                                    grid_cols = best_cols
                                    grid_rows = math.ceil(num_children / grid_cols)
                                    
                                    # Get child image dimensions (each child is minimap+POV horizontally combined)
                                    child_width = children_combined_images[0].width
                                    child_height = children_combined_images[0].height
                                    
                                    # Create grid
                                    grid_width = grid_cols * child_width
                                    grid_height = grid_rows * child_height
                                    children_grid_img = Image.new('RGB', (grid_width, grid_height), color='black')
                                    
                                    # Place children in grid (fill by rows, left to right, top to bottom)
                                    for i, child_img in enumerate(children_combined_images):
                                        row = i // grid_cols
                                        col = i % grid_cols
                                        x = col * child_width
                                        y = row * child_height
                                        children_grid_img.paste(child_img, (x, y))
                                        child_img.close()  # Close child image
                                    
                                    #print(f"[DEBUG] Created children grid: {grid_cols} cols × {grid_rows} rows for {num_children} children")
                                
                                # 4. Combine accumulative map (left) + children grid (right)
                                if accumulative_img or children_grid_img:
                                    # Determine final dimensions
                                    left_width = accumulative_img.width if accumulative_img else 0
                                    left_height = accumulative_img.height if accumulative_img else 0
                                    right_width = children_grid_img.width if children_grid_img else 0
                                    right_height = children_grid_img.height if children_grid_img else 0
                                    
                                    total_width = left_width + right_width
                                    max_height = max(left_height, right_height)
                                    
                                    # Create final manager view
                                    manager_view = Image.new('RGB', (total_width, max_height), color='black')
                                    
                                    # Paste accumulative map on left
                                    if accumulative_img:
                                        manager_view.paste(accumulative_img, (0, 0))
                                        accumulative_img.close()
                                    
                                    # Paste children grid on right
                                    if children_grid_img:
                                        manager_view.paste(children_grid_img, (left_width, 0))
                                        children_grid_img.close()
                                    
                                    # Convert to base64
                                    img_buffer = io.BytesIO()
                                    manager_view.save(img_buffer, format='PNG')
                                    img_data = img_buffer.getvalue()
                                    processed_obs['minimap_image_base64'] = base64.b64encode(img_data).decode('utf-8')
                                    #print(f"[DEBUG] Created manager view with accumulative map + children grid for agent {agent_id}")

                                    # 5. Compute coordinate metadata for hover-to-see-coordinates
                                    worker_positions = obs_data.get('worker_positions', {})
                                    map_size = obs_data.get('map_size', 100)

                                    # Compute accumulative map grid bounds
                                    acc_meta = None
                                    if accumulative_img is not None or left_width > 0:
                                        all_pos = [
                                            wp['position'] for wp in worker_positions.values()
                                            if wp.get('position')
                                        ]
                                        if all_pos:
                                            grid_xs = [p[0] for p in all_pos]
                                            grid_ys = [p[1] for p in all_pos]
                                            # Grid → world
                                            world_xs = [gx - (map_size - 1) / 2 for gx in grid_xs]
                                            world_zs = [(map_size - 1) / 2 - gy for gy in grid_ys]
                                            # Covering square (mirrors GameManager.ComputeCoveringSquare, margin=30)
                                            min_wx, max_wx = min(world_xs), max(world_xs)
                                            min_wz, max_wz = min(world_zs), max(world_zs)
                                            sq = max(max_wx - min_wx, max_wz - min_wz) + 60
                                            c_gx = (min_wx + max_wx) / 2 + (map_size - 1) / 2 + 1
                                            c_gy = (map_size - 1) / 2 - (min_wz + max_wz) / 2 + 1
                                            half = sq / 2
                                            acc_meta = {
                                                "pixel_x": 0,
                                                "pixel_y": 0,
                                                "pixel_width": left_width,
                                                "pixel_height": left_height,
                                                "bounds": {
                                                    "grid_min_x": c_gx - half,
                                                    "grid_max_x": c_gx + half,
                                                    "grid_min_y": c_gy - half,
                                                    "grid_max_y": c_gy + half,
                                                },
                                            }

                                    # Compute per-worker minimap bounds in the children grid
                                    grid_meta = None
                                    if children_grid_img is not None:
                                        worker_metas = []
                                        for wid in obs_data.get('leaf_worker_ids', []):
                                            wp = worker_positions.get(str(wid)) or worker_positions.get(wid, {})
                                            pos = wp.get('position')
                                            mr = wp.get('minimap_range', 10)
                                            bounds = None
                                            if pos:
                                                # -1 offset from Unity camera positioning
                                                agx, agy = pos[0], pos[1]
                                                bounds = {
                                                    "grid_min_x": agx - mr,
                                                    "grid_max_x": agx + mr,
                                                    "grid_min_y": agy - mr,
                                                    "grid_max_y": agy + mr,
                                                }
                                            worker_metas.append({
                                                "id": wid,
                                                "position": pos,
                                                "minimap_bounds": bounds,
                                            })
                                        grid_meta = {
                                            "pixel_x": left_width,
                                            "pixel_y": 0,
                                            "pixel_width": right_width,
                                            "pixel_height": right_height,
                                            "grid_cols": grid_cols,
                                            "grid_rows": grid_rows,
                                            "child_width": child_width,
                                            "child_height": child_height,
                                            "minimap_width": child_width // 2,
                                            "workers": worker_metas,
                                        }

                                    processed_obs['image_coord_metadata'] = {
                                        "type": "manager",
                                        "total_width": total_width,
                                        "total_height": max_height,
                                        "accumulative": acc_meta,
                                        "children_grid": grid_meta,
                                    }

                                    manager_view.close()
                            
                        except Exception as e:
                            print(f"[DEBUG] Error processing manager images for agent {agent_id}: {e}")
                    
                    # Handle worker agents - combine minimap + POV horizontally
                    elif obs_data.get('type') != -1 and 'base_path' in obs_data and 'id' in obs_data:
                        #print(f"[DEBUG] Processing worker agent {agent_id} ({obs_data['name']})")
                        try:
                            base_path = obs_data['base_path']
                            current_agent_id = obs_data['id']
                            combined_img = concatenate_minimap_pov(base_path, f"AGENT_{current_agent_id}")
                            
                            if combined_img:
                                # Convert to base64
                                import io
                                img_buffer = io.BytesIO()
                                combined_img.save(img_buffer, format='PNG')
                                img_data = img_buffer.getvalue()
                                processed_obs['minimap_image_base64'] = base64.b64encode(img_data).decode('utf-8')
                                #print(f"[DEBUG] Created combined minimap+POV for worker {agent_id}: {len(processed_obs['minimap_image_base64'])} chars")

                                # Compute coordinate metadata for hover
                                worker_pos = obs_data.get('last_position')
                                minimap_range = obs_data.get('minimap_range', 10)
                                mm_w = combined_img.width // 2  # minimap is left half
                                mm_h = combined_img.height
                                mm_bounds = None
                                if worker_pos:
                                    agx, agy = worker_pos[0], worker_pos[1]
                                    mm_bounds = {
                                        "grid_min_x": agx - minimap_range,
                                        "grid_max_x": agx + minimap_range,
                                        "grid_min_y": agy - minimap_range,
                                        "grid_max_y": agy + minimap_range,
                                    }
                                processed_obs['image_coord_metadata'] = {
                                    "type": "worker",
                                    "total_width": combined_img.width,
                                    "total_height": combined_img.height,
                                    "minimap": {
                                        "pixel_x": 0,
                                        "pixel_y": 0,
                                        "pixel_width": mm_w,
                                        "pixel_height": mm_h,
                                        "bounds": mm_bounds,
                                    },
                                    "pov": {
                                        "pixel_x": mm_w,
                                        "pixel_y": 0,
                                        "pixel_width": mm_w,
                                        "pixel_height": mm_h,
                                    },
                                }

                                combined_img.close()
                            else:
                                print(f"[DEBUG] No combined image for worker {agent_id}")
                        except Exception as e:
                            print(f"[DEBUG] Error converting image for worker {agent_id}: {e}")
                    
                    # Remove fields not needed in frontend
                    for field in ['base_path', 'children_names', 'worker_positions']:
                        if field in processed_obs:
                            del processed_obs[field]
                    
                    processed_observations[agent_id] = processed_obs
                
                #print(f"[DEBUG] Loaded and processed observations from file: {obs_file}")
                return processed_observations
            else:
                print(f"[DEBUG] Observations file does not exist: {obs_file}")
                return {}
        except Exception as e:
            print(f"[DEBUG] Could not load observations file: {e}")
            return {}
    except Exception as e:
        print(f"[DEBUG] Error in observations_batch: {e}")
        return {"error": f"Failed to get observations: {str(e)}"}

@app.post("/create_chats")
async def create_chats(request: CreateChatsRequest):
    """Create chats for a lobby"""
    try:
        if request.lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        if request.lobby_id not in game_chats:
            game_chats[request.lobby_id] = {}
        
        for chat_id, participants in request.chats.items():
            game_chats[request.lobby_id][chat_id] = {
                "participants": participants,
                "messages": []
            }
        
        return {"status": "success", "message": f"Created {len(request.chats)} chats"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chats: {str(e)}")

@app.post("/send_message")
async def send_message(request: SendMessageRequest):
    """Send a message to a specific chat"""
    try:
        if request.lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        if request.lobby_id not in game_chats:
            raise HTTPException(status_code=404, detail="No chats found for lobby")
        
        if request.chat_id not in game_chats[request.lobby_id]:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat = game_chats[request.lobby_id][request.chat_id]
        
        # Validate sender is participant
        if request.sender_id not in chat["participants"]:
            raise HTTPException(status_code=403, detail="Sender not participant in chat")
        
        # Add message
        chat["messages"].append({
            "sender_id": request.sender_id,
            "message": request.message,
            "timestamp": request.timestamp or datetime.now().isoformat()
        })
        
        return {"status": "success", "message": "Message sent"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.get("/chats/{lobby_id}")
async def get_chats(lobby_id: str):
    """Get all chats for a lobby"""
    try:
        if lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        chats = game_chats.get(lobby_id, {})
        return {"chats": chats}
        
    except Exception as e:
        return {"chats": {}}

@app.get("/game_status/{lobby_id}")
async def get_game_status(lobby_id: str):
    """Get status of a game session"""
    if lobby_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[lobby_id]
    return GameStatus(
        lobby_id=lobby_id,
        status=game["status"],
        active=game["active"]
    )

@app.post("/stop_game/{lobby_id}")
async def stop_game(lobby_id: str):
    """Stop a game session"""
    try:
        if lobby_id not in active_games:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Kill the entire process group (ORCH + Unity child) if running
        if lobby_id in game_processes:
            process = game_processes[lobby_id]
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # process already exited
            except Exception as e:
                print(f"Error killing process group for lobby {lobby_id}: {e}")
                try:
                    process.kill()
                except Exception:
                    pass
            del game_processes[lobby_id]
        
        # Update status
        active_games[lobby_id]["status"] = "stopped"
        active_games[lobby_id]["active"] = False
        
        return {"status": "success", "message": "Game stopped"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop game: {str(e)}")

@app.post("/shutdown")
async def shutdown():
    """Shutdown the algorithm service"""
    try:
        # Stop all running games
        for lobby_id in list(active_games.keys()):
            if active_games[lobby_id]["active"]:
                await stop_game(lobby_id)
        
        return {"status": "success", "message": "Algorithm service shutting down"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to shutdown: {str(e)}")

@app.get("/levels")
async def get_levels():
    """Return level metadata derived from build_config presets (single source of truth)."""
    from crew_algorithms.wildfire_alg.config.build_config import get_levels_for_frontend
    return {"levels": get_levels_for_frontend()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_games": len([g for g in active_games.values() if g["active"]]),
        "service": "ORCH Algorithm Service"
    }

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Port to run the service on")
    args = parser.parse_args()
    
    print(f"Starting ORCH Algorithm Service on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
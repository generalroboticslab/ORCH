#!/usr/bin/env python3
"""
Long-running test to wait for observations to populate and save debug output
"""
import requests
import time
import sys
import json
from datetime import datetime

def long_test():
    lobby_id = f"long_test_{int(time.time())}"
    
    # Open debug file
    debug_file = f"observations_debug_{lobby_id}.txt"
    
    with open(debug_file, 'w') as f:
        f.write(f"WILDFIRE Observations Debug Log\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Lobby ID: {lobby_id}\n")
        f.write("=" * 80 + "\n\n")
        
        start_request = {
            "lobby_id": lobby_id,
            "lobby_config": {
                "level": "Suppress_Fire_Extinguish",
                "seed": 42,
                "max_steps": 20,
                "human_agents": {},
                "hierarchy": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0]
                ]
            }
        }
        
        f.write(f"Start request: {json.dumps(start_request, indent=2)}\n\n")
        
        print(f"Starting long test: {lobby_id}")
        print(f"Debug output will be saved to: {debug_file}")
        
        response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=30)
        
        f.write(f"Start response: {response.status_code}\n")
        f.write(f"Start response body: {response.text}\n\n")
        
        if response.status_code != 200:
            print(f"Failed to start: {response.text}")
            f.write("FAILED TO START GAME\n")
            return False
        
        print("Game started! Waiting for observations...")
        f.write("Game started successfully!\n\n")
        
        # Wait up to 3 minutes checking every 5 seconds
        for i in range(36):
            time.sleep(5)
            
            f.write(f"=== Check {i+1} at {datetime.now()} ===\n")
            print(f"Check {i+1}...")
            
            # Check status first
            status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}", timeout=5)
            if status_resp.status_code == 200:
                status = status_resp.json()
                f.write(f"Status: {json.dumps(status, indent=2)}\n")
                print(f"  Status: {status.get('status', 'unknown')}, Active: {status.get('active', False)}")
            else:
                f.write(f"Status request failed: {status_resp.status_code}\n")
                status = {}
            
            # Check observations
            obs_resp = requests.get(f"http://localhost:8001/observations_batch/{lobby_id}", timeout=10)
            f.write(f"Observations response code: {obs_resp.status_code}\n")
            
            if obs_resp.status_code == 200:
                obs = obs_resp.json()
                f.write(f"Raw observations: {json.dumps(obs, indent=2)}\n")
                
                if isinstance(obs, dict):
                    if "error" in obs:
                        print(f"  Error: {obs['error']}")
                        f.write(f"Observations error: {obs['error']}\n")
                        if not status.get('active', False):
                            f.write("Game not active, breaking\n")
                            break
                    elif len(obs) == 0:
                        print("  Empty observations ({})")
                        f.write("Empty observations dict\n")
                    else:
                        print(f"  SUCCESS! Got observations for {len(obs)} agents")
                        f.write(f"SUCCESS: {len(obs)} agents with observations\n")
                        
                        # Print and save details for all agents
                        for agent_id, agent_obs in obs.items():
                            print(f"  Agent {agent_id}:")
                            f.write(f"\nAgent {agent_id} observations:\n")
                            
                            if isinstance(agent_obs, dict):
                                for key, value in agent_obs.items():
                                    if isinstance(value, list) and len(value) > 20:
                                        print(f"    {key}: [list with {len(value)} elements]")
                                        f.write(f"  {key}: [list with {len(value)} elements]\n")
                                        f.write(f"    First 5: {value[:5]}\n")
                                        f.write(f"    Last 5: {value[-5:]}\n")
                                    elif isinstance(value, str) and len(value) > 200:
                                        print(f"    {key}: [string with {len(value)} chars]")
                                        f.write(f"  {key}: [string with {len(value)} chars]\n")
                                        f.write(f"    First 100 chars: {value[:100]}\n")
                                    else:
                                        print(f"    {key}: {value}")
                                        f.write(f"  {key}: {value}\n")
                            else:
                                print(f"    Non-dict data: {type(agent_obs)} = {agent_obs}")
                                f.write(f"  Non-dict data: {type(agent_obs)} = {agent_obs}\n")
                        
                        f.write("\nFULL SUCCESS - OBSERVATIONS RECEIVED!\n")
                        return True
                else:
                    print(f"  Unexpected obs type: {type(obs)}")
                    f.write(f"Unexpected observation type: {type(obs)}\n")
            else:
                print(f"  Obs request failed: {obs_resp.status_code}")
                f.write(f"Observations request failed: {obs_resp.status_code} - {obs_resp.text}\n")
            
            f.write("\n")
            
            if not status.get('active', False):
                print("  Game ended")
                f.write("Game ended (not active)\n")
                break
        
        f.write(f"\nTest completed at {datetime.now()}\n")
        return False

if __name__ == "__main__":
    success = long_test()
    print(f"\nResult: {'SUCCESS' if success else 'TIMEOUT'}")
    
    # Show final container logs and save to file
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', 'wildfire-test-4'], 
                              capture_output=True, text=True, timeout=10)
        
        log_file = f"container_logs_{int(time.time())}.txt"
        with open(log_file, 'w') as f:
            f.write("WILDFIRE Container Logs\n")
            f.write("=" * 50 + "\n\n")
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\n")
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
        
        print(f"\nContainer logs saved to: {log_file}")
        
        # Also print recent logs to console
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("\nRecent container logs:")
            for line in lines[-20:]:
                if line.strip() and not line.startswith('INFO:     172.17.0.1'):
                    print(f"  {line}")
                    
    except Exception as e:
        print(f"Failed to get logs: {e}")
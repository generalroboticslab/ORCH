#!/usr/bin/env python3
"""
Test WILDFIRE with Nakama network connection
"""
import requests
import time
import json

def test_nakama_connection():
    lobby_id = f"nakama_test_{int(time.time())}"
    
    start_request = {
        "lobby_id": lobby_id,
        "lobby_config": {
            "level": "Suppress_Fire_Extinguish",
            "seed": 42,
            "max_steps": 10,
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
    
    print(f"Testing Nakama connection with lobby: {lobby_id}")
    
    # Start game
    response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=30)
    
    if response.status_code != 200:
        print(f"Failed to start: {response.text}")
        return
    
    print("Game started! Monitoring for debug output...")
    
    # Monitor for 60 seconds to see if we get past Unity initialization
    for i in range(12):  # 12 checks * 5 seconds = 60 seconds
        time.sleep(5)
        
        # Check status
        status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}")
        if status_resp.status_code == 200:
            status = status_resp.json()
            print(f"Check {i+1}: Status={status.get('status')}, Active={status.get('active')}")
            
            if not status.get('active', False):
                print("Game ended")
                break
        
        # Check observations
        obs_resp = requests.get(f"http://localhost:8001/observations_batch/{lobby_id}")
        if obs_resp.status_code == 200:
            obs = obs_resp.json()
            if isinstance(obs, dict) and len(obs) > 0:
                print(f"SUCCESS! Got observations for {len(obs)} agents")
                return True
    
    return False

if __name__ == "__main__":
    success = test_nakama_connection()
    
    print(f"\nResult: {'SUCCESS' if success else 'STILL WAITING'}")
    
    # Show container logs to see if debug statements appear
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '50', 'wildfire-test-5'], 
                              capture_output=True, text=True, timeout=10)
        
        print("\nContainer logs (looking for WILDFIRE debug output):")
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-30:]:
                if any(keyword in line for keyword in ['WILDFIRE_STDOUT', 'WILDFIRE_STDERR', 'made env', 'Environment reset', 'State keys', 'Agent.*observations']):
                    print(f"  DEBUG: {line}")
                elif 'STDOUT:' in line or 'STDERR:' in line:
                    print(f"  {line}")
                elif not line.startswith('INFO:     172.17.0.1'):
                    print(f"  {line}")
    except Exception as e:
        print(f"Failed to get logs: {e}")
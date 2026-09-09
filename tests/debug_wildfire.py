#!/usr/bin/env python3
"""
Debug script to test WILDFIRE execution directly in container
"""
import requests
import json
import time

def test_with_simple_config():
    """Test with minimal configuration"""
    print("Testing WILDFIRE with simple configuration...")
    
    start_request = {
        "lobby_id": "debug_test",
        "lobby_config": {
            "level": "Suppress_Fire_Extinguish", 
            "seed": 42,
            "max_steps": 5,  # Very short
            "human_agents": {},
            "hierarchy": [[0, 0, 0], [0, 0, 0], [1, 1, 0]],  # Minimal 3-agent setup
            "total_agents": 3,
            "agent_names": ["AGENT_1", "AGENT_2"], 
            "manager_names": ["AGENT_3"]
        }
    }
    
    print("Sending start_game request...")
    try:
        response = requests.post(
            "http://localhost:8001/start_game",
            json=start_request,
            timeout=30
        )
        
        if response.status_code == 200:
            print("SUCCESS: Game started!")
            print(f"Response: {response.json()}")
            
            # Wait and check status periodically
            for i in range(10):
                time.sleep(2)
                status_resp = requests.get(f"http://localhost:8001/game_status/debug_test")
                if status_resp.status_code == 200:
                    status = status_resp.json()
                    print(f"Status check {i+1}: {status}")
                    if not status.get("active", False):
                        break
                        
            # Try to get observations
            obs_resp = requests.get("http://localhost:8001/observations_batch/debug_test")
            if obs_resp.status_code == 200:
                obs = obs_resp.json()
                print(f"Final observations: {obs}")
            
        else:
            print(f"FAILED: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        
    # Get container logs
    print("\n" + "="*50)
    print("CONTAINER LOGS:")
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '20', 'wildfire-test'], 
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Failed to get logs: {e}")

if __name__ == "__main__":
    test_with_simple_config()
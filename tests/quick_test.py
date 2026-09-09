#!/usr/bin/env python3
"""
Quick test to see if absolute path fixes Unity launch
"""
import requests
import time

def quick_test():
    lobby_id = f"quick_test_{int(time.time())}"
    
    start_request = {
        "lobby_id": lobby_id,
        "lobby_config": {
            "level": "Suppress_Fire_Extinguish",
            "seed": 42,
            "max_steps": 5,
            "human_agents": {},
            "hierarchy": [[0, 0, 0], [0, 0, 0], [1, 1, 0]]
        }
    }
    
    print(f"Starting game: {lobby_id}")
    response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=20)
    
    if response.status_code == 200:
        print("Game started successfully!")
        
        # Wait for initialization
        for i in range(15):
            time.sleep(2)
            status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}")
            if status_resp.status_code == 200:
                status = status_resp.json()
                print(f"Check {i+1}: {status}")
                
                # Try observations
                obs_resp = requests.get(f"http://localhost:8001/observations_batch/{lobby_id}")
                if obs_resp.status_code == 200:
                    obs = obs_resp.json()
                    if isinstance(obs, dict) and len(obs) > 0 and "error" not in obs:
                        print(f"SUCCESS: Got observations with {len(obs)} agents!")
                        return True
                
                if not status.get("active", False):
                    print("Game ended")
                    break
    else:
        print(f"Failed to start: {response.status_code} - {response.text}")
    
    return False

if __name__ == "__main__":
    if quick_test():
        print("✅ SUCCESS!")
    else:
        print("❌ Still not working")
    
    # Show logs
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '50', 'wildfire-test-4'], 
                              capture_output=True, text=True, timeout=5)
        print("\nContainer logs:")
        if result.stdout:
            for line in result.stdout.split('\n')[-20:]:
                if line.strip():
                    print(f"  {line}")
    except:
        pass
#!/usr/bin/env python3
"""
Extended test to see if WILDFIRE eventually gets past Unity initialization
"""
import requests
import time
import subprocess

def test_longer_debug():
    lobby_id = f"long_debug_{int(time.time())}"
    
    start_request = {
        "lobby_id": lobby_id,
        "lobby_config": {
            "level": "Suppress_Fire_Extinguish",
            "seed": 42,
            "max_steps": 3,
            "human_agents": {},
            "hierarchy": [[0, 0, 0], [0, 0, 0], [1, 1, 0]]
        }
    }
    
    print(f"Starting longer debug test: {lobby_id}")
    
    # Start the game
    response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=30)
    
    if response.status_code != 200:
        print(f"Failed to start: {response.text}")
        return
    
    print("Game started! Monitoring for 2 minutes...")
    
    # Monitor for 2 minutes to see if it eventually gets past Unity
    last_logs_count = 0
    for i in range(24):  # 24 checks * 5 seconds = 2 minutes
        time.sleep(5)
        
        # Get container logs
        try:
            result = subprocess.run(['docker', 'logs', 'wildfire-test-6'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                
                # Check for new output
                if len(lines) > last_logs_count:
                    new_lines = lines[last_logs_count:]
                    last_logs_count = len(lines)
                    
                    print(f"Check {i+1}: Found {len(new_lines)} new lines")
                    
                    # Look for debug statements or other progress
                    for line in new_lines:
                        if '[DEBUG]' in line:
                            print(f"  DEBUG: {line}")
                        elif any(keyword in line for keyword in ['WILDFIRE_STDOUT', 'Environment reset', 'State keys', 'make_env']):
                            print(f"  PROGRESS: {line}")
                        elif 'STDOUT:' in line and 'memorysetup' not in line:
                            print(f"  STDOUT: {line}")
                else:
                    print(f"Check {i+1}: No new output")
        
        except Exception as e:
            print(f"Check {i+1}: Error getting logs: {e}")
        
        # Check game status
        status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}")
        if status_resp.status_code == 200:
            status = status_resp.json()
            if not status.get('active', False):
                print(f"Game ended with status: {status.get('status')}")
                break

    print("Test completed")

if __name__ == "__main__":
    test_longer_debug()
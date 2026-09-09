#!/usr/bin/env python3
"""
Simple test to get observations and save to file
"""
import requests
import json
from datetime import datetime

def simple_test():
    # Check existing games first
    response = requests.get("http://localhost:8001/health")
    print(f"Health check: {response.json()}")
    
    # Try the long_test that might still be running
    lobby_id = "long_test_1753120660"
    
    with open("simple_observations_debug.txt", "w") as f:
        f.write(f"Simple Observations Test\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Testing lobby: {lobby_id}\n")
        f.write("=" * 50 + "\n\n")
        
        # Check status
        status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}")
        f.write(f"Status response code: {status_resp.status_code}\n")
        if status_resp.status_code == 200:
            status = status_resp.json()
            f.write(f"Status: {json.dumps(status, indent=2)}\n\n")
            print(f"Game status: {status}")
        else:
            f.write(f"Status error: {status_resp.text}\n\n")
            print("No active game found")
            return
        
        # Get observations
        obs_resp = requests.get(f"http://localhost:8001/observations_batch/{lobby_id}")
        f.write(f"Observations response code: {obs_resp.status_code}\n")
        
        if obs_resp.status_code == 200:
            try:
                obs = obs_resp.json()
                f.write(f"Observations type: {type(obs)}\n")
                f.write(f"Observations length: {len(obs) if isinstance(obs, (dict, list)) else 'N/A'}\n")
                f.write(f"Full observations JSON:\n")
                f.write(json.dumps(obs, indent=2))
                f.write("\n\n")
                
                print(f"Observations: {type(obs)} with {len(obs) if isinstance(obs, (dict, list)) else 'N/A'} items")
                
                if isinstance(obs, dict) and obs:
                    f.write("DETAILED BREAKDOWN:\n")
                    f.write("-" * 30 + "\n")
                    
                    for agent_id, agent_data in obs.items():
                        f.write(f"\nAgent {agent_id}:\n")
                        f.write(f"  Type: {type(agent_data)}\n")
                        
                        if isinstance(agent_data, dict):
                            f.write(f"  Keys: {list(agent_data.keys())}\n")
                            for key, value in agent_data.items():
                                f.write(f"  {key}:\n")
                                f.write(f"    Type: {type(value)}\n")
                                if isinstance(value, (list, tuple)):
                                    f.write(f"    Length: {len(value)}\n")
                                    if len(value) > 0:
                                        f.write(f"    First item: {value[0]}\n")
                                        if len(value) > 5:
                                            f.write(f"    Sample: {value[:5]}...\n")
                                        else:
                                            f.write(f"    Full: {value}\n")
                                elif isinstance(value, str):
                                    f.write(f"    Length: {len(value)}\n")
                                    f.write(f"    Value: {value[:100]}{'...' if len(value) > 100 else ''}\n")
                                else:
                                    f.write(f"    Value: {value}\n")
                        else:
                            f.write(f"  Value: {agent_data}\n")
                        f.write("\n")
                    
                    print("SUCCESS: Found observation data! Check simple_observations_debug.txt for details")
                else:
                    print("Observations are empty or not a dict")
                    
            except Exception as e:
                f.write(f"Error parsing observations: {e}\n")
                f.write(f"Raw response: {obs_resp.text}\n")
                print(f"Error parsing observations: {e}")
        else:
            f.write(f"Observations request failed: {obs_resp.text}\n")
            print(f"Observations request failed: {obs_resp.status_code}")

if __name__ == "__main__":
    simple_test()
    print("Debug saved to simple_observations_debug.txt")
    print("Container logs saved to container_logs_full_debug.txt")
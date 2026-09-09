#!/usr/bin/env python3
"""
Persistent test to keep checking for observations until we get them
"""
import requests
import json
import time
import sys

ALGORITHM_SERVICE_URL = "http://localhost:8001"

def create_test_lobby_config():
    """Create test lobby configuration"""
    return {
        "level": "Suppress_Fire_Extinguish",
        "seed": 42,
        "max_steps": 50,  # Longer test to ensure we get observations
        "human_agents": {},
        "hierarchy": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 3
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 8
            [1, 1, 1, 1, 1, 1, 1, 1, 0]   # Agent 9 (manager)
        ]
    }

def start_game():
    """Start a new game"""
    lobby_id = f"persistent_test_{int(time.time())}"
    start_request = {
        "lobby_id": lobby_id,
        "lobby_config": create_test_lobby_config()
    }
    
    print(f"Starting game with lobby_id: {lobby_id}")
    response = requests.post(f"{ALGORITHM_SERVICE_URL}/start_game", json=start_request, timeout=30)
    
    if response.status_code == 200:
        print("Game started successfully!")
        return lobby_id
    else:
        print(f"Failed to start game: {response.status_code} - {response.text}")
        return None

def check_observations(lobby_id, attempt_num):
    """Check observations and return True if we get meaningful data"""
    try:
        response = requests.get(f"{ALGORITHM_SERVICE_URL}/observations_batch/{lobby_id}", timeout=10)
        
        if response.status_code == 200:
            observations = response.json()
            
            print(f"Attempt {attempt_num}: Observations received")
            
            if isinstance(observations, dict):
                if "error" in observations:
                    print(f"  Error: {observations['error']}")
                    return False
                elif len(observations) == 0:
                    print("  Empty observations dict")
                    return False
                else:
                    print(f"  SUCCESS: Got observations for {len(observations)} agents!")
                    
                    # Show details for first agent
                    for agent_id, obs_data in list(observations.items())[:1]:
                        print(f"  Agent {agent_id} observation keys: {list(obs_data.keys()) if isinstance(obs_data, dict) else 'Not a dict'}")
                        if isinstance(obs_data, dict):
                            for key, value in obs_data.items():
                                if isinstance(value, (list, tuple)):
                                    print(f"    {key}: [{len(value)} elements]")
                                elif isinstance(value, str) and len(value) > 100:
                                    print(f"    {key}: [long string, {len(value)} chars]")
                                else:
                                    print(f"    {key}: {value}")
                    
                    return True
            else:
                print(f"  Unexpected observation format: {type(observations)}")
                return False
        else:
            print(f"  Failed to get observations: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  Exception getting observations: {e}")
        return False

def get_game_status(lobby_id):
    """Get game status"""
    try:
        response = requests.get(f"{ALGORITHM_SERVICE_URL}/game_status/{lobby_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unknown", "active": False}
    except:
        return {"status": "error", "active": False}

def monitor_container_logs():
    """Show recent container logs"""
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '30', 'wildfire-test-3'], 
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            print("Recent STDOUT:")
            lines = result.stdout.split('\n')[-15:]  # Last 15 lines
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        if result.stderr:
            print("Recent STDERR:")
            lines = result.stderr.split('\n')[-10:]  # Last 10 lines
            for line in lines:
                if line.strip():
                    print(f"  {line}")
    except Exception as e:
        print(f"Failed to get logs: {e}")

def main():
    """Run persistent observation test"""
    print("WILDFIRE Persistent Observation Test")
    print("=" * 50)
    
    # Start game
    lobby_id = start_game()
    if not lobby_id:
        return False
    
    print(f"\nMonitoring observations for lobby: {lobby_id}")
    print("Checking every 3 seconds for up to 60 attempts (3 minutes)...")
    print()
    
    max_attempts = 60
    success = False
    
    for attempt in range(1, max_attempts + 1):
        # Check game status
        status = get_game_status(lobby_id)
        print(f"Attempt {attempt}: Game status = {status.get('status', 'unknown')}, Active = {status.get('active', False)}")
        
        # If game is not active, break
        if not status.get('active', False):
            print(f"Game is no longer active (status: {status.get('status')})")
            break
        
        # Check observations
        if check_observations(lobby_id, attempt):
            success = True
            break
        
        # Wait before next attempt
        if attempt < max_attempts:
            time.sleep(3)
    
    # Show final container logs
    print("\n" + "=" * 50)
    print("Final container logs:")
    monitor_container_logs()
    
    # Clean up
    try:
        requests.post(f"{ALGORITHM_SERVICE_URL}/stop_game/{lobby_id}", timeout=5)
    except:
        pass
    
    if success:
        print("\n🎉 SUCCESS: Received meaningful observations!")
        return True
    else:
        print(f"\n❌ No observations received after {max_attempts} attempts")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script for WILDFIRE algorithm service start_game endpoint
Tests environment creation and observation retrieval
"""

import requests
import json
import time
import sys
from typing import Dict, Any

# Algorithm service configuration
ALGORITHM_SERVICE_URL = "http://localhost:8001"

def create_test_lobby_config() -> Dict[str, Any]:
    """Create a minimal lobby configuration for testing"""
    return {
        "level": "Suppress_Fire_Extinguish",
        "seed": 42,
        "max_steps": 10,  # Short test
        "human_agents": {},  # No human agents for testing
        "hierarchy": [  # Simple hierarchy with one manager
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 1 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 2 (firefighter)  
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 3 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 4 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 5 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 6 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 7 (firefighter)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agent 8 (firefighter)
            [1, 1, 1, 1, 1, 1, 1, 1, 0]   # Agent 9 (manager)
        ],
        "total_agents": 9,
        "agent_names": ["AGENT_1", "AGENT_2", "AGENT_3", "AGENT_4", "AGENT_5", "AGENT_6", "AGENT_7", "AGENT_8"],
        "manager_names": ["AGENT_9"],
        "communication_mode": "team_chat"
    }

def test_health():
    """Test algorithm service health"""
    print("Testing algorithm service health...")
    try:
        response = requests.get(f"{ALGORITHM_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"Service healthy: {data}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_start_game():
    """Test start_game endpoint and environment creation"""
    print("\nTesting start_game endpoint...")
    
    lobby_id = "test_lobby_123"
    lobby_config = create_test_lobby_config()
    
    # Prepare start game request
    start_request = {
        "lobby_id": lobby_id,
        "lobby_config": lobby_config
    }
    
    print(f"Sending start_game request for lobby: {lobby_id}")
    print(f"   Level: {lobby_config['level']}")
    print(f"   Agents: {lobby_config['total_agents']}")
    print(f"   Max steps: {lobby_config['max_steps']}")
    
    try:
        # Start the game
        response = requests.post(
            f"{ALGORITHM_SERVICE_URL}/start_game",
            json=start_request,
            timeout=60  # Give it time to initialize
        )
        
        if response.status_code == 200:
            print("Game started successfully!")
            result = response.json()
            print(f"   Response: {result}")
            return lobby_id
        else:
            print(f"Start game failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Start game error: {e}")
        return None

def test_game_status(lobby_id: str):
    """Test game status endpoint"""
    print(f"\nTesting game status for lobby: {lobby_id}")
    
    try:
        response = requests.get(f"{ALGORITHM_SERVICE_URL}/game_status/{lobby_id}", timeout=10)
        
        if response.status_code == 200:
            status = response.json()
            print(f"Game status: {status}")
            return status
        else:
            print(f"Status check failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Status check error: {e}")
        return None

def test_observations(lobby_id: str):
    """Test observations endpoint to verify environment is running"""
    print(f"\nTesting observations for lobby: {lobby_id}")
    
    try:
        response = requests.get(f"{ALGORITHM_SERVICE_URL}/observations_batch/{lobby_id}", timeout=10)
        
        if response.status_code == 200:
            observations = response.json()
            print("Observations retrieved successfully!")
            
            # Check if we got observations
            if isinstance(observations, dict):
                if "error" in observations:
                    print(f"   Observation error: {observations['error']}")
                else:
                    print(f"   Observations for {len(observations)} agents")
                    
                    # Show details for first few agents
                    for i, (agent_id, obs) in enumerate(observations.items()):
                        if i >= 3:  # Limit output
                            break
                        print(f"   Agent {agent_id}:")
                        if isinstance(obs, dict):
                            for key, value in obs.items():
                                if isinstance(value, (list, tuple)) and len(value) > 10:
                                    print(f"     {key}: [{len(value)} elements]")
                                else:
                                    print(f"     {key}: {value}")
                        else:
                            print(f"     Data: {obs}")
            else:
                print(f"   Raw observations: {observations}")
                
            return observations
        else:
            print(f"Observations failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Observations error: {e}")
        return None

def test_stop_game(lobby_id: str):
    """Test stop game endpoint"""
    print(f"\nTesting stop_game for lobby: {lobby_id}")
    
    try:
        response = requests.post(f"{ALGORITHM_SERVICE_URL}/stop_game/{lobby_id}", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Game stopped: {result}")
            return True
        else:
            print(f"Stop game failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Stop game error: {e}")
        return False

def monitor_container_logs():
    """Show recent container logs to see WILDFIRE debug output"""
    print("\nRecent algorithm container logs:")
    import subprocess
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '50', 'wildfire-test'], 
                              capture_output=True, text=True, timeout=5)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except Exception as e:
        print(f"Failed to get logs: {e}")

def main():
    """Run all tests"""
    print("WILDFIRE Algorithm Service Test")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("Service not healthy, exiting")
        return False
    
    # Test 2: Start game
    lobby_id = test_start_game()
    if not lobby_id:
        print("Failed to start game")
        return False
    
    # Wait a bit for initialization
    print("\nWaiting 5 seconds for environment initialization...")
    time.sleep(5)
    
    # Test 3: Check status
    status = test_game_status(lobby_id)
    
    # Test 4: Get observations (this verifies environment is actually running)
    observations = test_observations(lobby_id)
    
    # Show container logs for debugging
    monitor_container_logs()
    
    # Test 5: Stop game
    test_stop_game(lobby_id)
    
    # Final assessment
    print("\n" + "=" * 50)
    if observations and not (isinstance(observations, dict) and "error" in observations):
        print("SUCCESS: Environment created and observations retrieved!")
        print("   The WILDFIRE algorithm container is working correctly.")
        return True
    else:
        print("PARTIAL SUCCESS: Game started but observations may be incomplete.")
        print("   Check the container logs above for WILDFIRE debug output.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
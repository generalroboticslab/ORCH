#!/usr/bin/env python3
"""
Test to find the exact point where WILDFIRE hangs
"""
import requests
import time
import subprocess

def test_exact_hang():
    lobby_id = f"hang_test_{int(time.time())}"
    
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
    
    print(f"Starting hang detection test: {lobby_id}")
    
    # Start the game
    response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=30)
    
    if response.status_code != 200:
        print(f"Failed to start: {response.text}")
        return
    
    print("Game started! Looking for debug statements...")
    
    # Check logs every 2 seconds for 30 seconds
    seen_debug_lines = set()
    
    for i in range(15):  # 15 checks * 2 seconds = 30 seconds
        time.sleep(2)
        
        # Get container logs
        try:
            result = subprocess.run(['docker', 'logs', 'wildfire-test-7'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                
                # Look for debug statements
                new_debug_found = False
                for line in lines:
                    if '[DEBUG]' in line and line not in seen_debug_lines:
                        print(f"NEW DEBUG: {line}")
                        seen_debug_lines.add(line)
                        new_debug_found = True
                
                if not new_debug_found:
                    # Look for other meaningful output
                    for line in lines[-5:]:
                        if any(keyword in line for keyword in ['STDOUT:', 'STDERR:', 'Unity', 'Command:', 'Starting WILDFIRE']):
                            if line not in seen_debug_lines:
                                print(f"OTHER: {line}")
                                seen_debug_lines.add(line)
        
        except Exception as e:
            print(f"Check {i+1}: Error getting logs: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("DEBUG STATEMENTS FOUND:")
    print("="*60)
    debug_statements = [line for line in seen_debug_lines if '[DEBUG]' in line]
    if debug_statements:
        for stmt in sorted(debug_statements):
            print(f"✓ {stmt}")
        
        # Determine where it likely hung
        if '[DEBUG] register_env_configs() completed' not in debug_statements:
            print("\n🔴 LIKELY HANG POINT: register_env_configs() call")
        elif '[DEBUG] Starting WILDFIRE algorithm initialization' not in debug_statements:
            print("\n🔴 LIKELY HANG POINT: Hydra main decorator or function start")
        else:
            print("\n🔴 HANG POINT: Later in initialization")
    else:
        print("❌ NO DEBUG STATEMENTS FOUND - Script never started executing")
        print("🔴 LIKELY HANG POINT: Before any Python code runs (subprocess/import issue)")

if __name__ == "__main__":
    test_exact_hang()
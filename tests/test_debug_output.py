#!/usr/bin/env python3
"""
Test to capture debug output and see exactly where WILDFIRE gets stuck
"""
import requests
import time
import subprocess

def test_debug_output():
    lobby_id = f"debug_test_{int(time.time())}"
    
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
    
    print(f"Starting debug test: {lobby_id}")
    
    # Start the game
    response = requests.post("http://localhost:8001/start_game", json=start_request, timeout=30)
    
    if response.status_code != 200:
        print(f"Failed to start: {response.text}")
        return
    
    print("Game started! Monitoring debug output...")
    
    # Monitor logs for 30 seconds to see where it gets stuck
    debug_lines = []
    
    for i in range(6):  # 6 checks * 5 seconds = 30 seconds
        time.sleep(5)
        
        # Get container logs
        try:
            result = subprocess.run(['docker', 'logs', 'wildfire-test-6'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                
                # Look for new DEBUG lines
                new_debug_lines = []
                for line in lines:
                    if '[DEBUG]' in line and line not in debug_lines:
                        new_debug_lines.append(line)
                        debug_lines.append(line)
                
                if new_debug_lines:
                    print(f"Check {i+1}: New debug output:")
                    for line in new_debug_lines:
                        print(f"  {line}")
                else:
                    print(f"Check {i+1}: No new debug output")
                
                # Also check for other relevant output
                for line in lines[-10:]:
                    if any(keyword in line for keyword in ['STDOUT:', 'STDERR:', 'Unity', 'make_env', 'Environment']) and line not in debug_lines:
                        print(f"  OTHER: {line}")
                        debug_lines.append(line)
        
        except Exception as e:
            print(f"Check {i+1}: Error getting logs: {e}")
        
        # Check game status
        status_resp = requests.get(f"http://localhost:8001/game_status/{lobby_id}")
        if status_resp.status_code == 200:
            status = status_resp.json()
            if not status.get('active', False):
                print("Game ended")
                break
    
    # Final log dump
    print("\n" + "="*50)
    print("FINAL DEBUG OUTPUT ANALYSIS:")
    print("="*50)
    
    try:
        result = subprocess.run(['docker', 'logs', 'wildfire-test-6'], 
                              capture_output=True, text=True, timeout=10)
        
        # Save full logs to file
        with open("debug_full_logs.txt", "w") as f:
            f.write("WILDFIRE Debug Full Logs\n")
            f.write("="*50 + "\n\n")
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\n")
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
        
        print("Full logs saved to debug_full_logs.txt")
        
        # Show all debug lines found
        if debug_lines:
            print(f"\nAll debug statements found ({len(debug_lines)}):")
            for line in debug_lines:
                if '[DEBUG]' in line:
                    print(f"  {line}")
        else:
            print("\nNO DEBUG STATEMENTS FOUND!")
            print("This means WILDFIRE script never started executing")
        
        # Show where it got stuck
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            last_meaningful_line = None
            for line in reversed(lines):
                if any(keyword in line for keyword in ['STDOUT:', 'STDERR:', '[DEBUG]', 'Unity', 'Starting WILDFIRE']):
                    last_meaningful_line = line
                    break
            
            if last_meaningful_line:
                print(f"\nLast meaningful output: {last_meaningful_line}")
        
    except Exception as e:
        print(f"Error in final analysis: {e}")

if __name__ == "__main__":
    test_debug_output()
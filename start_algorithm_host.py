#!/usr/bin/env python3
"""
Standalone Algorithm Service Runner
Runs the WILDFIRE algorithm service directly on the host (not containerized)
This allows containerized backend to communicate with host algorithm via HTTP
"""

import os
import sys
import subprocess
from pathlib import Path
import signal
import time

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    crew_algorithms_dir = script_dir / "crew-algorithms"
    algorithm_service_path = crew_algorithms_dir / "crew_algorithms" / "wildfire_alg" / "algorithm_service.py"
    
    if not algorithm_service_path.exists():
        print(f"Error: Algorithm service not found at {algorithm_service_path}")
        print(f"Expected location: {algorithm_service_path}")
        print(f"Current working directory: {script_dir}")
        sys.exit(1)
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(crew_algorithms_dir)
    
    # Check for required environment variables
    if 'OPENAI_API_KEY' not in env:
        print("Warning: OPENAI_API_KEY not set in environment")
        print("Set it with: export OPENAI_API_KEY=your_key_here")
    
    print("=" * 60)
    print("WILDFIRE Algorithm Service (Host Mode)")
    print("=" * 60)
    print(f"Working directory: {crew_algorithms_dir}")
    print(f"Service will run on: http://localhost:8001")
    print(f"Backend containers can connect via: http://host.docker.internal:8001")
    print("=" * 60)
    print("Press Ctrl+C to stop the service")
    print()
    
    process = None
    
    def signal_handler(signum, frame):
        print("\nShutting down algorithm service...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing algorithm service...")
                process.kill()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check if we can import fastapi in the current environment
        try:
            import fastapi
            print("* FastAPI found in current Python environment")
        except ImportError:
            print("X FastAPI not found in current Python environment")
            print("Please install dependencies:")
            print("  pip install fastapi uvicorn")
            print("  pip install -r crew-algorithms/crew_algorithms/wildfire_alg/requirements.txt")
            sys.exit(1)
        
        # Run the algorithm service
        print("Starting algorithm service...")
        process = subprocess.Popen([
            sys.executable, 
            str(algorithm_service_path)
        ], 
        cwd=str(crew_algorithms_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        
        print(f"Algorithm service started with PID: {process.pid}")
        
        # Monitor the process output for errors
        import threading
        import queue
        
        output_queue = queue.Queue()
        
        def read_output(pipe, label):
            for line in iter(pipe.readline, ''):
                if line:
                    output_queue.put((label, line.rstrip()))
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "OUT"), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "ERR"), daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Monitor initial startup
        startup_time = 0
        service_started = False
        
        while startup_time < 10:  # Wait up to 10 seconds for startup
            try:
                # Check for output
                while not output_queue.empty():
                    label, line = output_queue.get_nowait()
                    print(f"[{label}] {line}")
                    if "Uvicorn running on" in line or "Application startup complete" in line:
                        service_started = True
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"Process exited early with code: {process.poll()}")
                    break
                
                if service_started:
                    break
                    
                time.sleep(1)
                startup_time += 1
            except queue.Empty:
                time.sleep(0.1)
        
        if service_started:
            print("* Algorithm service is running and ready")
            print("Service is ready to accept connections from containerized backend")
            print()
        else:
            print("X Algorithm service may not have started properly")
            print("Check the output above for errors")
        
        # Continue monitoring output
        def monitor_output():
            while process.poll() is None:
                try:
                    while not output_queue.empty():
                        label, line = output_queue.get_nowait()
                        print(f"[{label}] {line}")
                except queue.Empty:
                    time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nAlgorithm service stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Algorithm service failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error running algorithm service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
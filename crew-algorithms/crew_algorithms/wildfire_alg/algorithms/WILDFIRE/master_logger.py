import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class MasterLogger:
    def __init__(self, log_dir: str = "outputs/master_logs"):
        """Initialize the master logger with both text and JSON output files."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        
        # Initialize log files
        self.text_log_path = os.path.join(log_dir, f"master_log_{timestamp}.txt")
        self.json_log_path = os.path.join(log_dir, f"master_log_{timestamp}.json")
        
        # Open text file for writing
        self.text_file = open(self.text_log_path, 'w', encoding='utf-8')
        self.json_entries = []
        
        # Write header
        self.text_file.write(f"MASTER LOG - Run: {self.run_id}\n")
        self.text_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.text_file.write("=" * 80 + "\n\n")
        
        print(f"Master logger initialized. Logs will be saved to:")
        print(f"  Text: {self.text_log_path}")
        print(f"  JSON: {self.json_log_path}")
    
    def log_event(self, timestep: int, agent_id: str, event_type: str, 
                  details: Dict[str, Any], phase: Optional[str] = None):
        """Log an event in both text and JSON formats."""
        # Create log entry
        entry = {
            "timestep": timestep,
            "agent_id": agent_id,
            "event_type": event_type,
            "phase": phase,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to JSON entries
        self.json_entries.append(entry)
        
        # Write to text file
        self._write_text_entry(entry)
        
        # Flush text file to ensure immediate writing
        self.text_file.flush()
    
    def _write_text_entry(self, entry: Dict[str, Any]):
        """Write a formatted text entry to the log file."""
        details = entry['details']
        
        if entry['event_type'] == "STATUS_PHASE":
            self._write_status_phase_condensed(entry, details)
        elif entry['event_type'] == "ACTION_PHASE":
            self._write_action_phase_condensed(entry, details)
        elif entry['event_type'] == "FINAL_ACTION":
            self._write_final_action_condensed(entry, details)
        else:
            self._write_feedback_event_condensed(entry, details)

        # Add divider after final action (execute) events
        if entry['event_type'] == "FINAL_ACTION":
            self.text_file.write("\n" + "=" * 80 + "\n")
        else:
            self.text_file.write("\n")
    
    def _write_status_phase_details(self, details: Dict[str, Any]):
        """Write status phase details in readable format."""
        if 'mission' in details:
            self.text_file.write(f"  Mission: {details['mission']}\n")
        if 'mission_percent' in details:
            self.text_file.write(f"  Mission Progress: {details['mission_percent']}%\n")
        if 'phase' in details:
            self.text_file.write(f"  Phase: {details['phase']}\n")
        if 'phase_percent' in details:
            self.text_file.write(f"  Phase Progress: {details['phase_percent']}%\n")
        if 'decision' in details:
            self.text_file.write(f"  Decision: {details['decision']}\n")
        if 'urgent' in details:
            self.text_file.write(f"  Urgent: {details['urgent']}\n")
    
    def _write_action_phase_details(self, details: Dict[str, Any]):
        """Write action phase details in readable format."""
        if 'decision' in details:
            self.text_file.write(f"  Decision: {details['decision']}\n")
        if 'mission' in details:
            self.text_file.write(f"  Mission: {details['mission']}\n")
        if 'phases' in details:
            self.text_file.write(f"  Phases: {', '.join(details['phases'])}\n")
        if 'task_designations' in details:
            self.text_file.write("  Task Designations:\n")
            for agent, task in details['task_designations'].items():
                self.text_file.write(f"    {agent}: {task}\n")
        if 'options_list' in details:
            self.text_file.write(f"  Options: {', '.join(details['options_list'])}\n")
    
    def _write_final_action_details(self, details: Dict[str, Any]):
        """Write final action tensor details in readable format."""
        if 'action_tensor' in details:
            self.text_file.write(f"  Action Tensor: {details['action_tensor']}\n")
        if 'action_description' in details:
            self.text_file.write(f"  Action: {details['action_description']}\n")
    
    def _write_status_phase_condensed(self, entry: Dict[str, Any], details: Dict[str, Any]):
        """Write condensed status phase details in one line."""
        line = f"T{entry['timestep']:3d} | AGENT_{entry['agent_id']} | STATUS"
        
        if 'mission' in details:
            mission_short = details['mission']
            line += f" | Mission: {mission_short}"
        
        if 'mission_percent' in details:
            line += f" | Progress: {details['mission_percent']}%"
        
        # For worker agents, show current options instead of phase
        if 'options_list' in details and details['options_list']:
            options_count = len(details['options_list'])
            current_option = details['options_list'][0]
            remaining = options_count - 1
            line += f" | Options: {current_option} (+{remaining} more)"
        elif 'phase' in details:
            phase_short = details['phase']
            line += f" | Phase: {phase_short}"
        
        if 'phase_percent' in details:
            line += f" | Phase: {details['phase_percent']}%"
        
        if 'decision' in details:
            line += f" | Decision: {details['decision']}"
        if 'urgent' in details:
            line += f" | Urgent: {details['urgent']}"
        
        self.text_file.write(line)
    
    def _write_action_phase_condensed(self, entry: Dict[str, Any], details: Dict[str, Any]):
        """Write condensed action phase details with full phases and task designations."""
        line1 = f"T{entry['timestep']:3d} | AGENT_{entry['agent_id']} | ACTION"
        
        if 'decision' in details:
            line1 += f" | {details['decision']}"
        
        if 'mission' in details:
            mission_short = details['mission']
            line1 += f" | Mission: {mission_short}"
        
        self.text_file.write(line1)
        
        # Show full phases
        if 'phases' in details and details['phases']:
            self.text_file.write(f"\n{'':>15} | Phases: {', '.join(details['phases'])}")
        
        # Show full task designations
        if 'task_designations' in details and details['task_designations']:
            self.text_file.write(f"\n{'':>15} | Tasks:")
            for agent, task in details['task_designations'].items():
                self.text_file.write(f"\n{'':>17} | {agent}: {task}")
        
        # Show generated options in detail
        if 'options_list' in details and details['options_list']:
            self.text_file.write(f"\n{'':>15} | Options:")
            for i, option in enumerate(details['options_list'], 1):
                self.text_file.write(f"\n{'':>17} | {i}. {option}")
        
        # Show generated phases in detail
        if 'new_phases_added' in details and details['new_phases_added']:
            self.text_file.write(f"\n{'':>15} | New Phases:")
            for i, phase in enumerate(details['new_phases_added'], 1):
                self.text_file.write(f"\n{'':>17} | {i}. {phase}")
    
    def _write_feedback_event_condensed(self, entry: Dict[str, Any], details: Dict[str, Any]):
        """Write condensed feedback/generic event details in one line."""
        line = f"T{entry['timestep']:3d} | AGENT_{entry['agent_id']} | {entry['event_type']}"
        for key, value in details.items():
            if key == "past_feedback":
                line += f" | past_feedback({len(value)} items)"
            elif key == "report":
                report_short =  str(value)
                line += f" | report: {report_short}"
            else:
                line += f" | {key}: {value}"
        self.text_file.write(line)

    def _write_final_action_condensed(self, entry: Dict[str, Any], details: Dict[str, Any]):
        """Write condensed final action details in one line."""
        line = f"T{entry['timestep']:3d} | ALL_AGENTS | EXECUTE"
        
        if 'action_tensor' in details:
            line += f" | {details['action_tensor']}"
        
        if 'action_description' in details:
            desc_short = details['action_description']
            line += f" | {desc_short}"

        
        
        self.text_file.write(line)
    
    def close(self):
        """Close the logger and save JSON file."""
        # Close text file
        self.text_file.write(f"\nRun completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.text_file.close()
        
        # Save JSON file
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "run_id": self.run_id,
                "start_time": self.json_entries[0]["timestamp"] if self.json_entries else None,
                "end_time": datetime.now().isoformat(),
                "total_events": len(self.json_entries),
                "events": self.json_entries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Master logger closed. Logged {len(self.json_entries)} events.")

# Global logger instance
master_logger = None

def init_master_logger(log_dir: str = "outputs/master_logs") -> MasterLogger:
    """Initialize the global master logger."""
    global master_logger
    master_logger = MasterLogger(log_dir)
    return master_logger

def get_master_logger() -> MasterLogger:
    """Get the global master logger instance."""
    global master_logger
    if master_logger is None:
        raise RuntimeError("Master logger not initialized. Call init_master_logger() first.")
    return master_logger 
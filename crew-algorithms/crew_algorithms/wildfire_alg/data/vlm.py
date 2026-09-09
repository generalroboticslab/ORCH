"""
VLM vs Perception Module Comparison Framework

This module compares two perception methods for wildfire agents:
1. Perception Module: ASCII grid -> GPT-4o -> text description
2. VLM (Vision): Image -> GPT-4o Vision -> text description

Both are evaluated against ground truth extracted from ASCII encoding.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
import base64
from pathlib import Path


@dataclass
class PerceptionResult:
    """Result from a single perception method"""
    text_output: str
    time_elapsed: float
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class GroundTruth:
    """Ground truth data extracted from ASCII grid"""
    ignited_cells: List[Tuple[int, int]]
    on_fire_cells: List[Tuple[int, int]]
    extinguishing_cells: List[Tuple[int, int]]
    civilian_cells: List[Tuple[int, int]]
    water_sources: List[Tuple[int, int]]
    agent_position: Tuple[int, int]


@dataclass
class JudgeResult:
    """Result from judge LLM evaluation"""
    accuracy_score: float
    fire_detection_score: float
    civilian_detection_score: float
    explanation: str


class GroundTruthExtractor:
    """Extract structured ground truth from ASCII grid encoding"""

    def __init__(self):
        self.cell_mapping = {
            '-': 'unrevealed',
            '0': 'brush',
            '1': 'light_forest',
            '2': 'medium_forest',
            '3': 'dense_forest',
            'i': 'ignited',
            'f': 'on_fire',
            'e': 'extinguishing',
            'x': 'fully_extinguished',
            'w': 'water_source',
            'B': 'building',
            'C': 'civilian',
            "'0'": 'wet_brush',
            "'1'": 'wet_light_forest',
            "'2'": 'wet_medium_forest',
            "'3'": 'wet_dense_forest',
            "'C'": 'civilian'
        }

    def extract(self, ascii_grid: str, agent_position: Tuple[int, int], map_range: int) -> GroundTruth:
        """
        Extract ground truth data from ASCII grid

        Args:
            ascii_grid: String representation of the grid
            agent_position: (x, y) position of the agent
            map_range: Size of the perception grid (e.g., 21 for 21x21)

        Returns:
            GroundTruth object with all extracted features
        """
        ignited = []
        on_fire = []
        extinguishing = []
        civilians = []
        water_sources = []

        # Parse the grid
        lines = ascii_grid.strip().split('\n')

        # Calculate top-left corner of the grid
        top_left_x = agent_position[0] - map_range // 2
        top_left_y = agent_position[1] - map_range // 2

        for row_idx, line in enumerate(lines):
            # Split by comma and clean up
            cells = [cell.strip().replace('*', '').replace("'", "") for cell in line.split(',') if cell.strip()]

            for col_idx, cell in enumerate(cells):
                # Calculate global position
                global_x = top_left_x + col_idx
                global_y = top_left_y + row_idx
                pos = (global_x, global_y)

                # Categorize cell
                if cell == 'i':
                    ignited.append(pos)
                elif cell == 'f':
                    on_fire.append(pos)
                elif cell == 'e':
                    extinguishing.append(pos)
                elif cell == 'C' or cell == "'C'":
                    civilians.append(pos)
                elif cell == 'w':
                    water_sources.append(pos)


        return GroundTruth(
            ignited_cells=ignited,
            on_fire_cells=on_fire,
            extinguishing_cells=extinguishing,
            civilian_cells=civilians,
            water_sources=water_sources,
            agent_position=agent_position
        )

    def to_json(self, ground_truth: GroundTruth) -> Dict:
        """Convert ground truth to JSON-serializable dict"""
        return asdict(ground_truth)


class PerceptionModule:
    """Simplified perception module using ASCII grid (based on worker_agent.py)"""

    # GPT-4o pricing (per 1M tokens)
    INPUT_PRICE = 2.50  # $2.50 per 1M input tokens
    OUTPUT_PRICE = 10.00  # $10.00 per 1M output tokens

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def _build_observation_string(self,
                                  ascii_grid: str,
                                  agent_position: Tuple[int, int],
                                  agent_type: int,
                                  map_range: int,
                                  extra_variables: List[float]) -> str:
        """Build observation string matching worker_agent.py format"""

        agent_type_str = self._get_agent_type_string(agent_type)
        extra_vars_str = self._get_extra_variables_string(agent_type, extra_variables)

        obs_string = f"""
You are an agent, and your current location is {agent_position} and thus your minimap view will be the range ({agent_position[0]-map_range//2} to {agent_position[0]+map_range//2}, {agent_position[1]-map_range//2} to {agent_position[1]+map_range//2}). The first column is x = {agent_position[0]-map_range//2},  and the first row is  y = {agent_position[1]-map_range//2}. Remember to take this into account in coordinate calculation.

This is your minimap view, with commas separating cells and newlines separating rows:
{ascii_grid}

Each cell is represented by a character corresponding to the type of terrain:
    0: brush (no trees)
    1: light forest (1 tree)
    2: medium forest (2 trees)
    3: dense forest (3 trees)
    i: Ignited
    f: On Fire
    e: Extinguishing
    x: Fully Extinguished
    w: Water Source Cell (no trees)
    B: building (no trees)

IGNORE ALL "-". Those are unrevealed cells. They will reveal themselves when you get closer to them.

The cells in single quotations are wet cells. 'C' cells are civilians.

The bolded cell is the current cell you are in at {agent_position}.
"""
        return obs_string

    def _get_agent_type_string(self, agent_type: int) -> str:
        """Get agent type as string"""
        types = {0: 'Firefighter', 1: 'Bulldozer', 2: 'Drone', 3: 'Helicopter'}
        return types.get(agent_type, 'Unknown')

    def _get_extra_variables_string(self, agent_type: int, extra_variables: List[float]) -> str:
        """Get string of extra variables"""
        extra_string = ""
        if agent_type == 0:  # Firefighter
            if extra_variables[0] == 0:
                extra_string += "You are NOT carrying a civilian.\n"
            else:
                extra_string += "You are carrying a civilian.\n"
            extra_string += f"You currently have {int(extra_variables[1])}/5 water\n"

        if agent_type == 3:  # Helicopter
            if extra_variables[0] == 0:
                extra_string += "You are NOT carrying any firefighters.\n"
            else:
                extra_string += f"You are carrying {int(extra_variables[0])}/5 firefighters.\n"

        return extra_string

    def perceive(self,
                 ascii_grid: str,
                 agent_position: Tuple[int, int],
                 agent_type: int,
                 map_range: int,
                 extra_variables: List[float]) -> PerceptionResult:
        """
        Generate perception using ASCII grid

        Returns:
            PerceptionResult with text output and metrics
        """
        agent_type_str = self._get_agent_type_string(agent_type)

        system_message = f"""You are a {agent_type_str} agent within a forest grid world.

Your job is to analyze observations and provide a perception summary."""

        user_message = f"""Here are your observations: {self._build_observation_string(ascii_grid, agent_position, agent_type, map_range, extra_variables)}

Create a detailed perception summary (<=100 words) using the following tags:

<perception>
A detailed summary of what you observe, including your location, surroundings, any fires, civilians, or important features you can see, and terrain types. Describe all fires and civilians in exact coordinates. Count the spaces/characters to find the exact coordinates. Remember that the first column is x = {agent_position[0]-map_range//2},  and the first row is  y = {agent_position[1]-map_range//2}.
</perception>
"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        start_time = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        elapsed_time = time.time() - start_time

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * self.INPUT_PRICE +
                output_tokens / 1_000_000 * self.OUTPUT_PRICE)

        # Parse perception from response
        perception_text = self._parse_tag_content(response.choices[0].message.content, "perception")

        # Store full prompt for logging
        full_prompt = f"SYSTEM:\n{system_message}\n\nUSER:\n{user_message}"

        return PerceptionResult(
            text_output=perception_text,
            time_elapsed=elapsed_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        ), full_prompt

    def _parse_tag_content(self, response: str, tag_name: str) -> str:
        """Parse content from a specific tag in the response"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        start_index = response.find(start_tag)
        end_index = response.find(end_tag)

        if start_index != -1 and end_index != -1 and end_index > start_index:
            content_start = start_index + len(start_tag)
            content = response[content_start:end_index].strip()
            return content
        else:
            return response  # Fallback to full response if tags not found


class VLMPerception:
    """Vision-based perception using GPT-4o Vision"""

    # GPT-4o Vision pricing (per 1M tokens)
    INPUT_PRICE = 2.50  # $2.50 per 1M input tokens
    OUTPUT_PRICE = 10.00  # $10.00 per 1M output tokens

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_agent_type_string(self, agent_type: int) -> str:
        """Get agent type as string"""
        types = {0: 'Firefighter', 1: 'Bulldozer', 2: 'Drone', 3: 'Helicopter'}
        return types.get(agent_type, 'Unknown')

    def _get_extra_variables_string(self, agent_type: int, extra_variables: List[float]) -> str:
        """Get string of extra variables"""
        extra_string = ""
        if agent_type == 0:  # Firefighter
            if extra_variables[0] == 0:
                extra_string += "You are NOT carrying a civilian.\n"
            else:
                extra_string += "You are carrying a civilian.\n"
            extra_string += f"You currently have {int(extra_variables[1])}/5 water\n"

        if agent_type == 3:  # Helicopter
            if extra_variables[0] == 0:
                extra_string += "You are NOT carrying any firefighters.\n"
            else:
                extra_string += f"You are carrying {int(extra_variables[0])}/5 firefighters.\n"

        return extra_string

    def perceive(self,
                 image_path: str,
                 agent_position: Tuple[int, int],
                 agent_type: int,
                 map_range: int,
                 extra_variables: List[float]) -> PerceptionResult:
        """
        Generate perception using vision model

        Args:
            image_path: Path to the image file
            agent_position: (x, y) position of the agent
            agent_type: Type of agent (0-3)
            map_range: Size of perception grid
            extra_variables: Additional state variables

        Returns:
            PerceptionResult with text output and metrics
        """
        agent_type_str = self._get_agent_type_string(agent_type)
        extra_vars_str = self._get_extra_variables_string(agent_type, extra_variables)

        system_message = f"""You are a {agent_type_str} agent within a forest grid world.

Your job is to analyze visual observations and provide a perception summary."""

        user_message = f"""You are viewing an image of your minimap. Your current location is {agent_position} and your view range is ({agent_position[0]-map_range//2} to {agent_position[0]+map_range//2}, {agent_position[1]-map_range//2} to {agent_position[1]+map_range//2}). 

The image shows a grid-based map where:
- Green shades: Forest terrain (darker = denser forest)
- Brown/Tan: Brush or open terrain
- Red/Orange: Fires (different intensities)
- Blue: Water sources
- White Civilian Icons: Civilians
- Black/unrevealed: Fog of war (unexplored areas)

Your current position is marked in the center of the grid at {agent_position}. 


Create a detailed perception summary (<=100 words) using the following tags:

<perception>
A detailed summary of what you observe, including your location, surroundings, any fires, and civilians. Describe all fires and civilians in exact coordinates. Count the spaces/characters to find the exact coordinates.
</perception>
"""

        # Encode image
        base64_image = self._encode_image(image_path)

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        start_time = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=300
        )
        elapsed_time = time.time() - start_time

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * self.INPUT_PRICE +
                output_tokens / 1_000_000 * self.OUTPUT_PRICE)

        # Parse perception from response
        perception_text = self._parse_tag_content(response.choices[0].message.content, "perception")

        return PerceptionResult(
            text_output=perception_text,
            time_elapsed=elapsed_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )

    def _parse_tag_content(self, response: str, tag_name: str) -> str:
        """Parse content from a specific tag in the response"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        start_index = response.find(start_tag)
        end_index = response.find(end_tag)

        if start_index != -1 and end_index != -1 and end_index > start_index:
            content_start = start_index + len(start_tag)
            content = response[content_start:end_index].strip()
            return content
        else:
            return response  # Fallback to full response if tags not found


class JudgeLLM:
    """Judge LLM to evaluate perception outputs against ground truth"""

    # GPT-4o pricing
    INPUT_PRICE = 2.50
    OUTPUT_PRICE = 10.00

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def evaluate(self,
                 perception_text: str,
                 ground_truth: GroundTruth) -> Tuple[JudgeResult, float]:
        """
        Evaluate perception text against ground truth

        Returns:
            Tuple of (JudgeResult, cost)
        """
        ground_truth_json = json.dumps(asdict(ground_truth), indent=2)

        system_message = """You are an expert evaluator. Your job is to assess how accurately a perception text describes the ground truth state of a wildfire grid environment."""

        user_message = f
        
"""Ground Truth Data (extracted from the environment):
{ground_truth_json}

Perception Text to Evaluate:
"{perception_text}"

Evaluate this perception text and provide:
1. Fire Detection Score (0-10): How accurately did it identify fires (ignited, on fire, extinguishing cells)?
2. Civilian Detection Score (0-10): How accurately did it identify civilians?

Respond using this format:

<fire_score>NUMBER</fire_score>
<civilian_score>NUMBER</civilian_score>
<explanation>
Brief explanation of your scoring
</explanation>"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        # Calculate cost
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens / 1_000_000 * self.INPUT_PRICE +
                output_tokens / 1_000_000 * self.OUTPUT_PRICE)

        # Parse response
        content = response.choices[0].message.content
        fire_score = float(self._parse_tag(content, "fire_score") or "0")
        civilian_score = float(self._parse_tag(content, "civilian_score") or "0")
        explanation = self._parse_tag(content, "explanation") or ""

        return JudgeResult(
            accuracy_score=(civilian_score+fire_score)//2,
            fire_detection_score=fire_score,
            civilian_detection_score=civilian_score,
            explanation=explanation.strip()
        ), cost

    def _parse_tag(self, text: str, tag_name: str) -> Optional[str]:
        """Parse content from XML-style tag"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"
        start_idx = text.find(start_tag)
        end_idx = text.find(end_tag)

        if start_idx != -1 and end_idx != -1:
            return text[start_idx + len(start_tag):end_idx].strip()
        return None


class ComparisonRunner:
    """Main runner for comparing perception methods"""

    def __init__(self, api_key: str, data_folder, output_path: str):
        self.api_key = api_key

        # Handle both single folder and list of folders
        if isinstance(data_folder, list):
            self.data_folder = [Path(folder) for folder in data_folder]
        else:
            self.data_folder = Path(data_folder)

        self.output_path = Path(output_path)

        # Create detailed log file path
        self.detailed_log_path = self.output_path.parent / f"{self.output_path.stem}_detailed.txt"

        self.perception_module = PerceptionModule(api_key)
        self.vlm_perception = VLMPerception(api_key)
        self.ground_truth_extractor = GroundTruthExtractor()
        self.judge = JudgeLLM(api_key)

        self.results = {
            "perception_module": {
                "samples": [],
                "avg_accuracy": 0.0,
                "avg_fire_score": 0.0,
                "avg_civilian_score": 0.0,
                "total_cost": 0.0,
                "avg_time": 0.0,
                "total_tokens": 0
            },
            "vlm": {
                "samples": [],
                "avg_accuracy": 0.0,
                "avg_fire_score": 0.0,
                "avg_civilian_score": 0.0,
                "total_cost": 0.0,
                "avg_time": 0.0,
                "total_tokens": 0
            }
        }

        # Initialize detailed log file
        self._init_detailed_log()

    def run(self):
        """Run comparison on all samples in data folder(s)"""
        # Support multiple data folders
        if isinstance(self.data_folder, list):
            all_metadata_files = []
            for folder in self.data_folder:
                folder_path = Path(folder)
                metadata_files = list(folder_path.glob("*_metadata.json"))
                all_metadata_files.extend([(f, folder_path) for f in metadata_files])

            if not all_metadata_files:
                print(f"No metadata files found in any of the folders")
                return

            print(f"Found {len(all_metadata_files)} samples across {len(self.data_folder)} folders")
        else:
            # Single folder mode (backward compatible)
            metadata_files = list(self.data_folder.glob("*_metadata.json"))

            if not metadata_files:
                print(f"No metadata files found in {self.data_folder}")
                return

            all_metadata_files = [(f, self.data_folder) for f in metadata_files]
            print(f"Found {len(metadata_files)} samples to process")

        for metadata_file, source_folder in all_metadata_files:
            sample_name = metadata_file.stem.replace("_metadata", "")
            print(f"\nProcessing sample: {sample_name} (from {source_folder.name})")

            try:
                self._process_sample_from_folder(sample_name, source_folder)
            except Exception as e:
                print(f"Error processing {sample_name}: {e}")
                continue

        # Compute aggregated metrics
        self._compute_aggregates()

        # Save results
        self._save_results()

        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE")
        print(f"{'='*60}")
        print(f"\nPerception Module:")
        print(f"  Avg Accuracy: {self.results['perception_module']['avg_accuracy']:.2f}")
        print(f"  Avg Fire Score: {self.results['perception_module']['avg_fire_score']:.2f}")
        print(f"  Avg Civilian Score: {self.results['perception_module']['avg_civilian_score']:.2f}")
        print(f"  Total Cost: ${self.results['perception_module']['total_cost']:.4f}")
        print(f"  Avg Time: {self.results['perception_module']['avg_time']:.3f}s")

        print(f"\nVLM:")
        print(f"  Avg Accuracy: {self.results['vlm']['avg_accuracy']:.2f}")
        print(f"  Avg Fire Score: {self.results['vlm']['avg_fire_score']:.2f}")
        print(f"  Avg Civilian Score: {self.results['vlm']['avg_civilian_score']:.2f}")
        print(f"  Total Cost: ${self.results['vlm']['total_cost']:.4f}")
        print(f"  Avg Time: {self.results['vlm']['avg_time']:.3f}s")

        print(f"\nResults saved to:")
        print(f"  JSON: {self.output_path}")
        print(f"  Detailed Log: {self.detailed_log_path}")

    def _process_sample(self, sample_name: str):
        """Process a single sample (backward compatible - single folder)"""
        self._process_sample_from_folder(sample_name, self.data_folder)

    def _process_sample_from_folder(self, sample_name: str, source_folder: Path):
        """Process a single sample from a specific folder"""
        # Load metadata
        metadata_path = source_folder / f"{sample_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load ASCII
        ascii_path = source_folder / f"{sample_name}_ascii.txt"
        with open(ascii_path, 'r') as f:
            ascii_grid = f.read()

        # Load image path
        image_path = source_folder / f"{sample_name}_image.png"
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return

        # Extract parameters from metadata
        agent_position = tuple(metadata['agent_position'])
        agent_type = metadata['agent_type']
        map_range = metadata['map_range']
        extra_variables = metadata['extra_variables']

        # Extract ground truth
        ground_truth = self.ground_truth_extractor.extract(ascii_grid, agent_position, map_range)

        # Run perception module
        print("  Running perception module...")
        perception_result, perception_prompt = self.perception_module.perceive(
            ascii_grid, agent_position, agent_type, map_range, extra_variables
        )

        # Evaluate perception module
        perception_judge, perception_judge_cost = self.judge.evaluate(
            perception_result.text_output, ground_truth
        )

        # Run VLM
        print("  Running VLM...")
        vlm_result = self.vlm_perception.perceive(
            str(image_path), agent_position, agent_type, map_range, extra_variables
        )

        # Evaluate VLM
        vlm_judge, vlm_judge_cost = self.judge.evaluate(
            vlm_result.text_output, ground_truth
        )

        # Store results
        perception_sample = {
            "sample_name": sample_name,
            "perception_text": perception_result.text_output,
            "accuracy": perception_judge.accuracy_score,
            "fire_score": perception_judge.fire_detection_score,
            "civilian_score": perception_judge.civilian_detection_score,
            "time": perception_result.time_elapsed,
            "tokens": perception_result.input_tokens + perception_result.output_tokens,
            "cost": perception_result.cost + perception_judge_cost,
            "judge_explanation": perception_judge.explanation,
            "full_prompt": perception_prompt
        }

        vlm_sample = {
            "sample_name": sample_name,
            "perception_text": vlm_result.text_output,
            "accuracy": vlm_judge.accuracy_score,
            "fire_score": vlm_judge.fire_detection_score,
            "civilian_score": vlm_judge.civilian_detection_score,
            "time": vlm_result.time_elapsed,
            "tokens": vlm_result.input_tokens + vlm_result.output_tokens,
            "cost": vlm_result.cost + vlm_judge_cost,
            "judge_explanation": vlm_judge.explanation
        }

        self.results["perception_module"]["samples"].append(perception_sample)
        self.results["vlm"]["samples"].append(vlm_sample)

        # Log detailed results to text file
        self._log_sample_details(sample_name, ascii_grid, ground_truth,
                                perception_sample, vlm_sample)

        print(f"    Perception Module - Accuracy: {perception_judge.accuracy_score:.1f}, Time: {perception_result.time_elapsed:.2f}s, Cost: ${perception_result.cost + perception_judge_cost:.4f}")
        print(f"    VLM - Accuracy: {vlm_judge.accuracy_score:.1f}, Time: {vlm_result.time_elapsed:.2f}s, Cost: ${vlm_result.cost + vlm_judge_cost:.4f}")

    def _compute_aggregates(self):
        """Compute aggregate statistics"""
        for method in ["perception_module", "vlm"]:
            samples = self.results[method]["samples"]
            if not samples:
                continue

            self.results[method]["avg_accuracy"] = sum(s["accuracy"] for s in samples) / len(samples)
            self.results[method]["avg_fire_score"] = sum(s["fire_score"] for s in samples) / len(samples)
            self.results[method]["avg_civilian_score"] = sum(s["civilian_score"] for s in samples) / len(samples)
            self.results[method]["total_cost"] = sum(s["cost"] for s in samples)
            self.results[method]["avg_time"] = sum(s["time"] for s in samples) / len(samples)
            self.results[method]["total_tokens"] = sum(s["tokens"] for s in samples)

    def _save_results(self):
        """Save results to JSON file"""
        with open(self.output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _init_detailed_log(self):
        """Initialize the detailed log file with header"""
        with open(self.detailed_log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("VLM vs PERCEPTION MODULE - DETAILED COMPARISON LOG\n")
            f.write("="*80 + "\n\n")

    def _log_sample_details(self, sample_name: str, ascii_grid: str,
                           ground_truth: GroundTruth,
                           perception_sample: dict, vlm_sample: dict):
        """Log detailed information for a single sample to text file"""
        with open(self.detailed_log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"SAMPLE: {sample_name}\n")
            f.write("="*80 + "\n\n")

            # ASCII Grid
            f.write("-"*80 + "\n")
            f.write("ASCII GRID:\n")
            f.write("-"*80 + "\n")
            f.write(ascii_grid)
            f.write("\n\n")

            # Ground Truth
            f.write("-"*80 + "\n")
            f.write("GROUND TRUTH:\n")
            f.write("-"*80 + "\n")
            f.write(f"Agent Position: {ground_truth.agent_position}\n")
            f.write(f"Ignited Cells: {ground_truth.ignited_cells}\n")
            f.write(f"On Fire Cells: {ground_truth.on_fire_cells}\n")
            f.write(f"Extinguishing Cells: {ground_truth.extinguishing_cells}\n")
            f.write(f"Civilian Cells: {ground_truth.civilian_cells}\n")
            f.write(f"Water Sources: {ground_truth.water_sources}\n")
            f.write("\n")

            # Perception Module Results
            f.write("-"*80 + "\n")
            f.write("PERCEPTION MODULE (ASCII-based):\n")
            f.write("-"*80 + "\n")
            f.write("PROMPT:\n")
            f.write(perception_sample['full_prompt'])
            f.write("\n\n")
            f.write(f"Output: {perception_sample['perception_text']}\n\n")
            f.write(f"Accuracy Score: {perception_sample['accuracy']:.1f}/100\n")
            f.write(f"Fire Detection Score: {perception_sample['fire_score']:.1f}/100\n")
            f.write(f"Civilian Detection Score: {perception_sample['civilian_score']:.1f}/100\n")
            f.write(f"Time: {perception_sample['time']:.3f}s\n")
            f.write(f"Tokens: {perception_sample['tokens']}\n")
            f.write(f"Cost: ${perception_sample['cost']:.4f}\n\n")
            f.write(f"Judge Explanation:\n{perception_sample['judge_explanation']}\n")
            f.write("\n")

            # VLM Results
            f.write("-"*80 + "\n")
            f.write("VLM (Vision-based):\n")
            f.write("-"*80 + "\n")
            f.write(f"Output: {vlm_sample['perception_text']}\n\n")
            f.write(f"Accuracy Score: {vlm_sample['accuracy']:.1f}/100\n")
            f.write(f"Fire Detection Score: {vlm_sample['fire_score']:.1f}/100\n")
            f.write(f"Civilian Detection Score: {vlm_sample['civilian_score']:.1f}/100\n")
            f.write(f"Time: {vlm_sample['time']:.3f}s\n")
            f.write(f"Tokens: {vlm_sample['tokens']}\n")
            f.write(f"Cost: ${vlm_sample['cost']:.4f}\n\n")
            f.write(f"Judge Explanation:\n{vlm_sample['judge_explanation']}\n")
            f.write("\n")


def main():
    """CLI entry point"""
    import argparse

    #parser = argparse.ArgumentParser(description="Compare Perception Module vs VLM")
    #parser.add_argument("--data-folder", required=True, help="Folder containing image+ASCII pairs")
    #parser.add_argument("--output", required=True, help="Output JSON file path")
    #parser.add_argument("--api-key", required=True, help="OpenAI API key")

    #args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


    api_key = os.environ['OPENAI_API_KEY']

    # Single folder mode (backward compatible)
    # data_folder = "../results/logs/WILDFIRE/VLM_Collection/42/2025-10-15-22-56-23/vlm_comparison_data"

    # Multiple folders mode - combine data from multiple runs
    data_folder = [
        "crew-algorithms/crew_algorithms/wildfire_alg/results/logs/WILDFIRE/VLM_Collection/42/2025-10-15-22-56-23/vlm_comparison_data",
        # Add more folders here to combine multiple runs
        "crew-algorithms/crew_algorithms/wildfire_alg/results/logs/WILDFIRE/VLM_Collection/4232/2025-10-19-17-53-22/vlm_comparison_data",
        "crew-algorithms/crew_algorithms/wildfire_alg/results/logs/WILDFIRE/VLM_Collection/1423122/2025-10-19-18-15-15/vlm_comparison_data",
    ]

    output = "vlm_results.json"



    runner = ComparisonRunner(
        api_key=api_key,
        data_folder=data_folder,
        output_path=output
    )

    runner.run()


if __name__ == "__main__":
    main()

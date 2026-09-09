import re
from openai import OpenAI
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.CAMON.agent import Agent


class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the TEMPLATE_primitives algorithm.
    
    Attributes:
        type (int): The type of action to be performed
        param_1 (int): First parameter (typically x-coordinate)
        param_2 (int): Second parameter (typically y-coordinate)
        description (str): Human-readable description of the action
    """
    type: int
    param_1: int
    param_2: int
    description: str

    def print_option(self)->None:
        """Prints the action details in a list format [type, param_1, param_2, description]."""
        print([self.type, self.param_1, self.param_2, self.description])

def translate_action(option_str: str, type: int, global_data: dict) -> Action:
    """
    Translates a natural language action description into a structured Action object using GPT-4.
    
    Args:
        option_str (str): Natural language description of the action
        type (int): Agent type (0=firefighter, 1=bulldozer, 2=drone, 3=helicopter)
        global_data (dict): Global state containing API keys and other shared data
        
    Returns:
        Action: Structured action object with type, parameters and description
    """
    system_message = f"""
                        You are the controller of a highly trained embodied agent within a grid forest world. 
                        Your job is to convert a string action into a structured format for robotic control."""
    # Determine type
    if type == 0:
        type_string = 'firefighter'
    elif type == 1:
        type_string = 'bulldozer'
    elif type == 2:
        type_string = 'drone'
    else:
        type_string = 'helicopter'
    # Load prompt
    prompt_path = f'algorithms/TEMPLATE_primitives/prompts/translator/{type_string}_translator.txt'
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().replace("ACTION", option_str)
    # Call OpenAI
    client = OpenAI(api_key=global_data['leader_agent'].api_key)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role':'system', 'content':system_message},
            {'role':'user', 'content':prompt}
        ],
        temperature=0.7
    )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content
    
    # Extract fields
    def extract(tag: str) -> str:
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", result, re.DOTALL)
        if not m:
            raise ValueError(f"Missing tag {tag}")
        return m.group(1).strip()
    try:
        return Action(
            type=int(extract('type')),
            param_1=int(extract('param_1')),
            param_2=int(extract('param_2')),
            description=extract('description')
        )
    except:
        return Action(
            type=0,
            param_1=0,
            param_2=0,
            description='ERROR EXECUTING: ' + extract('description')
        )


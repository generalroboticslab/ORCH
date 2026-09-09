import os
import re
from openai import OpenAI
from crew_algorithms.wildfire_alg.core.token_guard import guarded_chat_completion
from pydantic import BaseModel
from typing import List, Tuple, Dict
from crew_algorithms.wildfire_alg.algorithms.HMAS_2.agent import Agent
from crew_algorithms.wildfire_alg.libraries.firefighter_action_library import Run_Firefighter_Action
from crew_algorithms.wildfire_alg.libraries.bulldozer_action_library import Run_Bulldozer_Action
from crew_algorithms.wildfire_alg.libraries.drone_action_library import Run_Drone_Action
from crew_algorithms.wildfire_alg.libraries.helicopter_action_library import Run_Helicopter_Action

class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the HMAS_2 algorithm.
    
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

    def print_action(self) -> None:
        """Prints the action details in a list format [type, param_1, param_2, description]."""
        print([self.type, self.param_1, self.param_2, self.description])

def translate_action(action_str: str, type: int, global_data: dict) -> Action:
    """
    Translates a natural language action description into a structured Action object using GPT-4.
    
    Args:
        action_str (str): Natural language description of the action
        type (int): Agent type (0=firefighter, 1=bulldozer, 2=drone, 3=helicopter)
        global_data (dict): Global state containing API configuration
        
    Returns:
        Action: Structured action object with type, parameters and description
    """
    if type == 0:
        type_string = 'firefighter'
    elif type == 1:
        type_string = 'bulldozer'
    elif type == 2:
        type_string = 'drone'
    elif type == 3:
        type_string = 'helicopter'

    prompt_path = os.path.join(
        'algorithms',
        'HMAS_2', 'prompts', 'translator', f'{type_string}_translator.txt'
    )
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(
        'crew_algorithms',
        'wildfire_alg',
        'algorithms',
        'HMAS_2', 'prompts', 'translator', f'{type_string}_translator.txt'
    )
    
    with open(prompt_path, 'r', encoding='utf-8') as file:
        if action_str is None:
            action_str = "N/A"
        translator_prompt = file.read().replace("ACTION", action_str)
        

    system_message = (
        "You are the controller of a highly trained embodied agent within a grid forest world. "
        "Your job is to convert a string action into a structured format for robotic control. "
        "Do not reveal chain-of-thought reasoning. Output only the final answer."
    )
    user_message = translator_prompt

    if global_data["large_model"].find("gpt") != -1:
        client = OpenAI(
            api_key=global_data.get('api_key'),
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            # temperature=0,
            reasoning_effort="minimal",
        )
        
    else:
        client = OpenAI(
            api_key=global_data.get('api_key'),
            base_url=global_data.get('llm_url'),
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content

    def extract(tag: str) -> str:
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", result, re.DOTALL)
        if not m:
            print(f"Missing tag {tag}")
            m1 = re.search(fr"<{tag}>\s*(.*?)\s*", result, re.DOTALL)
            m2 = re.search(fr"\s*(.*?)\s*</{tag}>", result, re.DOTALL)
            if not m1 and not m2:
                return "N/A"
            else:
                if not m1:
                    return m2.group(1).strip()
                else:
                    return m1.group(1).strip()
        else:
            return m.group(1).strip()

    try:
        return Action(
            type=int(extract("type")),
            param_1=int(extract("param_1")),
            param_2=int(extract("param_2")),
            description=extract("description")
        )
    except:
        return Action(
            type=0,
            param_1=0,
            param_2=0,
            description='ERROR EXECUTING: ' + extract('description')
        )

def propose_actions(global_data: dict, past_conversation:list, retry_count:int=0):
    """
    Generates action proposals for all agents in the system.
    
    Args:
        global_data: Global state containing agent and environment information
        past_conversation: History of planning conversations
        retry_count: Current retry count for invalid action proposals
        
    Returns:
        tuple: (proposed_actions: dict, messages: list)
            - proposed_actions: Maps agent IDs to their proposed actions
            - messages: Updated conversation history
        
    Notes:
        - Maintains conversation history for context
        - Handles initial and subsequent planning rounds
        - Logs planning process to file
        - Retries at most 3 times for invalid proposals
    """
    #If first round
    if len(past_conversation)==0:
        current_state = {}


        for agent in global_data["agents"]:

            if agent.type==0 and agent.extra_variables[2]==1:
                current_state.update({f"AGENT_{agent.id}": {"perception":agent.last_perception, "available_actions": "            - Do nothing since you are in a helicopter."}})
                continue
            if agent.type == 0:
                type_string = 'firefighter'
            elif agent.type == 1:
                type_string = 'bulldozer'
            elif agent.type == 2:
                type_string = 'drone'
            elif agent.type == 3:
                type_string = 'helicopter'
            

            desc_path = os.path.join(
                'algorithms','HMAS_2', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
            )
            if not os.path.exists(desc_path):
                desc_path = os.path.join(
                'crew_algorithms',
                'wildfire_alg',
                'algorithms',
                'HMAS_2', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
            )
            with open(desc_path, 'r', encoding='utf-8') as file:
                abilities_string = file.read()
            
            current_state.update({f"AGENT_{agent.id}": {"perception":agent.last_perception, "available_actions": abilities_string}})

        try:
            task_string = global_data['agents'][0].current_task
        except Exception:
            task_string = "N/A"

        try:
            history_string = global_data["step_history"]
        except Exception:
            history_string = "N/A"

        try:
            current_state_string = current_state
        except Exception:
            current_state_string = "N/A"

        generate_action_string = f"""
                You are central planner directing agents in a cooperative multi-agent robotic task. 

                Your team's task is:
                {task_string}
                ---

                Your team's previous state action pairs at each step are:
                {history_string}
                ---

                Your team's current state and available actions are:
                {current_state_string}
                ---
                
                Now provide the next best action for each agent.

                Rules:
                1. You must provide exactly one action for each agent.
                2. Each action must be exactly one of that agent's available actions.
                3. Do not invent new actions.
                4. Do not propose multiple actions per agent.
                5. Include the 'do nothing' action only if it is the best available action.
                6. Output XML only. Do not include reasoning, markdown, or extra text.

                Required format:

                <actions>
                <AGENT_1>exact available action</AGENT_1>
                <AGENT_2>exact available action</AGENT_2>
                ...
                <AGENT_N>exact available action</AGENT_N>
                </actions>

                Make sure every agent appears exactly once.
                """
        system_message = f"""
                        You are central planner directing agents in a cooperative multi-agent robotic task. 
                        Your job is to provide the next best action for each agent. 
                        Do not reveal chain-of-thought reasoning. Output only the final answer."""
        
        user_message = generate_action_string

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]+past_conversation
    else:
        # 2 for prompts, 1 for response, 1 for feedback
        while len(past_conversation)>8:
            past_conversation.pop(2)
            past_conversation.pop(2)
        messages = past_conversation

    if global_data["large_model"].find("gpt") != -1:
        response = guarded_chat_completion(global_data["client"], 
            model=global_data["large_model"],
            messages=messages,
            # temperature=0,
            reasoning_effort="minimal",
        )
        
    else:
        response = guarded_chat_completion(global_data["client"], 
            model=global_data["large_model"],
            messages=messages,
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    content = response.choices[0].message.content
    messages.append({"role": "assistant", "content": content})
    proposed_actions = {}
    for agent in global_data.get('agents'):
       
        tag = f"AGENT_{agent.id}"
        m = re.search(fr"<{tag}>\s*(.*?)\s*</{tag}>", content, re.DOTALL)
        if m:
            a = m.group(1).strip()
            proposed_actions.update({f"AGENT_{agent.id}": a})
    
    # print("\n")
    # print("###################")
    # print(content)
    # print("###################")
    # print("PROPOSED ACTIONS:", proposed_actions)
    # print("###################")
    # print(global_data["agents"])
    # print("###################")
    # print("\n")
    
    if len(proposed_actions.keys())!= len(global_data["agents"]):
        print("INVALID NUMBER OF PROPOSED ACTIONS")
        if retry_count < 2:
            return propose_actions(global_data=global_data, past_conversation=past_conversation, retry_count=retry_count+1)
        print("MAX RETRIES REACHED: returning None actions for all agents")
        proposed_actions = {f"AGENT_{agent.id}": None for agent in global_data.get("agents", [])}
        return proposed_actions, messages
    

    if len(messages)==3:
        chat_string = f"Proposing Action Plan\n" + "-"*20 + "\n\n"
        for m in messages:
            chat_string += f"{m['role']}\n\n{m['content']}\n\n"
    else:
        chat_string = f"Revising Action Plan\n" + "-"*20 + "\n\n"
        for m in messages[len(messages)-2:]:
            chat_string += f"{m['role']}\n-----\n{m['content']}\n-----\n\n"
    chat_string += "-"*20 + "\nEND CHAT\n\n\n"
    filepath = os.path.join(global_data["path"], f"central_agent.txt")
    with open(filepath, "a", encoding="utf-8") as file:
        file.write(chat_string)

    return proposed_actions, messages
    

def provide_feedback(agent: Agent, global_data:dict, proposed_actions:dict):
    """
    Allows agents to provide feedback on proposed action plans.
    
    Args:
        agent: The agent providing feedback
        global_data: Global state containing team information
        proposed_actions: Dictionary of currently proposed actions
        
    Returns:
        str: Feedback message or "ACCEPT" if plan is satisfactory
        
    Notes:
        - Considers agent's role and capabilities
        - Evaluates plan feasibility
        - Provides constructive feedback
    """

    client = global_data["client"]
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    elif agent.type == 3:
        type_string = 'Helicopter'

    current_state = {}
    for a in global_data["agents"]:
        if a.type==0 and a.extra_variables[2]==1:
            current_state.update({f"AGENT_{a.id}": {"perception":a.last_perception, "available_actions": "            - Do nothing since you are in a helicopter."}})
            continue

        if a.type == 0:
            type_string = 'firefighter'
        elif a.type == 1:
            type_string = 'bulldozer'
        elif a.type == 2:
            type_string = 'drone'
        elif a.type == 3:
            type_string = 'helicopter'

        desc_path = os.path.join(
                'algorithms', 'HMAS_2', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
            )
        if not os.path.exists(desc_path):
            desc_path = os.path.join(
            'crew_algorithms',
            'wildfire_alg',
            'algorithms',
            'HMAS_2', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
        )
        with open(desc_path, 'r', encoding='utf-8') as file:
            abilities_string = file.read()
        current_state.update({f"AGENT_{a.id}": {"perception":a.last_perception, "available_actions": abilities_string}})

            
    generate_feedback_string = f"""
            You are AGENT_{agent.id}, a {type_string} Agent in a cooperative multi-agent robotic task. 

            Your team's task is:
            {global_data['agents'][0].current_task}
            ---

            Your team's previous state action pairs at each step are:
            {global_data["step_history"]}
            ---

            Your team's current state and available actions are:
            {current_state}
            ---

            The initial action plan from the central planner is:
            {proposed_actions}
            ---

            Now your job is to provide feedback to the action plan specficially regarding your agent. 
            If the plan is satisfactory, the feedback should only be 'ACCEPT'.

            Remember, you are AGENT_{agent.id} a {type_string} Agent, located at {agent.last_position}.

            <reasoning>(any reasoning or calculations)</reasoning>

            <feedback>'feedback'</feedback>

            """
    
    system_message = f"""
                    You are the AGENT_{agent.id}, an embodied agent in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    Your job is to provide feedback to the central planner's action plan, specfically regarding your agent. Do not reveal chain-of-thought reasoning. Output only the final answer."""
    
    user_message = generate_feedback_string
    
    if global_data["large_model"].find("gpt") != -1:
        client = OpenAI(
            api_key=global_data.get('api_key'),
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            # temperature=0,
            reasoning_effort="minimal",
        )
    else:
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content
    agent.log_chat("Providing Feedback", [("user", user_message), ("assistant", result)])
    m = re.search(r"<feedback>\s*(.*?)\s*</feedback>", result, re.DOTALL)
    if not m:
        print("NO FEEDBACK FOUND")
        return None
    feedback_str = m.group(1).strip()
    return feedback_str
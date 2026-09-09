import os
import re
from openai import OpenAI, BadRequestError
from crew_algorithms.wildfire_alg.core.token_guard import guarded_chat_completion
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.Embodied.agent import Agent
from crew_algorithms.wildfire_alg.libraries.firefighter_action_library import Run_Firefighter_Action
from crew_algorithms.wildfire_alg.libraries.bulldozer_action_library import Run_Bulldozer_Action
from crew_algorithms.wildfire_alg.libraries.drone_action_library import Run_Drone_Action
from crew_algorithms.wildfire_alg.libraries.helicopter_action_library import Run_Helicopter_Action

class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the Embodied algorithm.
    
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
    Converts a string description of an action into a structured Action object using GPT-4.
    
    Args:
        action_str (str): Natural language description of the action
        type (int): Agent type (0=firefighter, 1=bulldozer, 2=drone, 3=helicopter)
        global_data (dict): Global state containing API keys and other shared data
        
    Returns:
        Action: Structured action object with type, parameters and description
        
    Notes:
        - Uses OpenAI's GPT-4 to parse natural language into structured format
        - Loads agent-specific translation prompts from files
        - Handles errors by returning a "do nothing" action
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
        'Embodied', 'prompts', 'translator', f'{type_string}_translator.txt'
    )
    if not os.path.exists(prompt_path):
        prompt_path = os.path.join(
        'crew_algorithms',
        'wildfire_alg',
        'algorithms',
        'Embodied', 'prompts', 'translator', f'{type_string}_translator.txt'
    )
    
    with open(prompt_path, 'r', encoding='utf-8') as file:
        translator_prompt = file.read().replace("ACTION", action_str)

    system_message = (
        "You are the controller of a highly trained embodied agent within a grid forest world. "
        "Your job is to convert a string action into a structured format for robotic control."
    )
    user_message = translator_prompt

    try:    
        if global_data["large_model"].find("gpt") != -1:
            client = OpenAI(
                api_key=global_data["agents"][0].api_key,
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
                api_key=global_data["agents"][0].api_key,
                base_url=global_data["agents"][0].cfg.envs.llm_url,
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
    except BadRequestError as e:
        global_data["api_calls"]+=1
        result = "ERROR EXECUTING"
    
    if result.find("</think>") != -1:
        result = result.split("</think>")[-1].strip()
    
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

def communication_round(agent: Agent, global_data: dict):
    """
    Handles communication between agents by generating and distributing messages.
    
    Args:
        agent (Agent): The agent initiating communication
        global_data (dict): Global state containing team composition and environment info
        
    Effects:
        - Updates chat histories of all involved agents
        - Sends messages to individual agents and/or global channel
        - Logs communication in agent's chat history
        
    Notes:
        - Uses GPT-4 to generate contextually appropriate messages
        - Considers communication costs when deciding to send messages
        - Supports both direct and broadcast messages
    """
    if agent.cfg.envs.llm_model == "gpt":
        client = OpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
    else:
        client = OpenAI(
            api_key=agent.api_key,
            base_url=agent.cfg.envs.llm_url,
            timeout=3600,
        )    
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    elif agent.type == 3:
        type_string = 'Helicopter'

    team_comp_string = ""
    if global_data.get('firefighters'):
        team_comp_string += (
            f"{', '.join(f'AGENT_{a.id}' for a in global_data['firefighters'])} "
            f"{'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
        )
    if global_data.get('bulldozers'):
        team_comp_string += (
            f"{', '.join(f'AGENT_{a.id}' for a in global_data['bulldozers'])} "
            f"{'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
        )
    if global_data.get('drones'):
        team_comp_string += (
            f"{', '.join(f'AGENT_{a.id}' for a in global_data['drones'])} "
            f"{'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
        )
    if global_data.get('helicopters'):
        team_comp_string += (
            f"{', '.join(f'AGENT_{a.id}' for a in global_data['helicopters'])} "
            f"{'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"
        )

    past_string = "".join(f"{a}\n" for a in agent.past_actions)
    chat_string = ""
    for chat, msgs in agent.chat_history.items():
        chat_string += f"Chat with {chat}:\n"
        for source, content, time in msgs:
            chat_string += f"{source} (time: {time}): {content}\n"
        chat_string += "\n---\n\n"

    generate_communication_string = f"""
            You are AGENT_{agent.id}, a {type_string} Agent in a cooperative multi-agent robotic task. 
            Given your shared goal, chat history, and your progress and previous actions, please generate a list of short messages to members of your team in order to achieve the goal as possible.

            This is your team composition, including you:\n{team_comp_string}
            ---

            Your team's task is:\n{agent.current_task}
            ---

            Your status and observations:\n{agent.last_perception}
            ---

            Your past actions:\n{past_string}
            ---

            Your chats:\n\n{chat_string}
            ---


            You may send messages to individual agents or in a global channel. Think about the necessity of sending a message. There are costs to send messages. Provide your output in the following format. All names should be in all caps:

            <justification>(any justification or calculations)</justification>

            <RECIPIENT>'MESSAGE'</RECIPIENT>...<GLOBAL>'MESSAGE'</GLOBAL>

            For Example:

            <AGENT_A>message</AGENT_A>, <AGENT_C>message</AGENT_C>, <GLOBAL>message</GLOBAL>

            """
    system_message = f"""
                    You are Agent {agent.id}, an embodied agent in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    Your job is to communicate with the other agents in your team in order to achieve the goal as possible. Think about the necessity of sending a message/update. There are costs to send messages, even global ones."""
    user_message = generate_communication_string
    
    try:
        if global_data["large_model"].find("gpt") != -1:
            client = OpenAI(
                api_key=agent.api_key,
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
        content = response.choices[0].message.content
    except BadRequestError as e:
        print(f"BadRequestError for Agent_{agent.id}: {e}")
        global_data["api_calls"]+=1
        content = "ERROR EXECUTING COMMUNICATION"
    
    agent.log_chat("Sending Messages", [("user", user_message), ("assistant", content)])

    global_match = re.search(r"<GLOBAL>\s*(.*?)\s*</GLOBAL>", content, re.DOTALL)
    global_msg = global_match.group(1).strip() if global_match else None
    for recipient in global_data.get('agents', []):
        if recipient.id == agent.id:
            agent.add_message("GLOBAL", f"AGENT_{agent.id}", global_msg, global_data.get('time'))
            continue
        tag = f"<AGENT_{recipient.id}>"
        m = re.search(fr"{tag}\s*(.*?)\s*</AGENT_{recipient.id}>", content, re.DOTALL)
        if m:
            msg = m.group(1).strip()
            recipient.add_message(f"AGENT_{agent.id}", f"AGENT_{agent.id}", msg, global_data.get('time'))
            agent.add_message(f"AGENT_{recipient.id}", f"AGENT_{agent.id}", msg, global_data.get('time'))
        if global_msg:
            recipient.add_message("GLOBAL", f"AGENT_{agent.id}", global_msg, global_data.get('time'))


def action_round(agent: Agent, global_data: dict):
    """
    Generates and executes actions for an agent based on current state and communication.
    
    Args:
        agent (Agent): The agent to generate actions for
        global_data (dict): Global state containing team composition and environment info
        
    Effects:
        - Updates agent's action history
        - Executes generated actions in the environment
        - Logs action planning in agent's chat history
    """
    if agent.cfg.envs.llm_model == "gpt":
        client = OpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
    else:
        client = OpenAI(
            api_key=agent.api_key,
            base_url=agent.cfg.envs.llm_url,
            timeout=3600,
        )
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    elif agent.type == 3:
        type_string = 'Helicopter'

    past_string = "".join(f"{a}\n" for a in agent.past_actions)
    chat_string = ""
    for chat, msgs in agent.chat_history.items():
        chat_string += f"Chat with {chat}:\n"
        for source, content, time in msgs:
            chat_string += f"{source} (time: {time}): {content}\n"
        chat_string += "\n---\n\n"

    desc_path = os.path.join(
        'algorithms',
        'Embodied', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
    )
    if not os.path.exists(desc_path):
        desc_path = os.path.join(
        'crew_algorithms',
        'wildfire_alg',
        'algorithms',
        'Embodied', 'prompts', 'descriptions', f'{type_string.lower()}_description.txt'
    )
    
    with open(desc_path, 'r', encoding='utf-8') as file:
        abilities_string = file.read()

    generate_action_string = f"""
            You are AGENT_{agent.id}, a {type_string} Agent in a cooperative multi-agent robotic task. 

            Your team's task is:
            {agent.current_task}
            ---

            Your status and observations:
            {agent.last_perception}
            ---

            Your chat history:
            {chat_string}
            ---

            Your past actions:
            {past_string}
            ---

            Now your job is to provide the next best action for yourself.
            Remember, you are AGENT_{agent.id} a {type_string} Agent, located at {agent.last_position}.

            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be ONE and only ONE of these types. NO other responses are allowed.

{abilities_string}

            Provide your output in the following format:

           <justification>(any justification or calculations)</justification>

            <action>'MY NEXT ACTION'</action>

            Make sure you include enough details in your action such as explicit target coordinate locations. For example:

            <action>Move towards (500,500)</action>

            """
    system_message = f"""
                    You are the Agent {agent.id}, an embodied agent in a cooperative multi-agent robotic task. You are in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    Your job is to choose your next best action given your task, observations, past actions and chat history."""
    user_message = generate_action_string

    try:
        if global_data["large_model"].find("gpt") != -1:
            client = OpenAI(
                api_key=agent.api_key,
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
    except BadRequestError as e:
        print(f"BadRequestError for Agent_{agent.id}: {e}")
        global_data["api_calls"]+=1
        result = "ERROR EXECUTING ACTION"
    
    agent.log_chat("Choosing Action", [("user", user_message), ("assistant", result)])
    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    if not m:
        print("NO ACTION FOUND")
        return None
    action_str = m.group(1).strip()
    action = translate_action(action_str, agent.type, global_data)


    agent.log_chat("Executing Actions", [("system", action)])
    print(f"AGENT_{agent.id}: {action.description}")
    agent.past_actions.append(action.description)

    try:
        libraries = {
            0: Run_Firefighter_Action,
            1: Run_Bulldozer_Action,
            2: Run_Drone_Action,
            3: Run_Helicopter_Action
        }
        if agent.type in libraries:
            return libraries[agent.type](agent, action)
        else:
            print("Invalid agent type for action library")
            return [0, 0, 0]
    except:
        agent.log_chat("ERROR", [("system", "ERROR EXECUTING ACTION")])
        agent.past_actions.append(f"IDLE")
        return [0,0,0]

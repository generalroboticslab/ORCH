import re, os
from openai import OpenAI
from crew_algorithms.wildfire_alg.core.token_guard import guarded_chat_completion
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.COELA.agent import Agent


class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the COELA algorithm.
    
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
    prompt_path = f'algorithms/COELA/prompts/translator/{type_string}_translator.txt'
    if not os.path.exists(prompt_path):
        prompt_path = f"crew_algorithms/wildfire_alg/algorithms/COELA/prompts/translator/{type_string}_translator.txt"
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().replace("ACTION", option_str)
    # Call OpenAI
    if global_data["large_model"].find("gpt") != -1:
        client = OpenAI(
            api_key=global_data["agents"][0].api_key,
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':prompt}
            ],
            # temperature=0,
            return_none_on_context_error=True,
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
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':prompt}
            ],
            temperature=0,
            return_none_on_context_error=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    if response is None:
        return None
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    
    result = response.choices[0].message.content
    # if result.find("</think>") != -1:
    #     result = result.split("</think>")[-1].strip()

    
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


def generate_communication(agent: Agent, global_data: dict) -> str:
    """
    Generates communication messages for an agent to send to teammates.
    
    Args:
        agent (Agent): The agent generating the communication
        global_data (dict): Global state containing team composition and environment info
        
    Returns:
        str: Generated message to send to other agents, or None if no message needed
        
    Effects:
        - Logs communication in agent's chat history
    """

    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'

    team_comp_string = ""
    if len(global_data['firefighters']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if len(global_data['bulldozers']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if len(global_data['drones']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if len(global_data['helicopters']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"

    past_string = "".join(str(o.description) + "\n" for o in agent.past_options)
    chat_string = ''.join(f"{msg[0]}: \n{msg[1]}\n\n" for msg in agent.chat_history)

    generate_communication_string = f"""
            You are the communicator module of Agent {agent.id}, a {type_string} Agent in a cooperative multi-agent robotic task. 

            This is your team composition, including you:
            {team_comp_string}
            ---

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

            Your job is to propose a message to send to the chat/groupchat.

            
            Provide your output in the following format:

            <reasoning>(any reasoning or calculations)</reasoning>

            <message>'MESSAGE'</message>

            Note: The generated message should be accurate, helpful, and brief. Do not generate repetitive messages

            """

    system_message = f"""
                    You are the Communication Module of Agent {agent.id}, an embodied agent in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    Your job is to write a message to communicate to the other agents in the team. The generated message should be accurate, helpful, and brief. Do not generate repetitive messages"""
    if global_data["large_model"].find("gpt") != -1:
        client = OpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':generate_communication_string}
            ],
            # temperature=0,
            return_none_on_context_error=True,
            reasoning_effort="minimal",
        )
    else:
        client = OpenAI(
            api_key=agent.api_key,
            base_url=agent.cfg.envs.llm_url,
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':generate_communication_string}
            ],
            temperature=0,
            return_none_on_context_error=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    if response is None:
        return None
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content

    agent.log_chat('Proposing an Message', [('user', generate_communication_string), ('assistant', result)])
    m = re.search(r"<message>\s*(.*?)\s*</message>", result, re.DOTALL)
    if m:
        return m.group(1).strip()
    else:
        print("NO MESSAGE FOUND")
        return None

def generate_action(agent: Agent, global_data: dict, proposed_message: str) -> str:
    """
    Generates the next action for an agent based on current state and proposed communication.
    
    Args:
        agent (Agent): The agent to generate an action for
        global_data (dict): Global state containing team composition and environment info
        proposed_message (str): Message proposed to be sent to teammates
        
    Returns:
        str: Natural language description of the next action to take
        
    Effects:
        - Updates agent's action history
    """

    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'

    team_comp_string = ""
    if len(global_data['firefighters']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if len(global_data['bulldozers']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if len(global_data['drones']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if len(global_data['helicopters']) > 0:
        team_comp_string += f"{[f'Agent {a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"

    past_string = "".join(str(o.description) + "\n" for o in agent.past_options)
    chat_string = ''.join(f"{msg[0]}: \n{msg[1]}\n\n" for msg in agent.chat_history)


    #if in helicopter
    if agent.type==0 and agent.extra_variables[2]==1:
        abilities_string = "            - Do nothing since you are in a helicopter."
    else:
        path = f'algorithms/COELA/prompts/descriptions/{type_string.lower()}_description.txt'
        if not os.path.exists(path):
            path = f"crew_algorithms/wildfire_alg/algorithms/COELA/prompts/descriptions/{type_string.lower()}_description.txt"
        
        with open(path, 'r', encoding='utf-8') as f:
            abilities_string = f.read()

    generate_action_string = f"""
            You are Agent {agent.id}, a {type_string} Agent in a cooperative multi-agent robotic task.

            This is your team composition, including you:
            {team_comp_string}
            ---

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
            Remember, you are Agent {agent.id} a {type_string} Agent, located at {agent.last_position}.


            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be one of these types. NO other responses are allowed. Note that sending messages has a cost so think about the necessity of it.


            - [send message to groupchat] {proposed_message}
{abilities_string}

            
            Provide your output in the following format:

            <reasoning>(any reasoning or calculations)</reasoning>

            <action>'MY NEXT ACTION'</action>

            Include 'SEND_MESSAGE' in all caps like so, if and only if your action is to send the message. For example:

            <action>SEND_MESSAGE 'proposed_message'</action>
            
            """

    system_message = f"""
                    You are the Agent {agent.id}, an embodied agent in a cooperative multi-agent robotic task. You are in a {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    Your job is to choose your next best action given your task, observations, past actions and chat history."""
    if global_data["large_model"].find("gpt") != -1:
        client = OpenAI(
            api_key=agent.api_key,
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {'role':'system', 'content':system_message},
                {'role':'user', 'content':generate_action_string}
            ],
            # temperature=0,
            return_none_on_context_error=True,
            reasoning_effort="minimal",
        )
    else:
        client = OpenAI(
            api_key=agent.api_key,
            base_url=agent.cfg.envs.llm_url,
            timeout=3600,
        )
        response = guarded_chat_completion(client, 
            model=global_data["large_model"],
            messages=[
                {'role':'system','content':system_message},
                {'role':'user','content':generate_action_string}
            ],
            temperature=0,
            return_none_on_context_error=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    if response is None:
        return None
    global_data["api_calls"]+=1
    global_data["input_tokens"]+=response.usage.prompt_tokens
    global_data["output_tokens"]+=response.usage.completion_tokens
    result = response.choices[0].message.content
    agent.log_chat('Choosing an Action', [('user', generate_action_string), ('assistant', result)])
    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    if m:
        return m.group(1).strip()
    else:
        print("NO ACTION FOUND")
        return None

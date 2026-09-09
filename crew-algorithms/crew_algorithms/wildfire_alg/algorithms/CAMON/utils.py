import re, os
from openai import OpenAI, BadRequestError
from pydantic import BaseModel
from crew_algorithms.wildfire_alg.algorithms.CAMON.agent import Agent
from crew_algorithms.wildfire_alg.core.token_guard import guarded_chat_completion


class Action(BaseModel):
    """
    Represents an action that can be taken by an agent in the CAMON algorithm.
    
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
    prompt_path = f'algorithms/CAMON/prompts/translator/{type_string}_translator.txt'
    if not os.path.exists(prompt_path):
        prompt_path = f"crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/translator/{type_string}_translator.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read().replace("ACTION", option_str)
    # Call OpenAI
    
    try:
        if global_data["large_model"].find("gpt") != -1:
            client = OpenAI(
                api_key=global_data['leader_agent'].api_key,
                timeout=3600,
            )
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[
                    {'role':'system', 'content':system_message},
                    {'role':'user', 'content':prompt}
                ],
                label="CAMON translate_action",
                # temperature=0.7,
                reasoning_effort="minimal",
            )
        else:
            client = OpenAI(
                api_key=global_data['leader_agent'].api_key,
                base_url=global_data['leader_agent'].cfg.envs.llm_url,
                timeout=3600,
            )
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[
                    {'role':'system', 'content':system_message},
                    {'role':'user', 'content':prompt}
                ],
                label="CAMON translate_action",
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        global_data["api_calls"]+=1
        global_data["input_tokens"]+=response.usage.prompt_tokens
        global_data["output_tokens"]+=response.usage.completion_tokens
        result = response.choices[0].message.content
    except BadRequestError as e:
        global_data["api_calls"]+=1
        result = "ERROR PARSING ACTION"
    
    # Extract fields
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


def generate_plan(agent: Agent, global_data: dict) -> None:
    """
    Generates a plan for an agent based on current state and team composition.
    
    Args:
        agent (Agent): The agent to generate a plan for
        global_data (dict): Global state containing team composition and environment info
        
    Effects:
        - Updates agent's options list with next planned action
        - Logs planning process in agent's chat history
    """
    print(f"agent {agent.id}: generating actions")
    # Identify type
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'
    # Team composition
    team_comp_string = ''
    if global_data['firefighters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if global_data['bulldozers']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if global_data['drones']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if global_data['helicopters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"
    # Abilities descriptions

    team_abilities = ''

    desc_files = ['firefighter', 'bulldozer', 'drone', 'helicopter']

    for kind in desc_files:

        if global_data.get(kind+'s'):
            
            path = f'algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
            if not os.path.exists(path):
                path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
                
            with open(path, 'r', encoding='utf-8') as f:
                team_abilities += f.read()
                

    # Past and chat

    past_string = ''.join(str(o.description)+'\n' for o in agent.past_options)
    chat_string = ''.join(f"{time}: \n{msg}\n\n" for time, msg in agent.chat_history.items())
    global_str = []
    for data in global_data.items():
        if data[0].__contains__("AGENT"):
            global_str.append(data)

    # Prompt
    generate_plan_string = f"""
            You are AGENT_{agent.id} a {type_string} Agent, currently acting as the leader in a cooperative multi-agent robotic task. 
            This is your team composition, including you:
            {team_comp_string}

            Your team's current task is:
            {agent.current_task}
            ---

            Your past actions were:
            {past_string}

            ---
            This is your chat history with agents in your team:

            {chat_string}

            ---
            This is your teams'(including you) collective observations, locations, current actions, and past actions of all agents.
            {str(global_str)}

            Now your job is to provide the next best action for yourself, and OPTIONALLY: the next best action for any other agents. 
            Remember, you are AGENT_{agent.id} a {type_string} Agent, located at {agent.last_position}.

            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be one of these types. NO other responses are allowed.

            {team_abilities}

            Provide your output in the following format:

            <justification>(any justification or calculations)</justification>

            <action>'MY NEXT ACTION'</action>

            OPTIONAL-for other agents:

            <AGENT_ID-action>(AGENT_ID'S NEXT ACTION)<AGENT_ID-action>
            <AGENT_ID-message>(message to AGENTID)<AGENT_ID-message>

            For example:
            <AGENT_A-action>'action'</AGENT_A-action>
            <AGENT_A-message>'action'</AGENT_A-message>
            """
    # Call OpenAI
    if global_data["large_model"].find("gpt") != -1:
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
    system_message = f"""
                    You are AGENT_{agent.id}, currently acting as the leader in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    You have access to the collective observations and the progress of all agents. Your job is to plan the next best action for yourself, and OPTIONALLY: the next best action for any other agents."""
    user_message = generate_plan_string
    
    try:
        if global_data["large_model"].find("gpt") != -1:
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_message},{'role':'user','content':user_message}],
                label=f"CAMON generate_plan Agent_{agent.id}",
                # temperature=0.7,
                reasoning_effort="minimal",
            )
        else:
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_message},{'role':'user','content':user_message}],
                label=f"CAMON generate_plan Agent_{agent.id}",
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        global_data["api_calls"]+=1
        global_data["input_tokens"]+=response.usage.prompt_tokens
        global_data["output_tokens"]+=response.usage.completion_tokens
        result = response.choices[0].message.content
    except BadRequestError as e:
        global_data["api_calls"]+=1
        result = "ERROR PARSING ACTION"

    agent.log_chat("Generating Plan", [("user", user_message), ("assistant", result)])

    # Own action

    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    if m:
        option = translate_action(m.group(1), agent.type, global_data)
        agent.options = [option]
        print(f"agent {agent.id}: provided action: '{m.group(1)}' to itself")


    else:
        print("ERROR NO ACTION FOUND")
        return
    
    # Optional for others
    for a in global_data["agents"]:
        ma = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", result, re.DOTALL)
        mm = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", result, re.DOTALL)



        if ma and mm:
             # if in helicopter
            if a.type==0 and a.extra_variables[2]==1:
                a.options = [Action(type=0, param_1=0, param_2=0,description="ride helicopter")]
                continue
            print(f"agent {agent.id}: provided action '{ma.group(1)}'to {a.id}")
            opt = translate_action(ma.group(1), a.type, global_data)
            a.options = [opt]
            a.action_queue = []
            a.add_message(source=f"AGENT_{agent.id}", content=mm.group(1), time=global_data['time'])
            global_data.update({f'AGENT_{a.id}':{'name':f'AGENT_{a.id}','perception':a.last_perception,'position':a.last_position,'current_action':opt,'past_actions':a.past_options}})
    global_data['leader_agent'] = agent

    
def propose_plan(agent: Agent, global_data: dict) -> None:
    print(f"agent {agent.id}: proposing action to agent {global_data['leader_agent'].id}")

    # Identify type
    if agent.type == 0:
        type_string = 'Firefighter'
    elif agent.type == 1:
        type_string = 'Bulldozer'
    elif agent.type == 2:
        type_string = 'Drone'
    else:
        type_string = 'Helicopter'
    # Team composition
    
    team_comp_string = ''
    if global_data['firefighters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['firefighters']]} {'is' if len(global_data['firefighters'])==1 else 'are'} Firefighter Agents. Firefighter agents are general purpose agents with decent speed and observation capabilities. They can move, cut trees, spray water, and rescue civilians.\n\n"
    if global_data['bulldozers']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['bulldozers']]} {'is' if len(global_data['bulldozers'])==1 else 'are'} Bulldozer Agents. Bulldozer agents are specialized agents with exceptional tree-cutting abilities but limited speed.\n\n"
    if global_data['drones']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['drones']]} {'is' if len(global_data['drones'])==1 else 'are'} Drone Agents. Drone agents are specialized recon agents with exceptional speed and observations.\n\n"
    if global_data['helicopters']:
        team_comp_string += f"{[f'AGENT_{a.id}' for a in global_data['helicopters']]} {'is' if len(global_data['helicopters'])==1 else 'are'} Helicopter Agents. Helicopter agents are general support agents with exceptional speed and observations. They can move, pick up and drop off Firefighter Agents, and spray water\n\n"
    
    # Past and chat
    past_string = ''.join(str(o.description)+'\n' for o in agent.past_options)
    chat_string = ''.join(f"{time}: \n{msg}\n\n" for time, msg in agent.chat_history.items())

    # Description prompt
    desc_path = f'algorithms/CAMON/prompts/descriptions/{type_string.lower()}_description.txt'
    if not os.path.exists(desc_path):
        desc_path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{type_string.lower()}_description.txt'
    with open(desc_path, 'r', encoding='utf-8') as f:
        description_string = f.read()

    # Construct prompt
    proposal_str = f"""
                    You are AGENT_{agent.id}, an embodied {type_string} agent within a {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world and part of a collaboratve team of {len(global_data["agents"])} Agents.

                    This is your team's composition (including yourself):

                    {team_comp_string}

                    These are your current observations:

                    '{agent.last_perception}'

                    ---
                    This is your team's overall task: '{agent.current_task}'

                    Your past actions were:

                    {past_string}

                    ---
                    This is your chat history with agents in your team:

                    {chat_string}

                    ---
                    Your job is to propose your next action. These are your possible actions:
                    
                    {description_string}

                    This is a comprehensive list, so your action MUST be one of these types. NO other responses are allowed.

                    Provide your output in the following format:

                    <justification>(any justification or calculations)</justification>
                    <action>'MY NEXT ACTION'</action>

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
    system_msg = f"""
                    You are AGENT_{agent.id}, an embodied {type_string} agent.
                    You propose your next action based on your task, observations, past actions, and chat history.
                    """
    
    try:
        if agent.cfg.envs.llm_model == "gpt":
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_msg},{'role':'user','content':proposal_str}],
                label=f"CAMON propose_action Agent_{agent.id}",
                # temperature=0.7,
                reasoning_effort="minimal",
            )
        else:
            response = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_msg},{'role':'user','content':proposal_str}],
                label=f"CAMON propose_action Agent_{agent.id}",
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        global_data["api_calls"]+=1
        global_data["input_tokens"]+=response.usage.prompt_tokens
        global_data["output_tokens"]+=response.usage.completion_tokens
        result = response.choices[0].message.content
    except BadRequestError as e:
        result = "ERROR PARSING PROPOSAL"

    agent.log_chat('Proposing an Action', [('user', proposal_str), ('assistant', result)])

    m = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)

    if not m:
        print("ERROR NO ACTION FOUND")
        return
    proposed_action = m.group(1).strip()


    leader = global_data["leader_agent"]

    team_abilities = ''

    desc_files = ['firefighter', 'bulldozer', 'drone', 'helicopter']

    for kind in desc_files:

        if global_data.get(kind+'s'):

            path = f'algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
            if not os.path.exists(path):
                path = f'crew_algorithms/wildfire_alg/algorithms/CAMON/prompts/descriptions/{kind}_description.txt'
            
            with open(path, 'r', encoding='utf-8') as f:
                team_abilities += f.read()

    global_str = []
    for data in global_data.items():
        if data[0].__contains__("AGENT"):
            global_str.append(data)

    review_prompt = f"""

            You are AGENT_{leader.id}, currently acting as the leader in a cooperative multi-agent robotic task. 
            This is your team composition, including you:
            {team_comp_string}
            ---

            Your team's current task is:
            {agent.current_task}
            ---
            
            This is your teams'(including you) collective observations, locations, current actions, and past actions of all agents. Only you have all of this data.
            {str(global_str)}
            ---
            
            Your teammate AGENT_{agent.id}, a {type_string} Agent, is proposing a new action for itself:
            {proposed_action}
            ---


            Your job is to review this action and ACCEPT or REJECT it.

            Then provide the next best action for AGENT_{agent.id}, choosing a better one if REJECT or repeating/rewriting the proposed one if ACCEPT.
            Also send a message to AGENT_{agent.id} describing your choice.

            Additionally, you may announce information to other agents in your team with information.
            You may also choose to override actions for other agents as well. You must send a message to that agent if you do so. This interrupts their action, so only do this if you want to change their current action.
            

            These are all the possible actions for each type of agent. This is a comprehensive list, so the action MUST be one of these types. NO other responses are allowed.

            {team_abilities}

            
            Provide your output in the following format:

            <justification>(any justification or calculations)</justification>

            <decision> ACCEPT OR REJECT </decision>
            <action> AGENT_{agent.id}'s next action </action>
            <message> message to AGENT_{agent.id} </message>

            OPTIONAL-for other agents:

            <AGENT_ID-action>(AGENTID'S NEXT ACTION)<AGENT_ID-action>
            <AGENT_ID-message>(message to AGENTID)<AGENT_ID-message>

            For example: <AGENT_A-action>'action'</AGENT_A-action>

            Make sure 'action's are specific and include all information needed to execute, such as coordinates.
            
            YOU MUST HAVE AT LEAST THE <justification>, <decision>, <action>, <message> TAGS. SENDING MESSAGES OR PROPOSING ACTIONS TO OTHER AGENTS IS OPTIONAL.

            """
    
    system_msg = f"""
                    You are AGENT_{agent.id}, currently acting as the leader in a cooperative multi-agent robotic task. Your team is in a  {agent.cfg.envs.map_size} by {agent.cfg.envs.map_size} forest grid world that spans x:[0 to {agent.cfg.envs.map_size}] and y:[0 to {agent.cfg.envs.map_size}].
                    You have access to the collective observations and the progress of all agents. Your job is to review the proposed actions of your teammates and assign them actions.
                    """
    try:
        if agent.cfg.envs.llm_model == "gpt":
            client = OpenAI(
                api_key=leader.api_key,
                timeout=3600,
            )
            rev_resp = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_msg},{'role':'user','content':review_prompt}],
                label=f"CAMON review_proposals Agent_{agent.id}",
                # temperature=0.7,
                reasoning_effort="minimal",
            )
        else:
            client = OpenAI(
                api_key=leader.api_key,
                base_url=leader.cfg.envs.llm_url,
                timeout=3600,
            )
            rev_resp = guarded_chat_completion(
                client,
                model=global_data["large_model"],
                messages=[{'role':'system','content':system_msg},{'role':'user','content':review_prompt}],
                label=f"CAMON review_proposals Agent_{agent.id}",
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        global_data["api_calls"]+=1
        global_data["input_tokens"]+=rev_resp.usage.prompt_tokens
        global_data["output_tokens"]+=rev_resp.usage.completion_tokens
        review = rev_resp.choices[0].message.content
    except BadRequestError as e:
        review = "ERROR PARSING REVIEW"
        
    agent.log_chat("Review Proposal", [('user', review_prompt), ('assistant', review)])

    action_match = re.search(r"<action>\s*(.*?)\s*</action>", result, re.DOTALL)
    message_match = re.search(r"<message>\s*(.*?)\s*</message>", result, re.DOTALL)

    if action_match:
        action_str= action_match.group(1)

        option = translate_action(action_str, type = agent.type, global_data=global_data)
        agent.options = [option]
        if message_match:
            message_str= message_match.group(1)
            agent.add_message(source=f"AGENT_{leader.id}", content=message_str, time=global_data["time"])

            print(f"agent {leader.id}: {message_str}")
        print(f"agent {leader.id}: provided action: '{action_str}' to {agent.id}")


        for a in global_data["agents"]:

            agent_action_match = re.search(fr"<AGENT_{a.id}-action>\s*(.*?)\s*</AGENT_{a.id}-action>", result, re.DOTALL)
            agent_message_match = re.search(fr"<AGENT_{a.id}-message>\s*(.*?)\s*</AGENT_{a.id}-message>", result, re.DOTALL)

            
            
            if agent_action_match and agent_message_match:

                if a.type==0 and a.extra_variables[2]==1:
                    a.options = [Action(type=0, param_1=0, param_2=0,description="ride helicopter")]
                    continue

                agent_action_str= agent_action_match.group(1)
                agent_message_str= agent_message_match.group(1)
                print(f"agent {leader.id}: provided action: '{agent_action_str}' to {a.id}")

                option = translate_action(agent_action_str, type = a.type, global_data=global_data)
                a.options = [option]
                a.action_queue = []
                a.add_message(source = f"AGENT_{leader.id}", content=agent_message_str, time = global_data["time"])

                agent_data = {'name': f'AGENT_{a.id}', 
                    'perception': a.last_perception, 
                    'position': a.last_position, 
                    'current_action': a.options[0] if len(a.options)>0 else "IDLE", 
                    'past_actions': a.past_options}
                global_data.update({f'AGENT_{a.id}': agent_data})
                
    else:
        print("ERROR NO ACTION FOUND")

    global_data.update({"leader_agent": agent})
    print(f"agent {agent.id}: is now leader")

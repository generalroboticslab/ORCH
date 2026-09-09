from openai import OpenAI
import os
from pydantic import BaseModel
import numpy as np
import certifi



class Action(BaseModel):
    done: bool
    action: int
    x: int
    y: int
    explanation : str

    def print_action(self)->None:
        print([self.done, self.action, self.x, self.y, self.explanation])

class Critique(BaseModel):
    judge: bool
    explanation : str

class OptionSequence(BaseModel):
    actions: list[str]
    reasonings: list[str]


class Option(BaseModel):
    type: int
    param_1: int
    param_2: int
    description: str
    completion_condition: str
    def print_option(self)->None:
        print([self.type, self.param_1, self.param_2, self.description, self.completion_condition])


class AdaptiveOption(BaseModel):
    adaptive: bool
    type: int
    param_1: int
    param_2: int
    description: str
    completion_condition: str
    def print_option(self)->None:
        print([self.adaptive, self.type, self.param_1, self.param_2, self.description, self.completion_condition])

class Options(BaseModel):
    actions: list[Option]

class AdaptiveOptions(BaseModel):
    actions: list[AdaptiveOption]



os.environ["SSL_CERT_FILE"] = certifi.where()


# api_key = os.environ['OPENAI_API_KEY']
# client = OpenAI(api_key=api_key)

api_key = os.environ['QWEN_API_KEY']
client = OpenAI(
    api_key=api_key,
    base_url="https://localhost:8000/v1",
    )


def request(agent):

    sub_goal = agent.options[0]
    task = sub_goal.description
    completion_condition = sub_goal.completion_condition

    context_content = agent.actor_context
    prompt_content = agent.actor_prompt
    match(agent.type):
        case 0:
            maprange = 10
        case 1:
            maprange = 10
        case 2:
            maprange = 30
        case 3:
            maprange = 40

    memory_string="\n".join(agent.memory_buffer)
    if len(memory_string)==0:
        memory_string="No Previous Actions"
    prompt_content = prompt_content.replace("POSITION", str(agent.last_position))
    prompt_content = prompt_content.replace("MAPRANGE", f"({agent.last_position[0]-maprange}-{agent.last_position[0]+maprange}, {agent.last_position[1]-maprange}-{agent.last_position[1]+maprange}")
    prompt_content = prompt_content.replace("OBS", agent.last_perception)
    prompt_content = prompt_content.replace("CURRCELL", agent.curr_cell)
    prompt_content = prompt_content.replace("TASK", task)
    prompt_content = prompt_content.replace("MEMORY", memory_string)
    prompt_content = prompt_content.replace("COMPLETION", completion_condition)


    completion = client.chat.completions.create(
        model=agent.cfg.llms.actor_model,
        messages=[
            {"role": "system", "content": f"{context_content}"},
            {
                "role": "user",
                "content": f"{prompt_content}",
            }
        ],
        response_format={
           'type': 'json_schema',
           'json_schema': 
              {
                "name":"whocares", 
                "schema": Action.model_json_schema()
              }
         },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} 
    )

    action = Action.parse_raw(completion.choices[0].message.content)
    if agent.cfg.llms.verbose:
        action.print_action()
    return action

def critique_response(agent, action):
        match(agent.type):
            case 0:
                maprange = 10
            case 1:
                maprange = 10
            case 2:
                maprange = 30
            case 3:
                maprange = 40

        critic_context_content = agent.critic_context
        critic_prompt_content = agent.critic_prompt
        sub_goal = agent.options[0]
        task = sub_goal[0]
        completion_condition = sub_goal[1]

        if action.done:
            action_string = "No action because the task is complete. I will await further tasks. " +action.explanation 
        else:
            action_string = f"Type: {action.action}, Explanation: {action.explanation}"


        memory_string="\n".join(agent.memory_buffer)
        if len(memory_string)==0:
            memory_string="No Previous Actions"
        critic_prompt_content = critic_prompt_content.replace("POSITION", str(agent.last_position))
        critic_prompt_content = critic_prompt_content.replace("MAPRANGE", f"({agent.last_position[0]-maprange}-{agent.last_position[0]+maprange}, {agent.last_position[1]-maprange}-{agent.last_position[1]+maprange}")
        critic_prompt_content = critic_prompt_content.replace("OBS", agent.last_perception)
        critic_prompt_content = critic_prompt_content.replace("CURRCELL", agent.curr_cell)
        critic_prompt_content = critic_prompt_content.replace("TASK", task)
        critic_prompt_content = critic_prompt_content.replace("ACTION", action_string)
        critic_prompt_content = critic_prompt_content.replace("MEMORY", memory_string)
        critic_prompt_content = critic_prompt_content.replace("COMPLETION", completion_condition)
        if agent.cfg.llms.verbose:
            #print(critic_prompt_content)
            pass


        critic = client.chat.completions.create(
            model=agent.cfg.llms.critic_model,
            messages=[
                {"role": "system", "content": f"{critic_context_content}"},
                {
                    "role": "user",
                    "content": f"{critic_prompt_content}",
                }
            ],
            response_format={
            'type': 'json_schema',
            'json_schema': 
                {
                    "name":"whocares", 
                    "schema": Critique.model_json_schema()
                }
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        try:
            critique = Critique.parse_raw(critic.choices[0].message.content)
        except:
            print("error parsing")
            return critique_response(agent, action)

        
        if agent.cfg.llms.verbose:
            print((critique.judge, critique.explanation))
        return critique

def regenerate_response(agent, action, critique):


    context_content = agent.actor_context
    prompt_content = agent.actor_prompt
    match(agent.type):
        case 0:
            maprange = 10
        case 1:
            maprange = 10
        case 2:
            maprange = 30
        case 3:
            maprange = 40

    sub_goal = agent.subgoals[0]
    task = sub_goal[0]
    completion_condition = sub_goal[1]


    memory_string="\n".join(agent.memory_buffer)
    if len(memory_string)==0:
        memory_string="No Previous Actions"

    prompt_content = prompt_content.replace("POSITION", str(agent.last_position))
    prompt_content = prompt_content.replace("MAPRANGE", f"({agent.last_position[0]-maprange}-{agent.last_position[0]+maprange}, {agent.last_position[1]-maprange}-{agent.last_position[1]+maprange}")
    prompt_content = prompt_content.replace("OBS", agent.last_perception)
    prompt_content = prompt_content.replace("CURRCELL", agent.curr_cell)
    prompt_content = prompt_content.replace("TASK", task)
    prompt_content = prompt_content.replace("MEMORY", memory_string)
    prompt_content = prompt_content.replace("COMPLETION", completion_condition)
    if agent.cfg.llms.verbose:
        #print(prompt_content)
        pass

    critical_string = critique.explanation

    newcompletion = client.chat.completions.create(
        model=agent.cfg.llms.actor_model,
        messages=[
            {"role": "system", "content": f"{context_content}"},
            {
                "role": "user",
                "content": f"{prompt_content}",
            },
            {
                "role": "assistant",
                "content": str([action.done, action.action, action.x, action.y]),
            },
            {
                "role": "user",
                "content": critical_string
            },
        ],
        response_format={
        'type': 'json_schema',
        'json_schema': 
            {
                "name":"whocares", 
                "schema": Action.model_json_schema()
            }
        },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
    
    action = Action.parse_raw(newcompletion.choices[0].message.content)
    if agent.cfg.llms.verbose:
        action.print_action()

    return action

def request_options(agent):

    context_content = agent.manager_context
    prompt_content = agent.manager_prompt
    match(agent.type):
        case 0:
            maprange = 10
        case 1:
            maprange = 10
        case 2:
            maprange = 30
        case 3:
            maprange = 40

    prompt_content = prompt_content.replace("MAPSIZE-1", str(agent.cfg.envs.map_size-1))
    prompt_content = prompt_content.replace("MAPSIZE", str(agent.cfg.envs.map_size))
    prompt_content = prompt_content.replace("POSITION", str(agent.last_position))
    prompt_content = prompt_content.replace("MAPRANGE", f"({agent.last_position[0]-maprange} to {agent.last_position[0]+maprange}, {agent.last_position[1]-maprange} to {agent.last_position[1]+maprange}")
    prompt_content = prompt_content.replace("OBS", agent.last_perception)
    prompt_content = prompt_content.replace("CURRCELL", agent.curr_cell)
    prompt_content = prompt_content.replace("TASK", agent.current_task)


    option_sequence = None

    while not option_sequence:
        try:
            completion = client.chat.completions.create(
                model=agent.cfg.llms.planner_model,
                messages=[
                    {"role": "user", "content": f"{context_content}"},
                    {
                        "role": "user",
                        "content": f"{prompt_content}",
                    }
                ],
                response_format={
                'type': 'json_schema',
                'json_schema': 
                    {
                        "name":"whocares", 
                        "schema": OptionSequence.model_json_schema()
                    }
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            test_option_sequence = OptionSequence.parse_raw(completion.choices[0].message.content)
            option_sequence = test_option_sequence
        except:
            print("trying to generate option again")



    return option_sequence

def translate_options(agent, option_sequence):

    context_content = agent.translator_context
    prompt_content = agent.translator_prompt


    optionstring ="\n".join(option_sequence.actions)

    prompt_content = prompt_content.replace("ACTIONS", str(optionstring))

    completion = client.chat.completions.create(
        model=agent.cfg.llms.translator_model,
        messages=[
            {"role": "system", "content": f"{context_content}"},
            {
                "role": "user",
                "content": f"{prompt_content}",
            }
        ],
        response_format={
           'type': 'json_schema',
           'json_schema': 
              {
                "name":"whocares", 
                "schema": AdaptiveOptions.model_json_schema() if agent.cfg.llms.use_adaptive_options else Options.model_json_schema()
              }
         },
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} 
    )
    print("translated options")
    print(completion.choices[0].message.content)

    if agent.cfg.llms.use_adaptive_options:
        options = AdaptiveOptions.parse_raw(completion.choices[0].message.content)
    else:
        options = Options.parse_raw(completion.choices[0].message.content)


    return options
    



# Embodied Agents

## Philosophical Bits

We believe the essence of intelligence emerges from its dynamic interactions with the external environment, where the use of various tools becomes a pivotal factor in its development and manifestation.

The `EmbodiedAgent()` in CAMEL is an advanced conversational agent that leverages **code interpreters** and **tool agents** (*e.g.*, `HuggingFaceToolAgent()`) to execute diverse tasks efficiently. This agent represents a blend of advanced programming and AI capabilities, and is able to interact and respond within a dynamic environment.


## Quick Start


### 🕹 Step 0: Prepartions
```python
from camel.agents import EmbodiedAgent
from camel.generators import SystemMessageGenerator as sys_msg_gen
from camel.messages import BaseMessage as bm
from camel.types import RoleType
```

### 🕹 Step 1: Define the Role
We first need to set up the necessary information.
```python
# Set the role name and the task
role = 'Programmer'
task = 'Writing and executing codes.'

# Create the meta_dict and the role_tuple
meta_dict = dict(role=role, task=task)
role_tuple = (role, RoleType.EMBODIMENT)
```
The `meta_dict` and `role_type` will be used to generate the system message.
```python
# Generate the system message based on this
sys_msg = sys_msg_gen().from_dict(meta_dict=meta_dict,
                                  role_tuple=role_tuple)
```
### 🕹 Step 2: Initialize the Agent 🐫
Based on the system message, we are ready to initialize our embodied agent.
```python
# Feed the system message to the agent
embodied_agent = EmbodiedAgent(system_message=sys_msg,
                               tool_agents=None,
                               code_interpreter=None,
                               verbose=True)
```
Be aware that the default argument values for `tool_agents` and `code_interpreter` are `None`, and the underlying code interpreter is using the `SubProcessInterpreter()`, which handles the execution of code in Python and Bash within a subprocess.


### 🕹 Step 3: Interact with the Agent with `.step()`
Use the base message wrapper to generate the user message.
```python
usr_msg = bm.make_user_message(
    role_name='user',
    content=('1. write a bash script to install numpy. '
             '2. then write a python script to compute '
             'the dot product of [8, 9] and [5, 4], '
             'and print the result. '
             '3. then write a script to search for '
             'the weather at london with wttr.in/london.'))
```
And feed that into your agents:
```python
response = embodied_agent.step(usr_msg)
```
Under the hood, the agent will perform multiple actions within its action space in the OS to fulfill the user request. It will compose code to implement the action – no worries, it will ask for your permission before execution.

Ideally you should get the output similar to this, if you allow the agent to perform actions:
```python
print(response.msg.content)

>>> Executing code block 0: {Requirement already satisfied: numpy in ...}
>>> Executing code block 1: {76}
>>> Executing code block 2: {
>>> Weather report: london
>>>
>>>       \   /     Sunny
>>>        .-.      +4(1) °C
>>>     ― (   ) ―   ↘ 30 km/h
>>>        `-’      10 km
>>>       /   \     0.0 mm
>>> }
```
Let's celebrate the sunny day in London with the agent! : )

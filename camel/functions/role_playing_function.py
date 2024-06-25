# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import List

from colorama import Fore

from camel.configs import ChatGPTConfig
from camel.functions.openai_function import OpenAIFunction
from camel.societies import RolePlaying
from camel.types import ModelType
from camel.utils import print_text_animated


def role_playing_function(
    task_prompt: str,
    input: str,
    user_role: str,
    assistant_role: str,
    output_language: str,
) -> str:
    r"""Every time you think it is difficult or it is uncertain to complete a
    task or act, this role-playing function can help. Input your task, and the
    roles.

    Args:
        task_prompt (str): The task prompt for the AI assistant.
        input (str): The input information for the task.
        user_role (str): The role of the AI user.
        assistant_role (str): The role of the AI assistant.
        output_language (str): The language of expected response.

    Returns:
        str: The response for the task.
    """
    print_text_animated(
        Fore.YELLOW + "Apply role-playing to resolve the subtask:\n"
        f"{task_prompt}\n"
        f"Input information of the task.\n{input}\n",
        0.001,
    )
    task_prompt += "\n" + input

    model_type = ModelType.GPT_4O  # Default model
    agent_kwargs = {
        role: dict(
            model_type=model_type,
            model_config=ChatGPTConfig(max_tokens=4096, temperature=0.7),
        )
        for role in ["assistant", "user"]
    }

    role_play_session = RolePlaying(
        assistant_role_name=assistant_role,
        assistant_agent_kwargs=agent_kwargs["assistant"],
        user_role_name=user_role,
        user_agent_kwargs=agent_kwargs["user"],
        task_prompt=task_prompt,
        with_task_specify=False,
        output_language=output_language,
    )

    chat_turn_limit, n = 50, 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        print_text_animated(
            Fore.YELLOW + f"AI User:\n\n{user_response.msg.content}\n", 0.001
        )
        print_text_animated(
            Fore.YELLOW + "AI Assistant:\n\n"
            f"{assistant_response.msg.content}\n",
            0.001,
        )

        if (
            "CAMEL_TASK_DONE" in user_response.msg.content
            or "CAMEL_TASK_DONE" in assistant_response.msg.content
        ):
            break

        input_msg = assistant_response.msg

    assistant_reply: str = task_prompt + "\n"
    for record in role_play_session.assistant_agent.memory.retrieve()[-4:]:
        assistant_reply += (
            record.memory_record.message.content.replace("CAMEL_TASK_DONE", "")
            + "\n"
        )

    return assistant_reply


ROLE_PLAYING_FUNS: List[OpenAIFunction] = [
    OpenAIFunction(func) for func in [role_playing_function]
]

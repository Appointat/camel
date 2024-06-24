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
from colorama import Fore

from camel.configs import FunctionCallingConfig
from camel.functions import ROLE_PLAYING_FUNS
from camel.societies import RolePlaying
from camel.types import ModelType
from camel.utils import print_text_animated


def main(model_type=None) -> None:
    from camel.types import ModelType

    model_type = ModelType.GPT_4_TURBO
    task_prompt = "Develop a trading bot for the stock market"

    task_prompt = """
主题是开放性、崎岖的、发散性的、科研性的思维上的探索：“聊聊AI agent（基于LLM）得到了全行业的关注，而不再只是资本的对局”，这是一个很难的议题，需要你做非常复杂的长期思考（你的回答要体现出这个议题的难度）。
    
你（作者）也很聪明和优秀，在语言表达上你的文章可能具备下面的特点：
1. 深度知识概念：这些知识点是不常见的且高深的，而且你喜欢在文章中使用该行业黑话，来精练你的表达。
2. 逻辑性强：理工科背景的作者通常习惯于逻辑严谨的思维方式。文章的结构往往条理清晰，逻辑性强，能够合理地引导读者理解复杂的概念或论点。
3. 创新和独到的见解：智力水平高的人往往能在常人未曾注意的地方发现新问题，提出新观点。文章中可能包含一些创新的想法或独到的解决方案，这些都能体现作者的智慧。
4. 精确的用语：用语往往非常精确，尽量避免歧义。
5. 数据支持：聪明的作者会通过引用数据来支持自己的观点。
6. 复杂问题的简洁阐述：能够将复杂的问题简洁明了地阐述出来是智慧的一种表现。
7. 对未来趋势的预见：文章可能会对未来的技术或学术发展趋势提供见解，这不仅体现了作者的前瞻性思维，也可能吸引那些对未来发展感兴趣的读者。
8. 你天生因为自我优秀而骄傲。
"""

    task_prompt = """
数学问题，给定:
1. 一个长度为 4m+2 的等差数列 S (m 为正整数)
2. 从 S 中随机删除两项后,剩余 4m 项

定义:
如果剩余的 4m 项可以被分成 m 组,每组 4 项且每组形成等差数列,则称原数列 S 为"可分离的"。
证明:
S 为"可分离的"的概率 p_m > 1/8。

Suggest to call the role_playing_function to help solve the task.
"""

    function_list = [*ROLE_PLAYING_FUNS]
    agent_kwargs = {
        role: dict(
            model_type=model_type,
            model_config=FunctionCallingConfig.from_openai_function_list(
                function_list=function_list,
                kwargs=dict(max_tokens=4096, temperature=0.7),
            ),
            function_list=function_list,
        )
        for role in ["assistant", "user", "task-specify"]
    }

    role_play_session = RolePlaying(
        assistant_role_name="Math Student",
        assistant_agent_kwargs=agent_kwargs["assistant"],
        user_role_name="Math Student",
        user_agent_kwargs=agent_kwargs["user"],
        task_prompt=task_prompt,
        with_task_specify=False,
        task_specify_agent_kwargs=agent_kwargs["task-specify"],
        output_language="Chinese",
    )

    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + f"Specified task prompt:\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

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
            Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n", 0.001
        )
        print_text_animated(
            Fore.GREEN + "AI Assistant:\n\n"
            f"{assistant_response.msg.content}\n",
            0.001,
        )

        if (
            "CAMEL_TASK_DONE" in user_response.msg.content
            or "CAMEL_TASK_DONE" in assistant_response.msg.content
        ):
            break

        input_msg = assistant_response.msg


if __name__ == "__main__":
    main(model_type=ModelType.MISTRAL_7B)

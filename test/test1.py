import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

from src.PromptBuilder import PromptBuilder
from example.examples import examples

prompt_builder = PromptBuilder()


print("模板一\n")
prompt_ = prompt_builder.create_chat_prompt_template(
    system_prompt="你是一个全能的聊天助手",
    examples=examples,
)
print(prompt_.format(question="唐太宗李世民是一个什么样的人"))

print("模板二\n")
few_shot_chat_prompt = prompt_builder.create_few_shot_chat_prompt_template(
    system_prompt="你是一个全能的{type}助手",
    examples=examples
)

print(few_shot_chat_prompt.format(type="聊天", question="唐太宗李世民是一个什么样的人"))

print("模板三\n")
few_shot_chat_prompt = prompt_builder.create_few_shot_chat_prompt_template_with_selector(
    system_prompt="你是一个全能的{type}助手",
    examples=examples,
    example_num=2
)

print(few_shot_chat_prompt.format(type="聊天", question="唐太宗李世民是一个什么样的人"))

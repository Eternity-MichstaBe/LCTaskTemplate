import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    

from src.PromptBuilder import PromptBuilder

prompt_builder = PromptBuilder()

prompt = prompt_builder.create_chat_prompt_template(
    system_prompt="你是一个AI助手，请回答用户的问题。",
    examples=[
        {"question": "你好", "answer": "你好，我是AI助手。"},
        {"question": "你擅长什么？", "answer": "我擅长回答问题。"}
    ]
)

print(prompt)
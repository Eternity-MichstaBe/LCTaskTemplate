"""
聊天机器人、支持历史跟踪
"""

import os
import sys
import asyncio
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool, StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMChainBuilder import LLMChainBuilder
from src.LLMConfig import get_llm_configs
from src.PromptManager import SystemPromptManager

llmconfig = get_llm_configs(
    system_prompt="你是一个专业的聊天机器人，擅长于用户沟通，回答用户的问题",
    output_parser=StrOutputParser(),
    memory=True  # 启用记忆功能
)

# 创建聊天链
chat_chain = LLMChainBuilder().create_few_shot_chat_chain_with_selector(llmconfig, example_num=3)

REDIS_URL = "redis://localhost:6379/0"

def get_chat_history(user_id: str, session_id: str) -> RedisChatMessageHistory:
    """获取或创建聊天会话历史"""
    history = RedisChatMessageHistory(session_id=user_id + "-" + session_id, url=REDIS_URL)
    # 限制历史记录数量，只保留最近的10轮对话消息
    messages = history.messages
    if len(messages) > 20:
        # 清空所有消息
        history.clear()
        # 只添加最近的10条消息
        for message in messages[-20:]:
            history.add_message(message)

    return history

# 创建带有消息历史的可运行对象 - 普通聊天
chat_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户唯一标识",
            default="",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="会话唯一标识",
            default="",
            is_shared=True
        )
    ]
)

async def async_stream_chat_session():
    """异步流式聊天会话"""
    print("欢迎使用AI聊天助手（异步流式版本），输入'退出'结束对话")
    
    user_id = "zby"
    session_id = "session_1"
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break
        
        print("AI助手: ", end="", flush=True)
        
        # 使用普通聊天模式
        async for chunk in chat_with_history.astream(
            {"question": user_input},
            config={"configurable": {"user_id": user_id, "session_id": session_id}}
        ):
            print(chunk, end="", flush=True)

asyncio.run(async_stream_chat_session())

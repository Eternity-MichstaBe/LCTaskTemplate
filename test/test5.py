"""
记忆存储
"""

import os
import sys
import asyncio
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMChainBuilder import LLMChainBuilder
from src.LLMConfig import get_llm_configs
from src.LLMAgentBuilder import LLMAgentBuilder
from src.PromptManager import SystemPromptManager
from example.examples import examples

# 定义工具函数
def calculator_tool(expression: str) -> str:
    """计算器工具"""
    try:
        return f"计算结果: {eval(expression)}"
    except Exception as e:
        return f"格式错误请检查! 错误: {str(e)}"

def search_tool(query: str) -> str:
    """搜索引擎工具"""
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        if not results:
            return "未找到相关信息"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"结果{i}:\n"
                f"标题: {result.get('title', '无标题')}\n"
                f"内容: {result.get('content', '无内容')}\n"
                f"链接: {result.get('url', '无链接')}\n"
            )
        
        return "搜索结果:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"搜索出错: {str(e)}"

# 配置LLM - 普通聊天模式
llmconfig = get_llm_configs(
    examples=examples,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
    output_parser=StrOutputParser(),
    memory=True  # 启用记忆功能
)

# 配置LLM - Agent模式
agent_config = get_llm_configs(
    system_prompt=SystemPromptManager.get_agent_system_prompt(),
    agent=True,
    memory=True  # 启用记忆功能
)

# 创建聊天链
chat_chain = LLMChainBuilder().create_chat_chain(llmconfig)

# 创建Agent构建器
llm_agent_builder = LLMAgentBuilder(agent_config)

# 定义工具列表
tools = [
    Tool.from_function(
        name="calculator",
        func=calculator_tool,
        description="用于数学计算的工具，输入应为字符串形式的数学表达式"
    ),
    Tool.from_function(
        name="search_engine",
        func=search_tool,
        description="用于搜索信息的工具，输入应为字符串形式的问题或关键字"
    )
]

# 添加工具
llm_agent_builder.add_tools(tools)

# 创建Agent执行器
agent_executor = llm_agent_builder.create_agent_executor(verbose=True)

# 使用字典存储会话历史
chat_history = {}
agent_history = {}

def get_chat_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    """获取或创建聊天会话历史"""
    session_key = (user_id, session_id)
    if session_key not in chat_history:
        chat_history[session_key] = ChatMessageHistory()
    return chat_history[session_key]

def get_agent_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    """获取或创建Agent会话历史"""
    session_key = (user_id, session_id)
    if session_key not in agent_history:
        agent_history[session_key] = ChatMessageHistory()
    return agent_history[session_key]

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

# 创建带有消息历史的可运行对象 - Agent
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_agent_history,
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
    print("输入'agent模式'切换到Agent模式，输入'聊天模式'切换到普通聊天模式")
    
    user_id = "user123"
    session_id = "session456"
    
    # 默认使用普通聊天模式
    is_agent_mode = False
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break
        
        if user_input.lower() == 'agent模式':
            is_agent_mode = True
            print("已切换到Agent模式，可以使用工具功能")
            continue
            
        if user_input.lower() == '聊天模式':
            is_agent_mode = False
            print("已切换到普通聊天模式")
            continue
        
        print("\nAI助手: ", end="", flush=True)
        
        if is_agent_mode:
            # 使用Agent模式
            tools_desc = llm_agent_builder.get_tools_description()
            response = await agent_with_history.ainvoke(
                {"tools": tools_desc, "question": user_input},
                config={"configurable": {"user_id": user_id, "session_id": session_id}}
            )
            print(response['output'])
        else:
            # 使用普通聊天模式
            async for chunk in chat_with_history.astream(
                {"question": user_input},
                config={"configurable": {"user_id": user_id, "session_id": session_id}}
            ):
                print(chunk, end="", flush=True)

        print("\n历史记录1:\n", chat_history)
        print("\n历史记录2:\n", agent_history)


asyncio.run(async_stream_chat_session())

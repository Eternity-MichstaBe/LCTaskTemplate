"""
多模态agent测试
"""

import httpx
import base64
import os
import sys
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_message_histories import RedisChatMessageHistory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.LLMConfig import get_llm_configs
from src.llm.PromptManager import SystemPromptManager
from src.llm.LLMAgentBuilder import LLMAgentBuilder

# 启用 LangChain 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Test"


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
    
# 初始化Agent
llm_agent_builder = LLMAgentBuilder(
    get_llm_configs(
        system_prompt=SystemPromptManager.get_agent_system_prompt(),
        agent=True,
        memory=True
    )
)
    
# 定义工具列表
tools = [
    Tool.from_function(
        name="calculator",
        func=calculator_tool,
        description="用于数学计算的工具，输入应为字符串形式的数学表达式，调用工具对应的方法calculator_tool获取工具输出"
    ),
    Tool.from_function(
        name="search_engine",
        func=search_tool,
        description="用于在网页中搜索获取实时信息，输入应为字符串形式的问题或关键字，调用工具对应的方法search_tool获取工具输出"
    )
]

# 添加工具
llm_agent_builder.add_tools(tools)

# set_verbose(True)
# set_debug(True)

# 创建 Agent
agent_executor = llm_agent_builder.create_multimodal_agent_executor(verbose=True)
tools_desc = llm_agent_builder.get_tools_description()

REDIS_URL = "redis://localhost:6379/0"

def get_agent_history(user_id: str, session_id: str) -> RedisChatMessageHistory:
    """获取或创建Agent会话历史"""
    return RedisChatMessageHistory(session_id=user_id + "-" + session_id, url=REDIS_URL)

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

# 读取图像并编码为 base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_url_ = "https://th.bing.com/th/id/OIP.koy6VQmYMo9tMjDWe9xvQwHaE8?rs=1&pid=ImgDetMain"
image_data_ = base64.b64encode(httpx.get(image_url_).content).decode("utf-8")

res = agent_with_history.invoke(
    {"tools": tools_desc, "question": "请判断这张图片里是什么，然后去搜索更多最新的信息返回给我，注意按照输出模板的格式返回", "image_data": image_data_},
    config={
        "configurable": {"user_id": "zby", "session_id": "session_1"}
    }
)

print(res['output'])
"""
agent模式和langsmith服务监测
"""

import os
import sys
from langchain_core.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.globals import set_debug, set_verbose
from pydantic import BaseModel, Field
from langchain_core.tools import ToolException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import get_llm_configs
from src.PromptManager import SystemPromptManager
from src.LLMAgentBuilder import LLMAgentBuilder

# 启用 LangChain 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Test"

# 定义pydantic输入验证
class CalculatorInput(BaseModel):
    expression: str = Field(description="算术表达式")

class SearchInput(BaseModel):
    query: str = Field(description="查询的关键词或内容")

# 定义工具
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

def _handle_error(error: ToolException) -> str:
    return f"工具执行期间发生错误：{error.args[0]}"
    
# 定义工具列表
tools = [
    StructuredTool.from_function(
        name="calculator",
        func=calculator_tool,
        description="用于数学计算的工具，输入应为字符串形式的数学表达式",
        args_schema=CalculatorInput,
        # return_direct=True,
        handle_tool_error=_handle_error
    ),
    StructuredTool.from_function(
        name="search_engine",
        func=search_tool,
        description="用于在网页中搜索获取实时信息，输入应为字符串形式的问题或关键字",
        args_schema=SearchInput,
        # return_direct=True,
        handle_tool_error=_handle_error
    )
]

# 添加工具
llm_agent_builder.add_tools(tools)

# set_verbose(True)
# set_debug(True)

# 创建 Agent
agent_executor = llm_agent_builder.create_agent_executor(verbose=True)

# 执行查询
query = "我打算今年7月份去云南旅游，请帮我制作一份旅游路线和攻略，你需要考虑天气、交通、住宿、饮食等因素，请根据最新的信息完成该任务。"
tools_desc = llm_agent_builder.get_tools_description()

print(agent_executor.invoke({"tools": tools_desc, "question": query})['output'])

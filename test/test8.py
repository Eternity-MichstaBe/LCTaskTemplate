"""
agent模式和langsmith服务监测
"""

import os
import sys
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.globals import set_debug, set_verbose
import httpx
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import get_llm_configs
from src.PromptManager import SystemPromptManager
from src.LLMAgentBuilder import LLMAgentBuilder

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
        system_prompt=SystemPromptManager.get_agent_multimodal_system_prompt(),
        agent=True
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
agent_executor = llm_agent_builder.create_agent_multimodal_executor(verbose=True)

# 执行查询
query = "图片中展示的是什么，请给我搜索一些与其相关的最新的新闻信息"
image_url = "https://th.bing.com/th/id/R.65363320e5d8926fe1076d1890b85729?rik=mu77sO7cE0%2bBMg&riu=http%3a%2f%2fwww.baobeita.com%2fupload%2fimage%2fproduct%2f201412%2f10116104%2fc9e4becf-7ebb-4ff4-92c4-52be0efd428d-large.jpg&ehk=nCmDTpwGCetptQ4ZmKJIdfE1Ks46czlBx8r66tsdBM4%3d&risl=&pid=ImgRaw&r=0"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

tools_desc = llm_agent_builder.get_tools_description()
print(agent_executor.invoke({"tools": tools_desc, "question": query, "image_data": image_data})['output'])

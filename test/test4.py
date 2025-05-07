import os
import sys
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import get_llm_configs
from src.PromptManager import SystemPromptManager
from src.LLMAgent import LLMAgent
    
def calculator_tool(expression: str) -> str:
    """计算器工具"""
    try:
        return f"计算结果: {eval(expression)}"
    except:
        return "格式错误请检查!"

def search_tool(query: str) -> str:
    """搜索引擎工具"""
    try:
        search = TavilySearchResults(max_results=5)
        results = search.invoke(query)
        if not results:
            return "未找到相关信息"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"结果{i}:\n标题: {result.get('title', '无标题')}\n内容: {result.get('content', '无内容')}\n链接: {result.get('url', '无链接')}\n")
        
        return "搜索结果:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"搜索出错: {str(e)}"
    
# 初始化Agent
llm_agent = LLMAgent(
    get_llm_configs(
        system_prompt=SystemPromptManager.get_agent_system_prompt()
    )
)
    
# 添加工具
llm_agent.add_tools([
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
])

# 运行Agent
llm = llm_agent.get_agent_llm()
prompt = llm_agent.get_agent_prompt()
tools = llm_agent.get_agent_tools()
tools_desc = llm_agent.get_tools_description()

# 创建 Agent
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# 创建执行器
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"tools": tools_desc, "question": "江苏的省会是哪里，这个地方的现任市长是谁，多大年纪了"})
print(result)

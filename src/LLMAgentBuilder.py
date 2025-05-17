import os
import sys
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.tools import BaseTool, Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForToolRun, CallbackManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import LLMConfig
from src.LLMChainBuilder import LLMChainBuilder
from src.PromptBuilder import PromptBuilder


class LLMAgentBuilder:
    """
    LLM Agent类
    用于构建基于工具的智能代理
    """
    
    def __init__(self, config: LLMConfig):
        """初始化LLM Agent"""
        self.config = config 
        self.tools: List[BaseTool] = []
        self.llm_chain_builder = LLMChainBuilder()
        self.prompt_template_builder = PromptBuilder()
        self.agent_executor = None
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加单个工具到Agent"""
        self.tools.append(tool)
        
    def add_tools(self, tools: List[BaseTool]) -> None:
        """批量添加工具到Agent"""
        self.tools.extend(tools)
    
    def remove_tool(self, tool_name: str) -> bool:
        """根据名称移除工具"""
        initial_length = len(self.tools)
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        return len(self.tools) < initial_length
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_agent_llm(self) -> BaseLanguageModel:
        """获取Agent使用的语言模型"""
        return self.llm_chain_builder._init_chat_model(self.config)
    
    def get_agent_prompt(self) -> ChatPromptTemplate:
        """获取Agent使用的提示模板"""
        return self.prompt_template_builder.create_few_shot_chat_prompt_template(
            system_prompt=self.config.system_prompt, 
            examples=self.config.examples,
            is_agent=self.config.agent,
            is_memory=self.config.memory
        )
    
    def get_multimodal_agent_prompt(self) -> ChatPromptTemplate:
        """获取Agent使用的提示模板"""
        return self.prompt_template_builder.create_few_shot_multimodal_chat_prompt_template(
            system_prompt=self.config.system_prompt, 
            examples=self.config.examples,
            is_agent=self.config.agent,
            is_memory=self.config.memory
        )
    
    def get_agent_tools(self) -> List[BaseTool]:
        """获取Agent的所有工具"""
        return self.tools
    
    def get_tools_description(self) -> str:
        """获取所有工具的描述文本"""
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
    
    def create_agent_executor(self, verbose: bool = True, callback_manager: CallbackManager = None) -> AgentExecutor:
        """创建Agent执行器"""
        llm = self.get_agent_llm()
        prompt = self.get_agent_prompt()
        tools = self.get_agent_tools()
        
        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose, callback_manager=callback_manager)
        return self.agent_executor
    
    def create_multimodal_agent_executor(self, verbose: bool = True, callback_manager: CallbackManager = None) -> AgentExecutor:
        """创建多模态Agent执行器"""
        llm = self.get_agent_llm()
        prompt = self.get_multimodal_agent_prompt()
        tools = self.get_agent_tools()
        
        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose, callback_manager=callback_manager)
        return self.agent_executor


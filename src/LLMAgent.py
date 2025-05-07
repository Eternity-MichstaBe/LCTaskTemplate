import os
import sys
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, Tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.LLMConfig import LLMConfig
from src.LLMChainBuilder import LLMChainBuilder
from src.PromptBuilder import PromptBuilder


class LLMAgent:
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
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加工具到Agent"""
        self.tools.append(tool)
        
    def add_tools(self, tools: List[BaseTool]) -> None:
        """批量添加工具到Agent"""
        self.tools.extend(tools)
    
    def get_agent_llm(self):
        """创建Agent处理链"""
        return self.llm_chain_builder._init_chat_model(self.config)
    
    def get_agent_prompt(self):
        """创建Agent处理链"""
        return self.prompt_template_builder.create_chat_prompt_template(self.config.system_prompt, is_agent=True)
    
    def get_agent_tools(self) -> str:
        """获取所有工具的描述"""
        return self.tools
    
    def get_tools_description(self) -> List[BaseTool]:
        """获取所有工具的描述"""
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])


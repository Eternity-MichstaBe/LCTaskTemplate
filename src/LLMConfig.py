import configparser
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from langchain_core.output_parsers import BaseOutputParser


# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取配置文件的绝对路径
config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

# chatgpt接口
chatgpt_api_key = config['openai']['api_key']
chatgpt_api_base = config['openai']['api_base']

# deepseek接口
deepseek_api_key = config['deepseek']['api_key']
deepseek_api_base = config['deepseek']['api_base']


@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str
    temperature: float
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    system_prompt: str = ""
    examples: List[Tuple[str, str]] = None
    local: bool = False
    output_parser: BaseOutputParser = None


def get_llm_configs(**kwargs) -> LLMConfig:
    """获取LLM配置"""
    return LLMConfig(
        model_name=kwargs.get("model_name", "gpt-4o"),
        temperature=kwargs.get("temperature", 0.2),
        api_key=kwargs.get("api_key", chatgpt_api_key),
        api_base=kwargs.get("api_base", chatgpt_api_base),
        system_prompt=kwargs.get("system_prompt", ""),
        examples=kwargs.get("examples", []),
        local=kwargs.get("local", False),
        output_parser=kwargs.get("output_parser", None)
    )
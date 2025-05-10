import configparser
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from langchain_core.output_parsers import BaseOutputParser


class ConfigLoader:
    """配置加载器"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 获取配置文件的绝对路径
            config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.ini')
            
            cls._config = configparser.ConfigParser()
            cls._config.read(config_path)
        return cls._instance
    
    @classmethod
    def get_config(cls, section: str, key: str) -> str:
        """获取配置项"""
        if cls._config is None:
            cls()
        return cls._config.get(section, key, fallback="")


@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str
    temperature: float
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    system_prompt: str = ""
    examples: List[Dict[str, str]] = None
    local: bool = False
    output_parser: Optional[BaseOutputParser] = None
    agent: bool = False
    memory: bool = False


def get_llm_configs(**kwargs) -> LLMConfig:
    """获取LLM配置"""
    config_loader = ConfigLoader()
    
    return LLMConfig(
        model_name=kwargs.get("model_name", "gpt-4o"),
        temperature=kwargs.get("temperature", 0.2),
        api_key=kwargs.get("api_key", config_loader.get_config("openai", "api_key")),
        api_base=kwargs.get("api_base", config_loader.get_config("openai", "api_base")),
        system_prompt=kwargs.get("system_prompt", ""),
        examples=kwargs.get("examples", []),
        local=kwargs.get("local", False),
        output_parser=kwargs.get("output_parser", None),
        agent=kwargs.get("agent", False),
        memory=kwargs.get("memory", False)
    )
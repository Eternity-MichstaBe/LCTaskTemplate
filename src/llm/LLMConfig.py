import os
import sys
import configparser
from dataclasses import dataclass
from typing import Optional, List, Dict
from langchain_core.output_parsers import BaseOutputParser

# 添加项目根目录到路径
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)


@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str
    temperature: float
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    streaming: bool = False
    system_prompt: str = ""
    examples: List[Dict[str, str]] = None
    local: bool = False
    output_parser: Optional[BaseOutputParser] = None
    agent: bool = False
    memory: bool = False


def get_llm_configs(**kwargs) -> LLMConfig:
    """获取LLM配置"""

    return LLMConfig(
        model_name=kwargs.get("model_name", ""),
        temperature=kwargs.get("temperature", 0.2),
        api_key=kwargs.get("api_key", None),
        base_url=kwargs.get("base_url", None),
        streaming=kwargs.get("streaming", False),
        system_prompt=kwargs.get("system_prompt", ""),
        examples=kwargs.get("examples", []),
        local=kwargs.get("local", False),
        output_parser=kwargs.get("output_parser", None),
        agent=kwargs.get("agent", False),
        memory=kwargs.get("memory", False)
    )


cfg = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation()  # 支持 ${section:key} 插值
)
cfg.read(os.path.join(root, "config", "config.ini"), encoding="utf-8")

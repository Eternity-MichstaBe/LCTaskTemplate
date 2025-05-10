"""
输出格式
"""

import os
import sys
from pydantic import BaseModel, Field   

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.LLMChainBuilder import LLMChainBuilder
from src.LLMConfig import get_llm_configs

class standard(BaseModel):
    name: str = Field(description="电影名称")
    time: str = Field(description="电影上映时间")
    description: str = Field(description="电影简介")

llmconfig = get_llm_configs(
    model_name="gpt-4o",
    temperature=0.2,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
    output_parser=StrOutputParser()
)

llmconfig_ = get_llm_configs(
    model_name="gpt-4o",
    temperature=0.2,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答，请返回Json格式",
    output_parser=JsonOutputParser(pydantic_object=standard)
)

llm = LLMChainBuilder().create_chat_chain(llmconfig)
llm_ = LLMChainBuilder().create_chat_chain(llmconfig_)

# for chunk in llm.stream({"question": "周星驰都有哪些电影，请按时间顺序给我推荐几部"}):
#     print(chunk, end="", flush=True)

for chunk in llm_.stream({"question": "周星驰都有哪些电影，请按时间顺序给出10部"}):
    print(chunk, end="\n", flush=True)



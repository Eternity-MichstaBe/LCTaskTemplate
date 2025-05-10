"""
LLM测试
"""

import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from src.LLMChainBuilder import LLMChainBuilder
from src.LLMConfig import get_llm_configs
from example.examples import examples

llmconfig = get_llm_configs(
    model_name="gpt-4o",
    temperature=0.2,
    examples=examples,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。问题\n:{question}",
    output_parser=StrOutputParser()
)

llmconfig_ = get_llm_configs(
    model_name="gpt-4o",
    temperature=0.2,
    examples=examples,
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
    output_parser=StrOutputParser()
)

llm = LLMChainBuilder().create_few_shot_prompt_chain_with_selector(llmconfig, 2)
llm_ = LLMChainBuilder().create_few_shot_chat_chain_with_selector(llmconfig_, 2)

# 同步invoke
# print(llm.invoke({"question": "你如何看待AI的快速发展"}))
# print(llm_.invoke({"question": "你如何看待AI的快速发展"}))

# 同步stream
# for chunk in llm.stream({"question": "你如何看待AI的快速发展"}):
#     print(chunk, end="", flush=True)

# for chunk in llm.stream({"question": "你如何看待AI的快速发展"}):
#     print(chunk, end="", flush=True)

# 异步stream
async def astream_llm():
    async for chunk in llm.astream({"question": "你如何看待AI的快速发展"}):
        print(chunk, end="", flush=True)

async def astream_llm_():
    async for chunk in llm_.astream({"question": "你如何看待AI的快速发展"}):
        print(chunk, end="", flush=True)

async def main():
    # 顺序执行
    await astream_llm()
    print("\n")
    await astream_llm_()
    
    # 并行执行
    # await asyncio.gather(astream_llm(), astream_llm_())

asyncio.run(main())
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
    system_prompt="你是一个AI助手，请根据用户的问题给出回答。",
    output_parser=StrOutputParser()
)

llm_1 = LLMChainBuilder().create_prompt_chain(llmconfig)
llm_2 = LLMChainBuilder().create_few_shot_prompt_chain(llmconfig)
llm_3 = LLMChainBuilder().create_few_shot_prompt_chain_with_selector(llmconfig, 2)
llm_4 = LLMChainBuilder().create_chat_chain(llmconfig)
llm_5 = LLMChainBuilder().create_few_shot_chat_chain(llmconfig)
llm_6 = LLMChainBuilder().create_few_shot_chat_chain_with_selector(llmconfig, 2)

# print(llm_1.invoke({}))
# print(llm_2.invoke({}))
# print(llm_3.invoke({}))
# print(llm_4.invoke({"question": "唐太宗李世民是一个什么样的人"}))
# print(llm_5.invoke({"question": "唐太宗李世民是一个什么样的人"}))
# print(llm_6.invoke({"question": "唐太宗李世民是一个什么样的人"}))

# 同步流处理
chunks = []
for chunk in llm_4.stream({"type": "历史", "question": "唐太宗李世民是一个什么样的人"}):
    chunks.append(chunk)
    print(chunk, end="", flush=True)


# 异步流处理
async def llm_1_output():
    async for chunk in llm_1.astream({}):
        print(chunk, end="", flush=True)

async def llm_2_output():
    async for chunk in llm_2.astream({}):
        print(chunk, end="", flush=True)

async def llm_3_output():
    async for chunk in llm_3.astream({}):
        print(chunk, end="", flush=True)

async def llm_4_output():
    async for chunk in llm_4.astream({"type": "历史", "question": "唐太宗李世民是一个什么样的人"}):
        print(chunk, end="", flush=True)

async def llm_5_output():
    async for chunk in llm_5.astream({"type": "历史", "question": "唐太宗李世民是一个什么样的人"}):
        print(chunk, end="", flush=True)

async def llm_6_output():
    async for chunk in llm_5.astream({"type": "历史", "question": "唐太宗李世民是一个什么样的人"}):
        print(chunk, end="", flush=True)


async def main():
    await asyncio.gather(
        llm_1_output(),
        llm_2_output(),
        llm_3_output(),
        llm_4_output(),
        llm_5_output(),
        llm_6_output()
    )

asyncio.run(main())

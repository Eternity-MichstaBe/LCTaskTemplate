from langserve import RemoteRunnable
import asyncio

openai_llm = RemoteRunnable("http://localhost:8000/openai/")

async def ainvoke():
    async for chunk in openai_llm.astream({"question": "谁是中国最后一个皇帝"}):
        print(chunk, end="", flush=True)

asyncio.run(ainvoke())
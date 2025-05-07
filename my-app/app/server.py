from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.LLMConfig import get_llm_configs
from src.LLMChainBuilder import LLMChainBuilder

app = FastAPI(
    title="LangChain服务器",
    version="1.0",
    description="基于LangChain Runnable接口的API服务器",
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

prompt_chain = LLMChainBuilder().create_prompt_chain(
    get_llm_configs(
        system_prompt="你是知识渊博的历史学家，请回答以下问题:\n{question}"
    )
)

add_routes(
    app,
    prompt_chain,
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

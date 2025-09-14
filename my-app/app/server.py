from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm.LLMConfig import get_llm_configs
from src.llm.LLMChainBuilder import LLMChainBuilder


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
        system_prompt="你是一个专业的攻防决策专家，请回答以下问题:\n{question}",
        model_name="deepseek-reasoner",
        api_key="sk-51375b084e504ac49eb6cf892fe5b74f",
        api_base="https://api.deepseek.com"
    )
)

add_routes(
    app,
    prompt_chain,
    path="/attack",
)

add_routes(
    app,
    prompt_chain,
    path="/defense",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from langchain_core.runnables import RunnableLambda
chain = RunnableLambda(lambda x: {"output": f"Echo: {x}"})

add_routes(app, chain, path="/echo")

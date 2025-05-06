# from fastapi import FastAPI
# from pydantic import BaseModel

# class PingReq(BaseModel):
#     input: str

# PingReq.model_rebuild()

# app = FastAPI()

# @app.post("/ping")
# def ping(req: PingReq):
#     return {"msg": f"Echo: {req.input}"}

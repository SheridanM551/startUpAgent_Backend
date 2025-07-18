# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatReq(BaseModel):
    query: str

def simple_router(q: str) -> str:
    if any(k in q.lower() for k in ["百分比", "%", "營收", "成本"]):
        return "（這題屬於 SQL 工具，之後去查資料庫）"
    if any(k in q.lower() for k in ["案例", "失敗", "趨勢"]):
        return "（這題屬於 RAG 檢索，之後去 Pinecone）"
    return "目前不知道去哪，先用 LLM 回答：你好，這是預設回覆～"

@app.post("/chat")
async def chat(req: ChatReq):
    return {"answer": simple_router(req.query)}


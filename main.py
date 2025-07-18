# main.py ── 只做「直通 NIM」且沿用你成功的 payload 格式
import os, json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------
# 讀 .env
# ---------------------------------------------------
load_dotenv()
NIM_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NIM_API_KEY:
    raise RuntimeError("請在 .env 中設定 NVIDIA_API_KEY")

NIM_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "nvidia/llama3-chatqa-1.5-70b"

# ---------------------------------------------------
# FastAPI & schema
# ---------------------------------------------------
app = FastAPI()

class ChatReq(BaseModel):
    query: str
    stream: bool | None = False


# ---------------------------------------------------
# 呼叫 NIM（不傳多餘欄位）
# ---------------------------------------------------
def call_nim(prompt: str, stream: bool = False):
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": NIM_MODEL,
        "messages": [
            # 只傳 user message，與你成功的範例保持一致
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1024,
    }
    if stream:                       # 只有需要時才加入
        payload["stream"] = True

    return requests.post(
        NIM_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=stream,
        timeout=120,
    )

# ---------------------------------------------------
# 將 NIM stream 轉成 SSE
# ---------------------------------------------------
def sse_stream(nim_resp):
    try:
        for line in nim_resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                content = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                if content:
                    yield f"data: {content}\n\n"
    finally:
        nim_resp.close()

# ---------------------------------------------------
# /chat 端點
# ---------------------------------------------------
@app.post("/chat")
async def chat(req: ChatReq):
    try:
        resp = call_nim(req.query, stream=req.stream)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"NIM Error: {e}")

    if req.stream:
        return StreamingResponse(sse_stream(resp),
                                 media_type="text/event-stream")

    answer = resp.json()["choices"][0]["message"]["content"]
    return JSONResponse({"answer": answer})


@app.get("/health")
def health():
    return {"status": "ok", "model": NIM_MODEL}

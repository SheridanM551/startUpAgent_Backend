import os, json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain
from langchain import hub
from langchain.tools import Tool
from langchain.agents import (
    create_json_chat_agent,
    AgentExecutor,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ---------- Init ----------
load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    raise RuntimeError("NVIDIA_API_KEY 未設定")

# 小模型：負責規劃 (router + agent)
planner_llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key=API_KEY, 
  temperature=1,
  top_p=1,
  max_tokens=4096,
)

# 大模型：最終回答用
answer_llm = ChatNVIDIA(
  model="openai/gpt-oss-120b",
  api_key=API_KEY, 
  temperature=1,
  top_p=1,
  max_tokens=4096,
)

# ---------- tools ----------
def sql_query(sql: str) -> str:
    print(f"\n[SQL_TOOL] {sql}")
    if "medtech" in sql.lower():
        return "medical_startups_failure_rate = 0.42"
    return "overall_failure_rate = 0.29"    

def vector_search(query: str) -> str:
    print(f"\n[RAG_TOOL] {query}")
    return "【1】醫療器材監管門檻高...【2】COVID 後投資趨向保守..."

sql_tool = Tool(
    name="sql_db",
    func=sql_query,
    description=(
        "在結構化市場資料庫查詢統計數字，例如倒閉率、營收等。"
        "輸入 **必須** 是 SQL SELECT 陳述式。"
    ),
)
rag_tool = Tool(
    name="news_search",
    func=vector_search,
    description=(
        "在新聞/研究報告向量庫檢索相關段落。"
        "輸入為自然語言；輸出為多段帶【N】編號的摘要文本。"
    ),
)
tools = [sql_tool, rag_tool]
tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools])
tool_names = ", ".join([t.name for t in tools])
# ---------- Agent Prompt (Few-shot) ----------
system = """You are **StartupGPT**, an entrepreneurship analyst that helps users evaluate startup ideas, market conditions, and funding strategies.

## Your missio
For every user question, carefully decide whether external data is needed.  
You have two trusted tools:

{tool_desc}

## Workflow  — ReAct style
Iterate through the following steps **until** you are confident you can deliver a well-supported answer.

1. **Thought**: Reflect on what you need next (do NOT reveal chain-of-thought to the user).
2. **Action**: Choose **one** tool from `{tool_names}`.
3. **Action Input**: Provide the input for that tool.
4. **Observation**: Receive and read the tool’s output.

Repeat Thought → Action → Observation as many times as necessary.
""".format(
    tool_desc=tool_desc,
    tool_names=tool_names,
)

human = '''TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, \ The action to take. Must be one of {tool_names}
    "action_input": string \ The input to the action
}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \ You should put what you want to return to use here
}}
```

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{input}'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# ---------- 建 Agent ----------
agent = create_json_chat_agent(
    llm=planner_llm,
    tools=tools,
    prompt=prompt,            
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# ---------- FastAPI ----------
app = FastAPI()

PROD_ORIGINS = [
    "https://sheridanm551.github.io"
]

DEV_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS+PROD_ORIGINS,           # 建議白名單列出，不用 "*"
    # allow_origin_regex=ALLOW_ORIGIN_REGEX,  # 若要放寬至 *.github.io 再打開（擇一）
    allow_credentials=False,               # 若不需要 cookie / 認證建議關閉；若要開，origins 不可為 "*" 或太寬
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],# 若要讓前端讀自訂回應標頭，列在這邊
    max_age=86400,                         # 預檢快取（秒），減少 OPTIONS 次數
)

class ChatReq(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatReq):
    try:
        # 讓 Agent 自己規劃多工具
        result = agent_executor.invoke({"input": req.query})
        final = result.get("output", "").strip()

        # 若仍是空字串，保底用大模型直接回答
        if not final:
            final = answer_llm.invoke(req.query).content

        return JSONResponse({"answer": final})
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")

# ---------- 直接測 LLM ----------
class LLMReq(BaseModel):
    prompt: str
    model: str | None = None

@app.post("/test_llm")
async def test_llm(req: LLMReq):
    try:
        resp = answer_llm.invoke(req.prompt)
        return JSONResponse({"answer": resp.content})
    except Exception as e:
        raise HTTPException(500, f"NIM Error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}
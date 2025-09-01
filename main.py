import os, json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 手刻函數
from startUpAgent_Backend.news_rag.RAG_main import report_generator
# from startUpAgent_Backend.statistic_search.advisor_main import advice_generator
from startUpAgent_Backend.statistic_search.user_plot import user_plot_pipeline
from startUpAgent_Backend import config

# ---------- FastAPI ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.DEV_ORIGINS+config.PROD_ORIGINS,          
    # allow_origin_regex=ALLOW_ORIGIN_REGEX,  
    allow_credentials=False,               
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=86400,                         
)

# ----------------RAG---------------- #
@app.post("/rag")
async def rag(req: config.RAGreq):
    ''' contains two type of RAG: news and statistic'''
    try:
        req_dict = req.model_dump()
        desc, plot_ = user_plot_pipeline(req_dict)
        print("INFO:     Completed statistic generation")
        report = await report_generator(req_dict, desc)
        print("INFO:     Completed news generation")
        return JSONResponse({"plot": plot_, "report": report})
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")

# @app.post("/test_llm")
# async def test_llm(req: ChatReq):
#     try:
#         resp = answer_llm.invoke(req.query)
#         return JSONResponse({"answer": resp.content})
#     except Exception as e:
#         raise HTTPException(500, f"NIM Error: {e}")

@app.get("/health")
async def health():
    return {"status": config.SERVER_STATUS}
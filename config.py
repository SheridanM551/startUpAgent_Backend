from openai import BaseModel
from enum import Enum
import os
from dotenv import load_dotenv

# ---------- Init ----------
load_dotenv("startUpAgent_Backend/_.env")  # 載入環境變數檔案
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY 未設定")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY 未設定")

SERVER_STATUS = 0

PROD_ORIGINS = [
    "https://sheridanm551.github.io"
]

DEV_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://localhost:9487",
]

class ChatReq(BaseModel):
    query: str

class RAGreq(BaseModel):
    Industry_Group: str
    country: str
    current_employees: int
    total_funding: int
    founded: int
    current_objectives: str
    strengths: str
    weaknesses: str

class RagStatus(Enum):
    IDLE = 0
    RETRIEVING_STATISTIC_DATA = 1
    GENERATING_NEWS_QUERY = 2
    RETRIEVING_NEWS_DATA = 3
    GENERATING_NEWS_BULLETS = 4
    GENERATING_REPORT = 5
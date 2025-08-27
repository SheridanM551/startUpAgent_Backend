import json
import re
from typing import List, Tuple
from startUpAgent_Backend import config
import asyncio

# LangChain
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import AsyncOpenAI
from openai.types import ReasoningEffort
from langchain_core.documents import Document

# RAG
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

if not config.NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY 未設定")
if not config.PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY 未設定")
INDEX_NAME = "techcrunch-articles"  
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
LLM_MODEL = "openai/gpt-oss-20b" 
WRITER_LLM = "openai/gpt-oss-120b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

RAG_genQuestion_prompt = '''
You are a Retrieval Query Planner specialized in TechCrunch coverage.

GOAL
Given a startup profile JSON containing "answers", produce three tiers of natural-language search queries that can be embedded directly for semantic retrieval of TechCrunch articles.

OUTPUT
Return STRICT JSON only, with this schema:
{
    "precision": "string",
    "expanded": "string",
    "broad": "string"
}

CONSTRAINTS
- precision: one concise sentence focusing on exact signals (funding amount, round, industry, geography, etc.).
- expanded: a longer, more descriptive query including synonyms, related contexts, and company objectives.
- broad: a short, general query focusing only on industry + funding verbs.
- No commentary, no markdown, no trailing commas.
- If some fields are missing in the input, still return valid JSON with empty strings "".
- Do NOT include the word "TechCrunch" in the query (the index is already TechCrunch-only).

INPUT

'''

def make_writer_prompt(retrieved_text: str, user_profile_json: str) -> str:
    """
    Build a single writer prompt for a RAG 'writer' model.
    Inputs:
      - retrieved_text: concatenated or JSON-formatted contexts you retrieved (English only).
      - user_profile_json: the user's form answers plus normalized fields (English only).

    Returns:
      - A full instruction string for the writer model (no sys/user split).

    Note: best-practice 代表是一般性的「行業慣例 / 管理建議 / 經驗法則」
    """
    json_format = {
        "title": "string",
        "executive_summary": [
            "string",
            "string",
            "string"
        ],
        "market_news": [
            {"text": "string", "citation": [0,1]},
        ],
        "funding_signals": [
            {"text": "funding update", "citation": []}
        ],
        "competitive_landscape": [
            {"text": "competitive insight", "citation": []}
        ],
        "recommendations": {
            "no_regret_moves": [
            {"text": "string", "citation": []},
            ],
            "experiments": [
            {"text": "string", "citation": []},
            ],
            "longer_term_bets": [
            {"text": "string", "citation": []},
            ]
        },
        "risks_watchlist": [
            {"text": "string", "citation": []},
        ],
        "metrics_to_track": [
            {"KPI": "string", "Reason": "string", "citation": []},
        ]
    }
    return f"""
You are a startup/tech journalist. Produce a concise, executive-style English report using ONLY the evidence in <RETRIEVED_CONTEXTS>. Tailor insights to <USER_PROFILE>. 
If evidence is missing, write "insufficient evidence". Do not invent facts. 
No need to output source, just make sure you use the right citation.
Use numeric citations as integer arrays (e.g., "citations": [1,2]) wherever evidence-based, empty arrays for best-practice.

=== REQUIREMENTS ===
- Title
- Executive Summary (3–5 bullets)
- Market & News Pulse (4–6 bullets, each with [n] citation)
- Funding Signals (2–4 bullets with [n])
- Competitive Landscape (2–4 bullets with [n])
- Recommendations — What It Means for You
    - **No-regret moves (0–30 days):** 2–3 bullets
    - **Experiments (30–90 days):** 2–3 bullets
    - **Longer-term bets (90+ days):** 1–2 bullets
- Risks & Watchlist (2–4 bullets with [n])
- Metrics to Track (3–5 KPIs; note [n] vs. best-practice)

=== OUTPUT FORMAT (JSON) ===
{json.dumps(json_format, ensure_ascii=False)}

<RETRIEVED_CONTEXTS>
{retrieved_text}
</RETRIEVED_CONTEXTS>

<USER_PROFILE>
{user_profile_json}
</USER_PROFILE>
"""

pc = Pinecone(api_key=config.PINECONE_API_KEY)
embedder = NVIDIAEmbeddings(
    model=EMBED_MODEL,
    base_url=NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY
)
client = AsyncOpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY
)
vector_store = PineconeVectorStore(
    index=pc.Index(INDEX_NAME),        
    embedding=embedder,
    text_key="text"    
)

def _safe_json_loads(s: str) -> dict:
    """盡量把含雜訊/程式碼框的輸出轉為 JSON 物件。"""
    s = s.strip().strip('\n')
    try:
        return json.loads(s)
    except Exception:
        # 嘗試擷取最外層的大括號
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
        raise
def _normalize_money(text: str, keep_approx_symbol: bool = True) -> str:
    s = str(text)

    # 1) 把常見的「不斷行空白」統一成一般空白
    s = s.replace("\u202f", " ").replace("\u00A0", " ")  # 窄不斷行空白 / NBSP
    s = re.sub(r"\\u202f", " ", s)  # 處理字面上的 \u202f
    s = re.sub(r"\s+", " ", s)

    # 2) 將 LaTeX 遺留的 $~$ 視為「約 等於 + 美元」
    #    $~$771M -> ≈ $771M 或 $771M
    def _approx_repl(_):
        return "≈ $" if keep_approx_symbol else "$"
    s = re.sub(r"\$\s*~\s*\$", _approx_repl, s)

    # 3) 若有單獨的 LaTeX "~" 用作不換行空白，移除或轉空白
    #    在數字/單位之間的 ~ 替換成空白
    s = re.sub(r"~", " ", s)

    # 5) 規整貨幣與數字/單位的空白
    s = re.sub(r"\$\s+", "$", s)                      # $ 1.2 → $1.2
    s = re.sub(r"(\d)\s+([KMB])\b", r"\1\2", s)       # 1.2 M → 1.2M
    s = re.sub(r"(\$)\s+([0-9])", r"\1\2", s)         # $ 771 → $771
    s = re.sub(r"(≈)\s+\$", r"\1 $", s)               # ≈  $ → ≈ $

    # 7) 再次收束多餘空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

async def _call_api(prompt:str, model:str, reasoning_effort:ReasoningEffort) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=1,
            max_tokens=4096,
            reasoning_effort=reasoning_effort
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[news] Error calling {model} API:", e)

async def _get_news_from_hits(docs: List[Document]) -> Tuple[dict, List[str]]:
    s = set()
    for d in docs:
        s.add(d.metadata["url"])
    ret = []
    i = 0
    citations = dict() # <title:url>
    for url in s:
        chunks = await _search_vectors(query='_', k=20, filter={"url": url})
        title = f"[{i}] " + chunks[0].metadata.get("title", "No Title") + "\n"
        
        # sort by metadata::chunk_index
        chunks = sorted(chunks, key=lambda x: x.metadata.get("chunk_index", 0))
        txt = "".join([c.page_content for c in chunks])
        
        # remove noises from text
        txt = txt.replace("Email Facebook Twitter LinkedIn", "")

        ret.append(title + txt)
        citations[title] = url
        i += 1
    return citations, ret

async def _search_vectors(query: str, k: int, filter: dict=None) -> List[Document]:
    return await asyncio.to_thread(
        vector_store.similarity_search,
        query,
        k=k,
        filter=filter
    )

async def news_generator(answers:dict, debug=False) -> dict:
    print("INFO:     [News] Generating news...")
    config.SERVER_STATUS = config.RagStatus.GENERATING_NEWS_QUERY.value
    # step 1: gen question
    payload_json = str(answers)
    if debug:
        print(RAG_genQuestion_prompt + payload_json)
    resp = await _call_api(RAG_genQuestion_prompt + payload_json, model=LLM_MODEL, reasoning_effort="medium")
    # step 2: parse format
    resp = _safe_json_loads(resp)
    precision = (resp.get("precision") or "").strip()
    expanded  = (resp.get("expanded") or "").strip()
    broad     = (resp.get("broad") or "").strip()
    if debug:
        print("Precision:", precision)
        # print("Expanded:", expanded)
        # print("Broad:", broad)

    # step 3: retrieve（先用 expanded）
    config.SERVER_STATUS = config.RagStatus.RETRIEVING_NEWS_DATA.value
    hits = await _search_vectors(precision, k=10)
    citations, context = await _get_news_from_hits(hits)
    if debug: print(len(context), "hits from precision")

    # step 4: generate report
    writer_prompt = make_writer_prompt(
        "\n\n".join(context), payload_json
    )
    if debug: print(writer_prompt)
    config.SERVER_STATUS = config.RagStatus.GENERATING_NEWS_REPORT.value
    resp = await _call_api(writer_prompt, model=WRITER_LLM, reasoning_effort="low") # low to prevent overthinking
    resp = _safe_json_loads(_normalize_money(resp))
    return {"content": json.dumps(resp, ensure_ascii=False), "citations": json.dumps(citations, ensure_ascii=False)}

if __name__ == "__main__":
    sample_answers = {
        "Industry_Group": "Health Care",
        "country": "US",
        "current_employees": 80,
        "total_funding": 12000000,
        "founded": 2018,
        "current_objectives": "expand telemedicine services into rural regions; develop AI-driven diagnostics platform",
        "strengths": "strong partnerships with local hospitals; scalable cloud infrastructure",
        "weaknesses": "regulatory hurdles; limited brand recognition in urban markets"
    }
    result = asyncio.run(news_generator(sample_answers, debug=True))
    # display result (dict)
    print("=== News Generation Result ===")
    for key, value in result.items():
        print(f"{key}: {value}")
        print()

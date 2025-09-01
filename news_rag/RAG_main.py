import json
import re
from typing import List, Tuple, Dict, Any
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
LLM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1" 
# LLM_MODEL = "openai/gpt-oss-20b" 
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
news_json_format = {
    "news_bullets": [
        { "point": "string", "detail": "string", "citations": [0] },
        { "point": "string", "detail": "string", "citations": [1] },
        { "point": "string", "detail": "string", "citations": [2] },
        { "point": "string", "detail": "string", "citations": [3] },
        { "point": "string", "detail": "string", "citations": [4] }
    ]
}
def make_news_summary_prompt(retrieved_text: str, user_profile_json: str) -> str:
    return f'''
You are a news summarization model. Read two inputs:
1) NEWS_ITEMS (a list of news articles with titles and content).
2) USER_PROFILE (a JSON with user preferences).

Your task: produce one JSON object ONLY, following the schema below EXACTLY (same keys, types, and order). Do not include explanations, markdown, comments, or any extra fields. Do not echo statistics or raw metadata; write synthesized, plain-English narrative.

-- INPUTS --------------------------------------------------------
NEWS_ITEMS:
{retrieved_text}

USER_PROFILE:
{user_profile_json}
------------------------------------------------------------------

-- OUTPUT_FORMAT -------------------------------------------------
{json.dumps(news_json_format, ensure_ascii=False)}
------------------------------------------------------------------

-- WRITING_RULES -------------------------------------------------
News bullets:
- Create 5 bullets that a founder must know now.
- Each bullet: 
  - “point”: a crisp headline (≤12 words).
  - “detail”: 1–3 sentences explaining relevance/implication.
  - “citations”: array of 1–2 integers referencing NEWS_ITEMS indices (0-based). Choose the most relevant sources only.

'''

json_format_v2 = {
    "cover": {
        "adjectives": ["string","string","string"],
        "one_liner": "string",
        "three_takeaways": [
        { "text": "string", "direction": "lead" },
        { "text": "string", "direction": "lag" },
        { "text": "string", "direction": "par" }
        ]
    },
    "charts": {
        "current_employees": {
        "caption": "string",
        "callouts": ["string","string"]
        },
        "total_funding": {
        "caption": "string",
        "callouts": ["string","string"]
        },
        "industry_distribution": {
        "caption": "string",
        "callouts": ["string","string"]
        }
    },
    "peers": {
        "overview": "string",
        "peer_snapshots": [
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" },
        { "company": "string", "blurb": "string" }
        ]
    },
    "recommendations": [
        { "keyword": "string", "description": "string" },
        { "keyword": "string", "description": "string" },
        { "keyword": "string", "description": "string" }
    ]
}

def make_writer_prompt_v2(stats:str, news_bullets: dict, user_profile_json: str):
    
    return f'''You are an analyst. Read two inputs:
1) GROWJO_TEXT (contains three tagged sections):
   <top 10 company description> ... </top 10 company description>
   <compare to top 10> ... </compare to top 10>
   <all industry info> ... </all industry info>

2) NEWS_BULLETS.

3) USER_PROFILE (a JSON with company details).

Your task: produce one JSON object ONLY, following the schema below EXACTLY (same keys, types, and order). Do not include explanations, markdown, comments, or any extra fields. Do not echo statistics or raw metadata; write synthesized, plain-English narrative.

-- INPUTS --------------------------------------------------------
GROWJO_TEXT:
{stats}

NEWS_BULLETS (0-indexed for citation indices):
{json.dumps(news_bullets, ensure_ascii=False)}

USER_PROFILE:
{user_profile_json}
------------------------------------------------------------------

-- OUTPUT_FORMAT -------------------------------------------------
{json.dumps(json_format_v2, ensure_ascii=False)}
------------------------------------------------------------------

-- WRITING RULES -------------------------------------------------
General:
- Output must be valid JSON and match the schema exactly (no extra keys).
- Use concise, executive tone. No numbers copied verbatim from stats; synthesize insights in words (e.g., “upper-quartile hiring scale” rather than raw values).
- If a field can’t be determined, infer conservatively from context; do NOT invent facts not suggested by inputs.
- Keep company names exactly as in GROWJO_TEXT when available.

Cover:
- adjectives: 3 descriptors (each 1–3 words).
- one_liner: 1 sentence positioning statement.
- three_takeaways: 3 bullets; set “direction” to one of {"lead","lag","par"}.

Charts (derived from <compare to top 10> and <all industry info>):
- caption: 1–2 sentences explaining what the chart shows for the user vs peers/industry.
- callouts: 1–2 short phrases highlighting the most important contrasts or context.

Peers:
- overview: 3–5 sentences summarizing what fast-rising peers in the same industry are doing (from <top 10 company description> and any peer context implied by compare/industry sections).
- peer_snapshots: exactly 10 items. For each, “company” is the company name; “blurb” is 1–2 sentences focusing on what they do, product/market angle, and any notable differentiator. Do not include numeric stats.

Recommendations:
- Provide 3 No-Regret moves.
- Each item:
  - “keyword”: 1–3 words (e.g., “Pricing test”, “Compliance playbook”, “Channel focus”).
  - “description”: 1–2 sentences with specific, actionable guidance tied to observed gaps/opportunities.

Do not include explanations, markdown, comments, or any extra fields
'''

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
        if model == "openai/gpt-oss-20b" or model == "openai/gpt-oss-120b":
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=1,
                max_tokens=4096,
                reasoning_effort=reasoning_effort
            )
            return resp.choices[0].message.content
        elif model == "nvidia/llama-3.3-nemotron-super-49b-v1":
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":"detailed thinking on"},
                          {"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return resp.choices[0].message.content
    except Exception as e:
        raise Exception(f"[news] Error calling {model} API:", e)

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

async def report_generator(answers:dict, stats_description: str, debug=False) -> dict:
    print("INFO:     [News] Generating querys...")
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
        print("\n>>> Precision:", precision)
        # print("Expanded:", expanded)
        # print("Broad:", broad)

    # step 3: retrieve
    config.SERVER_STATUS = config.RagStatus.RETRIEVING_NEWS_DATA.value
    print("INFO:     [News] Retrieving news data...")
    hits = await _search_vectors(precision, k=10)
    citations, context = await _get_news_from_hits(hits)
    if debug: print(len(context), "hits from precision")

    # step 4: generate news bullets
    config.SERVER_STATUS = config.RagStatus.GENERATING_NEWS_BULLETS.value
    print("INFO:     [News] Generating news bullets...")
    news_bullets = await _call_api(make_news_summary_prompt("\n\n".join(context), payload_json), model=WRITER_LLM, reasoning_effort="low")
    news_bullets = _safe_json_loads(news_bullets)

    if debug:
        print(json.dumps(news_bullets, ensure_ascii=False))

    # step 5: generate overall report
    print("INFO:     [News] Generating report...")
    writer_prompt = make_writer_prompt_v2(stats_description, news_bullets, payload_json)

    if debug: print(writer_prompt)
    config.SERVER_STATUS = config.RagStatus.GENERATING_REPORT.value
    resp = await _call_api(writer_prompt, model=WRITER_LLM, reasoning_effort="low") # low to prevent overthinking
    if debug:
        print("Writer response:", resp)

    resp = _safe_json_loads(_normalize_money(resp))
    resp["news_bullets"] = news_bullets.get("news_bullets", [])
    return {"content": resp, "citations": citations}

# async def news_retriever(answers:dict, debug=False) -> Tuple[dict, List[str]]:
#     print("INFO:     [News] Retrieving news...")
#     config.SERVER_STATUS = config.RagStatus.GENERATING_NEWS_QUERY.value
#     # step 1: gen question
#     payload_json = str(answers)
#     if debug:
#         print(RAG_genQuestion_prompt + payload_json)
#     resp = await _call_api(RAG_genQuestion_prompt + payload_json, model=LLM_MODEL, reasoning_effort="medium")
#     # step 2: parse format
#     resp = _safe_json_loads(resp)
#     precision = (resp.get("precision") or "").strip()
#     expanded  = (resp.get("expanded") or "").strip()
#     broad     = (resp.get("broad") or "").strip()
#     if debug:
#         print("Precision:", precision)
#         # print("Expanded:", expanded)
#         # print("Broad:", broad)
#     config.SERVER_STATUS = config.RagStatus.RETRIEVING_NEWS_DATA.value
#     hits = await _search_vectors(precision, k=10)
#     return await _get_news_from_hits(hits)

if __name__ == "__main__":
    sample_answers = {
        "Industry_Group": "Artificial Intelligence",
        "country": "US",
        "current_employees": 54,
        "total_funding": 25000000,
        "founded": 2014,
        "current_objectives": "hit $1.5M ARR in 12 months; land first enterprise logo",
        "strengths": "excellent multilingual support; lightweight API",
        "weaknesses": "limited admin features; no SOC2; weak sales motion",
        "desc": "The company develops an AI-powered customer support automation platform that specializes in multilingual markets across Asia-Pacific. Its lightweight API integrates seamlessly into existing CRMs and messaging apps, allowing SMEs to quickly deploy chatbots and virtual assistants in over 12 languages. Having raised $25M across Seed, Series A, and a recent Series B round, the company is now under pressure from investors to prove enterprise adoption. While it has seen strong traction with SMBs and mid-market clients in Taiwan and Southeast Asia, it has struggled to close larger enterprise contracts due to limited admin dashboards, lack of SOC2 compliance, and an inexperienced outbound sales team. The next 12 months are critical as the company plans to professionalize its go-to-market motion, hire enterprise sales talent, and pursue certifications to win the trust of Fortune 500 and regional conglomerates."
    }
    stat = '''<top 10 company description>
{"company_name":"Formic Technologies","Industry":"AI","Industry_Group":"Artificial Intelligence","country":null,"ranking":651,"founded":2011.0,"current_employees":89,"total_funding":59100000.0,"valuation":null,"description":"We deliver the automation system to do the job right, parts and service to keep it running the way you need it to, and the technology and software to capture and analyze data for continuous improvement on your line - all for one fixed monthly rate. No CapEx. No Hiring. No Problems. Get A Quote See Pricing\nIn short, this company was founded in 2011. It has grown by 43% and currently has 89 employees. This company has raised $59,100,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Story","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":14,"founded":2011.0,"current_employees":119,"total_funding":193000000.0,"valuation":2250000000.0,"description":"Story is the infrastructure layer that makes IP and real-world data programmable, enforceable, and monetizable. Story tokenizes IP and makes it programmable, from ownership to remix to monetization. Start Building Explore\nIn short, this company was founded in 2011 in United States. It has grown by 69% and currently has 119 employees. This company has raised $193,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"krea.ai","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":130,"founded":2011.0,"current_employees":45,"total_funding":83000000.0,"valuation":500000000.0,"description":"Krea makes generative AI intuitive. Generate, edit, and enhance images and videos using powerful AI for free.\nIn short, this company was founded in 2011 in United States. It has grown by 125% and currently has 45 employees. This company has raised $83,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Mechanical Orchard","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":100,"founded":2022.0,"current_employees":122,"total_funding":86200000.0,"valuation":95000000.0,"description":"Mechanical Orchard is an AI-native technology company that de-risks the process of bringing old but critical computer systems up to date. We modernize and run crucial business applications used by some of the largest companies around the world. Our goal? Help our customers stay ahead of the curve, compete at an ever-accelerating pace, and win in their markets.\nIn short, this company was founded in 2022 in United States. It has grown by 35% and currently has 122 employees. This company has raised $86,200,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Midjourney","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":505,"founded":2011.0,"current_employees":171,"total_funding":null,"valuation":null,"description":"Midjourney is an independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species. We are a small self-funded team focused on design, human infrastructure, and AI. We have 11 full-time staff and an incredible set of advisors.\nIn short, this company was founded in 2011 in United States. It has grown by 18% and currently has 171 employees. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Crescendo","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":72,"founded":2011.0,"current_employees":73,"total_funding":50000000.0,"valuation":500000000.0,"description":"Crescendo provides the world's first all-included service of AI technology and CX experts that increases customer engagement while reducing costs. AI-powered customer support for CX leaders that automates, resolves, and scales - backed by human expertise for seamless interactions across all channels. Crescendo provides CX leaders with the best AI support platform. Understand, respond, and resolve with intelligent automation, always supported by real people. Crescendo is the only omnichannel AI platform built for customer service leaders that auto scales, gathers insights, and delights customers - backed by real people. See pricing Watch demo\nIn short, this company was founded in 2011 in United States. It has grown by 175% and currently has 73 employees. This company has raised $50,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Speak","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":66,"founded":2011.0,"current_employees":240,"total_funding":161000000.0,"valuation":1000000000.0,"description":"Talk out loud, get instant feedback, and become fluent with the world\u00e2\u0080\u0099s most advanced AI language tutor. Start \u00c2 Speaking \u00e2\u0086\u0092\nIn short, this company was founded in 2011 in United States. It has grown by 46% and currently has 240 employees. This company has raised $161,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Luma AI","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":31,"founded":2021.0,"current_employees":166,"total_funding":114000000.0,"valuation":300000000.0,"description":"Luma's mission is to build multimodal general intelligence that can generate, understand, and operate in the physical world\nIn short, this company was founded in 2021 in United States. It has grown by 202% and currently has 166 employees. This company has raised $114,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Music.AI","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":504,"founded":2011.0,"current_employees":261,"total_funding":11900000.0,"valuation":null,"description":"Built for scale and powered by state-of-the-art, ethical AI solutions for audio and music applications\u2014delivering the highest-quality audio separation available. Contact Sales\nIn short, this company was founded in 2011 in United States. It has grown by 63% and currently has 261 employees. This company has raised $11,900,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}
{"company_name":"Together AI","Industry":"AI","Industry_Group":"Artificial Intelligence","country":"United States","ranking":10,"founded":2022.0,"current_employees":217,"total_funding":537000000.0,"valuation":3300000000.0,"description":"Run and fine-tune generative AI models with simple APIs and scalable GPU clusters. Train & deploy at scale on The AI Acceleration Cloud.\nIn short, this company was founded in 2022 in United States. It has grown by 166% and currently has 217 employees. This company has raised $537,000,000 so far. It also focuses on the Artificial Intelligence industry, specializing in AI."}

</top 10 company description>

<compare to top 10>
For **current_employees** , n=10. The IQR spans [96.5, 206] with median 144, minimum vlaue 45 and , maximum vlaue 261. User's input value is 54, around the 10.0th percentile. This lies within the whisker range.
For **total_funding** , n=9. The IQR spans [5.91e+07, 1.61e+08] with median 8.62e+07, minimum vlaue 1.19e+07 and , maximum vlaue 5.37e+08. User's input value is 2.5e+07, around the 11.1th percentile. This lies within the whisker range.
</compare to top 10>

<all industry info>
For **Industry_Group** , n=1000. The most common category is 'Software' (149). User's category 'Artificial Intelligence' has rank 3 and accounts for 9.70% of entries.
</all industry info>'''
    result = asyncio.run(report_generator(sample_answers, stat, debug=True))
    print("===== REPORT =====")
    print(json.dumps(result, indent=4))
    with open("./report.json", "w") as f:
        json.dump(result, f, indent=4)

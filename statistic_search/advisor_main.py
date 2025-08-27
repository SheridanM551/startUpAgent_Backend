import json
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

import os
from .advisor_assistant import StrategyAdvisor, DEFAULT_SYSTEM_PROMPT_V2
from typing import Dict
from .topk_company import *
from .. import config

async def advice_generator(user_input:dict, greedy_record=1, debug=0)-> Dict:
    """
    A pipeline integrate
    - Find top 10 similar companies
    - Generate differences, analysis, suggestions according to information of 10 similar companies    
    """
    print("INFO:     [Advisor] Generating advice...")
    config.SERVER_STATUS = config.RagStatus.RETRIEVING_STATISTIC_DATA.value
    top10 = recommend_topk_for_input(
        user_input=user_input, k=10,
        # constraints={"Industry": "Artificial Intelligence", "country": "United States"},
        use_cols=None,  # or specify: ["Industry","country","current_employees","employee_growth","total_funding","founded"]
        show_cols=display_cols
    )

    def ordered(row):
        return f"- Company {row['rank_in_list']}\n" + row["description"]

    top10["description"] = top10.apply(ordered, axis=1)

    if debug:
        print(top10)

    top10_descriptions = top10["description"].to_list()

    advisor = StrategyAdvisor(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=config.NVIDIA_API_KEY,
        temperature=0.4,
        top_p=0.9,
        max_tokens=4000,
        system_prompt=DEFAULT_SYSTEM_PROMPT_V2,
        use_think_tag=True
    )

    # ---- First pass (already asks for advanced_suggestions in the same JSON)
    config.SERVER_STATUS = config.RagStatus.GENERATING_STATISTIC_REPORT.value
    result = await advisor.analyze(top10_descriptions, user_input)

    ok, missing = advisor.minimal_schema_check(result)
    if not ok:
        print("\n⚠️ Missing fields:", missing)

    # advisor.pretty_print_top_actions(result, k=5)

    # ---- Second pass: enrich advanced_suggestions directly from open_questions
    if not result.get("advanced_suggestions"):
        advanced = advisor.expand_advanced_suggestions(
            companies_ctx=top10_descriptions,
            my_company=user_input,
            open_questions=result.get("open_questions", []),
        )
        result["advanced_suggestions"] = advanced
    if greedy_record:
        result["top10_descriptions"] = top10_descriptions
    return result


test_inputs = [

    {
        "Industry_Group": "Health Care",
        "country": "United States",
        "current_employees": 213,
        "total_funding": 75_000_000,
        "founded": 2010,
        "current_objectives": "expand telemedicine platform",
        "strengths": "HIPAA compliance",
        "weaknesses": "slow product iteration"
    },

    {
        "Industry_Group": "Gaming",
        "country": "Japan",
        "current_employees": 38,
        "total_funding": 8_000_000,
        "founded": 2018,
        "current_objectives": ["launch cross-platform title", "grow community to 1M users"],
        "strengths": ["strong art style", "VR experience"],
        "weaknesses": ["limited marketing budget", "small dev team"]
    },

    {
        "Industry_Group": "Energy",
        "country": "Germany",
        "current_employees": 413,
        "total_funding": 150_000_000,
        "founded": 2005,
        "current_objectives": ["expand solar farm network", "secure long-term storage solutions"],
        "strengths": ["government subsidies", "green brand reputation"],
        "weaknesses": ["grid integration challenges", "supply chain bottlenecks"]
    },

    {
        "Industry_Group": "Financial",
        "country": "Singapore",
        "current_employees": 96,
        "total_funding": 40_000_000,
        "founded": 2016,
        "current_objectives": ["obtain digital bank license", "increase SME lending"],
        "strengths": ["fast KYC onboarding", "mobile-first design"],
        "weaknesses": ["limited compliance team", "customer churn in retail banking"]
    },

    {
        "Industry_Group": "Biotechnology",
        "country": "United Kingdom",
        "current_employees": 61,
        "total_funding": 5_000_000,
        "founded": 2012,
        "current_objectives": ["scale CRISPR research", "form pharma partnerships"],
        "strengths": ["strong IP portfolio", "experienced research team"],
        "weaknesses": ["long R&D cycles", "capital-intensive experiments"]
    },

    {
        "Industry_Group": "Manufacturing",
        "country": "Mexico",
        "current_employees": 509,
        "total_funding": 10_000_000,
        "founded": 1998,
        "current_objectives": ["automate production line", "reduce defect rate by 15%"],
        "strengths": ["low-cost labor", "proximity to US market"],
        "weaknesses": ["outdated ERP system", "high energy consumption"]
    },

    {
        "Industry_Group": "Sustainability",
        "country": "Denmark",
        "current_employees": 23,
        "total_funding": 5_000_000,
        "founded": 2021,
        "current_objectives": ["launch carbon credit platform", "secure seed round extension"],
        "strengths": ["young and agile team", "strong ESG narrative"],
        "weaknesses": ["limited customer traction", "regulatory uncertainty"]
    },

    {
        "Industry_Group": "Travel",
        "country": "India",
        "current_employees": 371,
        "total_funding": 60_000_000,
        "founded": 2013,
        "current_objectives": ["expand into tier-2 cities", "integrate AI recommendations"],
        "strengths": ["large local user base", "mobile-first adoption"],
        "weaknesses": ["low margins", "seasonal demand swings"]
    },

    {
        "Industry_Group": "Software",
        "country": "Canada",
        "current_employees": 75,
        "total_funding": 20_000_000,
        "founded": 2017,
        "current_objectives": ["migrate to SaaS pricing", "expand to EU market"],
        "strengths": ["robust devops culture", "strong UI/UX"],
        "weaknesses": ["weak enterprise sales", "limited integrations"]
    },

    {
        "Industry_Group": "Consumer Goods",
        "country": "Brazil",
        "current_employees": 128,
        "total_funding": 15_000_000,
        "founded": 2011,
        "current_objectives": ["expand e-commerce channel", "improve supply chain efficiency"],
        "strengths": ["brand recognition", "loyal customer base"],
        "weaknesses": ["high logistics cost", "limited digital transformation"]
    },

]

if __name__ == "__main__":
    import asyncio
    i = 0
    result = asyncio.run(advice_generator(test_inputs[i], greedy_record=1))
    os.makedirs("./advisor_output", exist_ok=True)
    with open(f"./advisor_output/test_inputs_{datetime.now().strftime('%m%d%H')}_{i}.json", "w") as f:
        json.dump(result, f, indent=2)
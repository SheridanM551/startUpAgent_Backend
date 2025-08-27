# advisor_assistant.py
from __future__ import annotations
import os, json, re, sys, time
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI


DEFAULT_SYSTEM_PROMPT = """
You are a pragmatic startup/strategy advisor. Your job:
1) Compare the provided top 10 similar companies to "my_company".
2) Identify what my_company should do next to become more competitive.

Constraints:
- Replace the term "my_company" with "your company" when giving advice.
- Be concrete and actionable. Avoid vague advice.
- Prioritize by impact and effort. Use a simple score to justify priorities.
- If data is missing, make lightweight assumptions and call them out.
- Output EXACTLY one JSON object conforming to the schema below. No extra prose.

JSON schema to return:
{
  "quick_diagnosis": "2-3 sentence summary of where my_company stands vs peers",
  "key_differences": [
    {"dimension": "e.g., pricing / go-to-market / product depth / operations", "my_company": "...", "top_peer": "CompanyName", "so_what": "..."}
  ],
  "prioritized_actions": [
    {"action": "...", "why": "...", "effort_1_to_5": 3, "impact_1_to_5": 5, "priority_score": 15}
  ],
  "30_60_90_plan": {
    "day_30": ["..."],
    "day_60": ["..."],
    "day_90": ["..."]
  ],
  "risks_and_mitigations": [
    {"risk": "...", "mitigation": "..."}
  ],
  "metrics_to_track": ["north-star metric", "input metric 1", "input metric 2"],
  "open_questions": ["missing data you’d ask for next"]
}
- After you generate open_questions, also derive "advanced_suggestions" by iterating over them.
  For each open question, add an item to "advanced_suggestions" with:
  {
    "open_question": "<copy one item from open_questions>",
    "why_it_matters": "...",
    "how_to_answer_fast": ["data to pull", "experiments", "customer calls", "queries"],
    "low_cost_proxy": "quick signal while real data is pending",
    "decision_rule": "IF <condition> THEN <action>; ELSE <action>",
    "what_if_branch": {
      "answer_yes": {"action": "...", "expected_impact_1_to_5": 5},
      "answer_no": {"action": "...", "expected_impact_1_to_5": 3}
    },
    "ice_score": {"impact": 1-5, "confidence": 1-5, "ease": 1-5, "total": impact*confidence*ease}
  }
- Keep items concise and immediately actionable.
"""

DEFAULT_SYSTEM_PROMPT_V2 = """
You are a pragmatic startup/strategy advisor. Your job:
1) Compare the provided top 10 similar companies to "my_company".
2) Identify what "your company" should do next to become more competitive.

Constraints:
- Replace the term "my_company" with "your company" when giving advice.
- Be concrete and actionable. Avoid vague advice.
- Prioritize by impact and effort. Use a simple score to justify priorities.
- If data is missing, make lightweight assumptions and call them out.
- Output EXACTLY one JSON object conforming to the schema below. No extra prose.

JSON schema to return:
{
  "quick_diagnosis": "2-3 sentence summary of where your company stands vs peers",
  "key_differences": [
    {"dimension": "pricing / go-to-market / product depth / operations", "your_company": "...", "top_peer": "CompanyName", "so_what": "..."}
  ],
  "prioritized_actions": [
    {"action": "...", "why": "...", "effort_1_to_5": 3, "impact_1_to_5": 5, "priority_score": 15}
  ],
  "30_60_90_plan": {
    "day_30": ["..."],
    "day_60": ["..."],
    "day_90": ["..."]
  ],
  "risks_and_mitigations": [
    {"risk": "...", "mitigation": "..."}
  ],
  "metrics_to_track": ["north-star metric", "input metric 1", "input metric 2"],
  "open_questions": ["strategic uncertainties that still need clarification"],
  "advanced_suggestions": [
    {
      "advanced_question": "Formulate as a decision or hypothesis, e.g. 'Should your company prioritize enterprise clients before achieving SOC2 compliance?'",
      "why_it_matters": "Explain why this decision is critical and what’s at stake.",
      "how_to_answer_fast": ["low-cost tests, quick customer experiments, or proxy metrics"],
      "low_cost_proxy": "A cheap or fast signal that approximates the real answer (e.g., landing page signups, pilot test, survey).",
      "decision_rule": "IF <condition> THEN <action>; ELSE <alternative action>.",
      "scenario_planning": {
        "answer_yes": {"action": "...", "expected_impact_1_to_5": 5},
        "answer_no": {"action": "...", "expected_impact_1_to_5": 3}
      },
      "priority": "Urgent / Medium-term / Long-term"
    }
  ]
}

Instructions for advanced_suggestions:
- Do NOT just restate missing data. Instead, turn open_questions into **concrete strategic choices** or **testable hypotheses**.
- Each suggestion must include a **decision rule** or **scenario plan** so that your company knows what to do in either case.
- Always include a **low-cost proxy** for quick validation before committing resources.
- Ensure suggestions are actionable within a 30/60/90 day framework.
"""


FOLLOWUP_SYSTEM_PROMPT = """
You are a startup research planner. Create advanced, decision-oriented suggestions based on open questions.
Return ONLY a JSON array named advanced_suggestions (no wrapper text, no prose).
Each item must follow:
{
  "open_question": "...",
  "why_it_matters": "...",
  "how_to_answer_fast": ["...","..."],
  "low_cost_proxy": "...",
  "decision_rule": "IF ... THEN ...; ELSE ...",
  "what_if_branch": {
     "answer_yes": {"action": "...", "expected_impact_1_to_5": 5},
     "answer_no": {"action": "...", "expected_impact_1_to_5": 3}
  },
  "ice_score": {"impact": 1-5, "confidence": 1-5, "ease": 1-5, "total": impact*confidence*ease}
}
"""


class StrategyAdvisor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        temperature: float = 0.4,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        use_think_tag: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        stream=False # Output thoughts of LLM or not
    ):
        """
        api_key: read from env `NVIDIA_API_KEY` if None
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set NVIDIA_API_KEY env var or pass api_key.")
        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.use_think_tag = use_think_tag
        self.system_prompt = system_prompt
        self.stream = stream

    @staticmethod
    def _make_user_prompt(top10_descriptions: List[str], my_company: Dict[str, Any]) -> str:
        return (
            "You are given market data:\n\n"
            f"top10_similar_companies = {json.dumps(top10_descriptions, ensure_ascii=False)}\n"
            f"my_company = {json.dumps(my_company, ensure_ascii=False)}\n\n"
            "Task:\n"
            "- Compare my_company to the peer set.\n"
            "- Recommend the top 5 next actions with effort/impact scoring and a 30/60/90 plan.\n"
            "- Return ONLY the JSON object per the schema from the system message.\n"
        )

    def _request_stream(self, messages: List[Dict[str, str]]) -> str:
        """Stream response and return the raw concatenated text."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )
        buf = []
        for chunk in completion:
            delta = chunk.choices[0].delta.content
            if delta:
                sys.stdout.write(delta)  # live preview
                sys.stdout.flush()
                buf.append(delta)
        return "".join(buf)
    
    async def _request(self, messages: List[Dict[str, str]]) -> str:
        """Get response text, handling both streaming and non-streaming."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )

        # Streaming path
        if self.stream:
            buf = []
            for chunk in resp:  # resp is an iterator
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    buf.append(delta)
            return "".join(buf)

        # Non-streaming path: resp is a single object
        # Some SDKs use .message.content; older ones may use .text
        choice = resp.choices[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if content is None:
            content = getattr(choice, "text", None)  # fallback if provider returns .text
        return content or ""


    @staticmethod
    def _extract_json_block(raw: str) -> Optional[str]:
        """Find the first {...} block (tolerant of code fences)."""
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return m.group(0) if m else None

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _ensure_advanced_suggestions(payload: dict) -> dict:
        # Ensure top-level "advanced_suggestions" exists, even if empty
        if "advanced_suggestions" not in payload:
            payload["advanced_suggestions"] = []
        return payload

    async def analyze(
        self,
        top10_descriptions: List[str],
        my_company: Dict[str, Any],
        inject_think_tag: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """One-pass analysis that should already include advanced_suggestions per the system prompt."""
        user_prompt = self._make_user_prompt(top10_descriptions, my_company)
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if (self.use_think_tag if inject_think_tag is None else inject_think_tag):
            messages.append({"role": "system", "content": "/think"})
        messages.append({"role": "user", "content": user_prompt})

        # raw = self._request_stream(messages)
        raw = await self._request(messages)
        json_block = self._extract_json_block(raw)
        if not json_block:
            raise ValueError("No JSON object found in model output. Check the raw text above.")

        parsed = self._try_parse_json(json_block)
        if not parsed:
            raise ValueError("Failed to parse JSON. Inspect the raw JSON block for minor syntax issues.")

        return self._ensure_advanced_suggestions(parsed)

    def expand_advanced_suggestions(
        self,
        companies_ctx: List[str],
        my_company: Dict[str, Any],
        open_questions: List[str],
    ) -> List[Dict[str, Any]]:
        """Optional 2nd pass to enrich advanced_suggestions from open_questions."""
        followup_user = (
            "Context (summaries may reference these):\n"
            f"companies = {json.dumps(companies_ctx, ensure_ascii=False)}\n"
            f"my_company = {json.dumps(my_company, ensure_ascii=False)}\n\n"
            "Open questions to expand:\n"
            f"{json.dumps(open_questions, ensure_ascii=False)}\n\n"
            "Produce the JSON array 'advanced_suggestions' only."
        )

        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": followup_user},
        ]
        text = self._request_stream(messages)
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def pretty_print_top_actions(payload: Dict[str, Any], k: int = 5) -> None:
        actions = payload.get("prioritized_actions", [])[:k]
        print("\n\nTop actions:")
        for a in actions:
            try:
                print(f"- {a['action']}  (impact {a['impact_1_to_5']}, effort {a['effort_1_to_5']}, score {a['priority_score']})")
            except KeyError:
                print(f"- {a}")

    @staticmethod
    def minimal_schema_check(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Light schema guard; returns (ok, missing_fields)."""
        required_keys = [
            "quick_diagnosis",
            "key_differences",
            "prioritized_actions",
            "30_60_90_plan",
            "risks_and_mitigations",
            "metrics_to_track",
            "open_questions",
        ]
        missing = [k for k in required_keys if k not in payload]
        return (len(missing) == 0, missing)

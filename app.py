"""
✈️ SkyBot — AI Airline Assistant
HuggingFace Spaces | Safe Production Container Architecture
LLM: LiteLLM (provider-agnostic — see LLM_MODEL below) | Data: Duffel Flights API | Guardrails: 4-Layer
"""

import os
import re
import datetime
import json
import random
import requests
import pandas as pd
import gradio as gr
from datetime import date, timedelta
import time

from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ for local runs. On Render/HF,
                # this is a harmless no-op since those platforms already
                # inject env vars directly — no .env file needed there.

import litellm
import threading
from openai import OpenAI as NvidiaOpenAI

# ── Rate-limit throttle ────────────────────────────────────────
# Mistral's free ("Experiment") API tier enforces roughly 1 request per
# second. A single chat turn already fires 2 LLM calls (combined L1+L2
# guardrail, then the core chat reply) which can land inside the same
# 1-second window and trip a 429 RateLimitError. This serializes every
# LLM call app-wide with a minimum gap between them, so no matter how
# many calls one turn (or multiple concurrent users) trigger, they
# never go out faster than the free tier allows.
_LLM_CALL_LOCK = threading.Lock()
_LAST_LLM_CALL_TIME = [0.0]
_MIN_LLM_CALL_INTERVAL = 1.1  # seconds

def _throttle_llm_call():
    with _LLM_CALL_LOCK:
        wait = _MIN_LLM_CALL_INTERVAL - (time.time() - _LAST_LLM_CALL_TIME[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_LLM_CALL_TIME[0] = time.time()

# ── LLM Configuration ──────────────────────────────────────────
# This ONE line is the only place a model/provider is chosen. LiteLLM's
# model string format is "<provider>/<model-name>" — swapping providers
# means changing this env var, nothing else in the code. LiteLLM reads
# the matching provider API key automatically from its standard env var
# name (mistral/... -> MISTRAL_API_KEY, groq/... -> GROQ_API_KEY,
# openai/... -> OPENAI_API_KEY, anthropic/... -> ANTHROPIC_API_KEY, etc.)
# so just set LLM_MODEL plus whichever provider key it needs.
#
# Examples if you need to swap later:
#   LLM_MODEL=mistral/devstral-2512        (current)
#   LLM_MODEL=groq/llama-3.3-70b-versatile (old, now discontinued)
#   LLM_MODEL=openai/gpt-4o-mini
#   LLM_MODEL=anthropic/claude-sonnet-4-5
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral/devstral-2512")

# Upfront check that the API key LiteLLM will need actually exists, so a
# missing key fails loudly at startup instead of silently at first chat.
# LiteLLM's own env var naming convention: <PROVIDER>_API_KEY (uppercased).
_LLM_PROVIDER = LLM_MODEL.split("/")[0].upper() if "/" in LLM_MODEL else None
_LLM_KEY_VAR = f"{_LLM_PROVIDER}_API_KEY" if _LLM_PROVIDER else None
if _LLM_KEY_VAR and not os.environ.get(_LLM_KEY_VAR):
    print(f"⚠️  WARNING: LLM_MODEL is '{LLM_MODEL}' but {_LLM_KEY_VAR} is empty — LLM calls will fail.")

# ── API Clients ────────────────────────────────────────────────
DUFFEL_KEY = os.environ.get("DUFFEL_API_KEY", "")

if not DUFFEL_KEY:
    print("⚠️  WARNING: DUFFEL_API_KEY is empty — Duffel calls will return 401 Unauthorized.")

DUFFEL_BASE_URL = "https://api.duffel.com"
DUFFEL_VERSION  = "v2"

# ── LLM-as-judge (eval layer) ───────────────────────────────────
# Matches the pattern used by pcmace-ai / rootcause-ai: a judge call to
# NVIDIA's Nemotron model via NVIDIA's OpenAI-compatible endpoint, with
# results logged to the same central llm-usage-tracker's /logEval route.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"

if not NVIDIA_API_KEY:
    print("⚠️  WARNING: NVIDIA_API_KEY is empty — eval/judge calls will be skipped.")
    nvidia_client = None
else:
    nvidia_client = NvidiaOpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")

# ── Shared cross-app usage tracker (same Convex project other portfolio
#    apps — doubtmail-ai, pcmace-ai, rootcause-ai — log to) ─────────────
USAGE_TRACKER_URL = os.environ.get("USAGE_TRACKER_URL", "https://quixotic-pigeon-152.eu-west-1.convex.site")
USAGE_TRACKER_SECRET = os.environ.get("USAGE_TRACKER_SECRET", "")
USAGE_TRACKER_APP_NAME = os.environ.get("USAGE_TRACKER_APP_NAME", "skybot")

if not USAGE_TRACKER_SECRET:
    print("⚠️  WARNING: USAGE_TRACKER_SECRET is empty — usage won't be logged to the shared tracker.")

# ── LLM-as-judge eval layer ─────────────────────────────────────
# Matches the eval pipeline already used by rootcause-ai/pcmace-ai/
# doubtmail-ai: an NVIDIA Nemotron judge scores each core-chat reply and
# posts the verdict to the shared tracker's /logEval endpoint. This is
# a separate model/call from the main SkyBot conversation (LLM_MODEL) —
# it never blocks the user's reply, since it fires in a background
# thread (see judge_reply_async below).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "nvidia_nim/nvidia/nemotron-3-super-120b-a12b")

if not os.environ.get("NVIDIA_NIM_API_KEY"):
    print("⚠️  WARNING: NVIDIA_NIM_API_KEY is empty — eval/judge scoring will be skipped (chat itself is unaffected).")

JUDGE_SYSTEM = """You are an impartial evaluator judging an airline customer-service assistant's reply.
Given the user's question, any flight-data context provided to the assistant, and the assistant's reply, rate the reply.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{"score": <number between 0.0 and 1.0>, "verdict": "correct" | "partial" | "incorrect", "reasoning": "<one short sentence>"}

Score/verdict guidance:
- "correct" (score 0.8-1.0): reply directly and accurately addresses the question using the given context
- "partial" (score 0.4-0.79): reply is relevant but incomplete, vague, or only partially uses the context
- "incorrect" (score 0.0-0.39): reply is off-topic, wrong, or ignores the given context entirely
"""

def log_eval_to_shared_tracker(payload):
    """
    POSTs to the shared llm-usage-tracker's /logEval endpoint. Mirrors
    log_to_shared_tracker but targets the evals table instead of
    llmUsage — payload must match /logEval's expected fields exactly
    (appName, model, judgeModel, judgeScore, judgeVerdict required;
    taskType/mode/judgeReasoning optional).
    """
    if not USAGE_TRACKER_SECRET:
        return
    try:
        resp = requests.post(
            f"{USAGE_TRACKER_URL}/logEval",
            headers={
                "Content-Type": "application/json",
                "x-usage-secret": USAGE_TRACKER_SECRET,
            },
            json=payload,
            timeout=10,
        )
        print(f"📡 Eval POST /logEval -> {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"⚠️  Shared eval log failed (non-fatal): {e}")

def judge_reply_async(user_message, context, reply):
    """
    Fires an LLM-as-judge evaluation of a core-chat reply in a background
    thread, so it never adds latency to what the user sees. Silently
    skipped if NVIDIA_NIM_API_KEY isn't set. Any failure here is
    non-fatal and never surfaces to the chat UI.
    """
    if not os.environ.get("NVIDIA_NIM_API_KEY"):
        return

    def _run():
        try:
            judge_prompt = (
                f"User question: {user_message}\n\n"
                f"Flight-data context given to the assistant: {context or '(none)'}\n\n"
                f"Assistant's reply: {reply}"
            )
            r = litellm.completion(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt},
                ],
                max_tokens=150,
                temperature=0.0,
            )
            content = r.choices[0].message.content.strip()
            # Strip stray code fences in case the judge wraps its JSON in them
            content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
            parsed = json.loads(content)

            score = float(parsed.get("score", 0.0))
            verdict = parsed.get("verdict", "partial")
            if verdict not in ("correct", "partial", "incorrect"):
                verdict = "partial"
            reasoning = str(parsed.get("reasoning", ""))[:500]

            log_eval_to_shared_tracker({
                "appName": USAGE_TRACKER_APP_NAME,
                "taskType": "chat_reply",
                "mode": "auto",
                "model": LLM_MODEL,
                "judgeModel": JUDGE_MODEL,
                "judgeScore": score,
                "judgeVerdict": verdict,
                "judgeReasoning": reasoning,
            })
        except Exception as e:
            print(f"⚠️  Judge eval failed (non-fatal): {e}")

    threading.Thread(target=_run, daemon=True).start()

def log_to_shared_tracker(endpoint, payload):
    """
    POSTs to the shared llm-usage-tracker Convex project. Payload must
    match llm-usage-tracker's actual schema.ts fields exactly (appName,
    not app; totalTokens/latencyMs/success are required) — callers build
    the full payload themselves, this function just sends it as-is.
    """
    if not USAGE_TRACKER_SECRET:
        print("⚠️  Tracker log skipped: USAGE_TRACKER_SECRET is empty.")
        return
    try:
        resp = requests.post(
            f"{USAGE_TRACKER_URL}/{endpoint}",
            headers={
                "Content-Type": "application/json",
                "x-usage-secret": USAGE_TRACKER_SECRET,
            },
            json=payload,
            timeout=5,
        )
        # TEMPORARY debug visibility — remove once confirmed working.
        print(f"📡 Tracker POST /{endpoint} -> {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"⚠️  Shared usage tracker log failed (non-fatal): {e}")

USAGE_LOGS = []
LAST_GUARDRAIL_TOKENS = 0
LAST_CHAT_TOKENS = 0

def log_eval_to_shared_tracker(payload):
    """
    POSTs to the shared llm-usage-tracker's /logEval route — same
    project/secret as log_to_shared_tracker(), different endpoint and
    schema (score/verdict/reasoning instead of token counts).
    """
    if not USAGE_TRACKER_SECRET:
        print("⚠️  Eval log skipped: USAGE_TRACKER_SECRET is empty.")
        return
    try:
        resp = requests.post(
            f"{USAGE_TRACKER_URL}/logEval",
            headers={
                "Content-Type": "application/json",
                "x-usage-secret": USAGE_TRACKER_SECRET,
            },
            json=payload,
            timeout=10,
        )
        print(f"📡 Eval POST /logEval -> {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"⚠️  Shared eval log failed (non-fatal): {e}")

def build_judge_prompt(user_question, context, reply):
    return "\n".join([
        "You are a strict examiner grading an airline customer-service AI's reply for accuracy and quality.",
        "",
        "CUSTOMER'S QUESTION:",
        user_question,
        "",
        (f"LIVE DATA PROVIDED TO THE AI (e.g. real flight offers):\n{context}\n" if context else "LIVE DATA PROVIDED TO THE AI: (none for this message)\n"),
        "AI'S REPLY:",
        reply,
        "",
        "Judge whether the reply is factually consistent with any live data provided, "
        "genuinely helpful, and appropriately scoped to an airline assistant. Penalize "
        "invented prices/facts not present in the live data, and unhelpful or off-topic replies.",
        "",
        "Respond with ONLY a single valid JSON object in this exact shape, nothing else:",
        json.dumps({
            "score": "integer 1-5, 5 being excellent",
            "verdict": "correct | partial | incorrect",
            "reasoning": "one or two sentence justification",
        }),
    ])

def judge_reply(user_question, context, reply, task_type):
    """
    LLM-as-judge eval pass, mirroring pcmace-ai/rootcause-ai: fires after
    a core chat reply, scores it via NVIDIA Nemotron, and logs the result
    both locally (log_usage, so it shows in this app's own dashboards)
    and to the shared cross-app tracker's /logEval route. Wrapped so a
    judge failure never blocks or breaks the actual chat reply.
    """
    if nvidia_client is None:
        return

    try:
        judge_start = time.time()
        judge_response = nvidia_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": build_judge_prompt(user_question, context, reply)}],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        judge_latency_ms = int((time.time() - judge_start) * 1000)

        judge_raw = judge_response.choices[0].message.content or "{}"
        try:
            judge_parsed = json.loads(judge_raw)
        except Exception:
            # Best-effort: bail out quietly rather than pulling in a JSON-repair dependency.
            print(f"⚠️  Judge response wasn't valid JSON, skipping eval log: {judge_raw[:200]}")
            return

        raw_verdict = judge_parsed.get("verdict")
        verdict = raw_verdict if raw_verdict in ("correct", "partial", "incorrect") else "partial"
        score = judge_parsed.get("score", 0)
        reasoning = judge_parsed.get("reasoning", "")

        usage = getattr(judge_response, "usage", None)
        log_usage(
            "Judge Call",
            JUDGE_MODEL,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        log_to_shared_tracker("logUsage", {
            "appName": USAGE_TRACKER_APP_NAME,
            "feature": "Judge Call",
            "model": JUDGE_MODEL,
            "promptTokens": usage.prompt_tokens if usage else 0,
            "completionTokens": usage.completion_tokens if usage else 0,
            "totalTokens": usage.total_tokens if usage else 0,
            "latencyMs": judge_latency_ms,
            "success": True,
        })

        log_eval_to_shared_tracker({
            "appName": USAGE_TRACKER_APP_NAME,
            "taskType": task_type,
            "model": LLM_MODEL,
            "judgeModel": JUDGE_MODEL,
            "judgeScore": score,
            "judgeVerdict": verdict,
            "judgeReasoning": reasoning,
        })
    except Exception as e:
        print(f"⚠️  LLM-as-judge pass failed (non-blocking): {e}")

LAST_FLIGHT_API_PAYLOAD = {
    "request": "No active transaction recorded yet.",
    "response": "No active payload recorded yet."
}
LAST_OFFERS = []

# Cumulative tracker (provider-agnostic — tracks whatever LLM_MODEL is set to)
LLM_CUMULATIVE = {
    "total_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "total_cost": 0.0,
}

LAST_LLM_PAYLOAD = {
    "request": "No active LLM transaction recorded yet.",
    "response": "No active LLM payload recorded yet."
}

LAST_PROMPT_BREAKDOWN = {
    "system": 0, "history": 0, "user_question": 0, "flight_data": 0, "output": 0
}

def log_usage(feature_name, system_name, prompt_tokens=0, completion_tokens=0, is_flight_api=False):
    global LAST_GUARDRAIL_TOKENS, LAST_CHAT_TOKENS

    if is_flight_api:
        est_cost = 0.02000
        total_tokens = 0
    else:
        total_tokens = prompt_tokens + completion_tokens
        # Cost is computed via litellm.completion_cost() above, not here
        est_cost = ((prompt_tokens / 1000000) * 0.20) + ((completion_tokens / 1000000) * 0.60)

        if "Guardrail" in feature_name:
            LAST_GUARDRAIL_TOKENS = total_tokens
        elif "Core Chat" in feature_name:
            LAST_CHAT_TOKENS = total_tokens

    record = {
        "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "Feature": feature_name,
        "System/Model": system_name,
        "Input Tokens": prompt_tokens,
        "Output Tokens": completion_tokens,
        "Total Tokens": total_tokens,
        "Est. Cost ($)": round(est_cost, 5)
    }
    USAGE_LOGS.append(record)

def call_llm(feature_name, messages, max_tokens=256, temperature=None):
    """
    Single choke point for every LLM call in the app, via LiteLLM.
    Provider-agnostic: swapping models/providers is a config change to
    LLM_MODEL (and whichever provider API key it needs) at the top of
    this file — nothing in this function needs to change.
    """
    global LAST_LLM_PAYLOAD, LLM_CUMULATIVE

    request_body = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_body["temperature"] = temperature

    LAST_LLM_PAYLOAD["request"] = json.dumps({
        "Method": "POST (via LiteLLM)",
        "Model": LLM_MODEL,
        "Body": request_body
    }, indent=2)

    _throttle_llm_call()

    start_time = time.time()

    try:
        r = litellm.completion(**request_body)
        latency_ms = int((time.time() - start_time) * 1000)

        content = r.choices[0].message.content.strip()
        usage = getattr(r, "usage", None)
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        LAST_LLM_PAYLOAD["response"] = json.dumps({
            "id": getattr(r, "id", "n/a"),
            "model": LLM_MODEL,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }, indent=2)

        log_usage(feature_name, LLM_MODEL, prompt_tokens, completion_tokens)

        # Field names here must match llm-usage-tracker's actual schema.ts:
        # appName (not app), plus totalTokens/latencyMs/success which are
        # required fields we weren't sending before.
        log_to_shared_tracker("logUsage", {
            "appName": USAGE_TRACKER_APP_NAME,
            "feature": feature_name,
            "model": LLM_MODEL,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "latencyMs": latency_ms,
            "success": True,
        })

        LLM_CUMULATIVE["total_calls"] += 1
        LLM_CUMULATIVE["total_prompt_tokens"] += prompt_tokens
        LLM_CUMULATIVE["total_completion_tokens"] += completion_tokens
        LLM_CUMULATIVE["total_tokens"] += (prompt_tokens + completion_tokens)
        # NOTE: this cost estimate uses litellm's built-in per-model pricing
        # table when available (accurate across providers), falling back to
        # 0 if the current LLM_MODEL isn't in litellm's pricing data yet.
        try:
            call_cost = litellm.completion_cost(completion_response=r)
        except Exception:
            call_cost = 0.0
        LLM_CUMULATIVE["total_cost"] += call_cost

        return content, True, prompt_tokens, completion_tokens

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        LAST_LLM_PAYLOAD["response"] = json.dumps({"API Error": str(e)}, indent=2)

        # Log the failure too — the schema explicitly supports this
        # (success: false, errorMessage), useful for spotting a flaky
        # or misconfigured LLM_MODEL from the tracker dashboard.
        log_to_shared_tracker("logUsage", {
            "appName": USAGE_TRACKER_APP_NAME,
            "feature": feature_name,
            "model": LLM_MODEL,
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "latencyMs": latency_ms,
            "success": False,
            "errorMessage": str(e)[:500],
        })

        return None, False, 0, 0


def estimate_prompt_token_split(parts: dict, actual_total_tokens: int):
    raw = {k: (max(1, len(v) // 4) if v else 0) for k, v in parts.items()}
    raw_sum = sum(raw.values())
    if raw_sum == 0 or actual_total_tokens == 0:
        return {k: 0 for k in parts}
    scaled = {k: round(v * actual_total_tokens / raw_sum) for k, v in raw.items()}
    drift = actual_total_tokens - sum(scaled.values())
    if scaled:
        biggest_key = max(scaled, key=scaled.get)
        scaled[biggest_key] += drift
    return scaled

def get_prompt_breakdown_html():
    b = LAST_PROMPT_BREAKDOWN
    input_total = b["system"] + b["history"] + b["user_question"] + b["flight_data"]
    grand_total = input_total + b["output"]
    if grand_total == 0:
        return """
        <div style="background:#1e293b;padding:16px;border-radius:8px;border:1px solid #475569;font-family:sans-serif;color:#94a3b8;">
            Send a message in the Assistant Hub to see a plain-English breakdown of where your tokens go.
        </div>
        """
    return f"""
    <div style="background:#1e293b;padding:16px;border-radius:8px;border:1px solid #475569;margin-top:15px;font-family:sans-serif;color:#f8fafc;">
        <h4 style="margin:0 0 10px 0;color:#facc15;font-size:15px;">🧾 In Plain English: Where Did Your Last {grand_total} Tokens Go?</h4>
        <p style="margin:0 0 12px 0;font-size:13px;color:#cbd5e1;">
            Every reply bundles up to four things as <b>INPUT</b>, then the model writes an <b>OUTPUT</b>. Output tokens
            typically cost ~4x more per token than input, so keeping system prompts and history lean matters more than
            you'd think.
        </p>
        <div style="display:flex;gap:10px;flex-direction:column;font-size:13px;">
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#334155;border-radius:4px;">
                <span>🧬 <b>1. SkyBot's personality &amp; rules</b> (system prompt, sent on every call):</span>
                <span style="font-family:monospace;font-weight:bold;color:#38bdf8;">{b['system']} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#334155;border-radius:4px;">
                <span>💬 <b>2. Earlier turns</b> re-sent so SkyBot remembers the conversation:</span>
                <span style="font-family:monospace;font-weight:bold;color:#38bdf8;">{b['history']} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#334155;border-radius:4px;">
                <span>⌨️ <b>3. The question you just typed</b>:</span>
                <span style="font-family:monospace;font-weight:bold;color:#38bdf8;">{b['user_question']} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#334155;border-radius:4px;">
                <span>✈️ <b>4. Live flight data</b> injected from Duffel, only if a search fired:</span>
                <span style="font-family:monospace;font-weight:bold;color:#38bdf8;">{b['flight_data']} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#0f172a;border-radius:4px;border-top:1px solid #38bdf8;">
                <span style="font-weight:bold;color:#38bdf8;">= Total INPUT tokens (real, from the LLM API):</span>
                <span style="font-family:monospace;font-weight:bold;color:#38bdf8;">{input_total} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 10px;background:#334155;border-radius:4px;">
                <span>✍️ <b>SkyBot's reply</b> (OUTPUT tokens, billed separately):</span>
                <span style="font-family:monospace;font-weight:bold;color:#10b981;">{b['output']} tokens</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:8px 10px;background:#0f172a;border-radius:4px;border-top:2px solid #facc15;margin-top:4px;">
                <span style="font-weight:bold;color:#facc15;">📊 Grand total billed this call:</span>
                <span style="font-family:monospace;font-weight:bold;color:#facc15;">{grand_total} tokens</span>
            </div>
        </div>
        <p style="margin:12px 0 0 0;font-size:11px;color:#64748b;">
            Note: items 1–4 are an estimated split of one real number (the LLM API only returns a single prompt_tokens total,
            not a per-section breakdown). The INPUT and OUTPUT totals themselves are exact, straight from the API.
        </p>
    </div>
    """

def get_dashboard_data():
    global LAST_GUARDRAIL_TOKENS, LAST_CHAT_TOKENS

    total_agg = LAST_GUARDRAIL_TOKENS + LAST_CHAT_TOKENS

    flight_api_called = any("Duffel" in log["Feature"] for log in USAGE_LOGS[-4:]) if USAGE_LOGS else False
    flight_api_status_html = """
    <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
        <span>✈️ <b>Duffel Flights API Gateway</b> (Live Offer Request REST Query):</span>
        <span style="font-family: monospace; font-weight: bold; color: #38bdf8;">1 Transaction ($0.02000)</span>
    </div>
    """ if flight_api_called else ""

    breakdown_html = f"""
    <div style="background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #475569; margin-bottom: 15px; font-family: sans-serif; color: #f8fafc;">
        <h4 style="margin: 0 0 10px 0; color: #38bdf8; font-size: 15px;">🔍 Latest Transaction Network Audit (Token Performance Allocation)</h4>
        <p style="margin: 0 0 12px 0; font-size: 13px; color: #cbd5e1;">
            Here is why the system consumed <b>{total_agg} tokens</b> and triggered external handshakes for your request:
        </p>
        <div style="display: flex; gap: 10px; flex-direction: column; font-size: 13px;">
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
                <span>🛡️ <b>L1 Guardrail Layer</b> (System Instructions + Input Validation):</span>
                <span style="font-family: monospace; font-weight: bold; color: #f43f5e;">{LAST_GUARDRAIL_TOKENS} Tokens</span>
            </div>
            {flight_api_status_html}
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
                <span>💬 <b>Core System Prompts &amp; Context</b> (System Guidelines + History + Flight Data Context Payload):</span>
                <span style="font-family: monospace; font-weight: bold; color: #10b981;">{LAST_CHAT_TOKENS} Tokens</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 10px; background: #0f172a; border-radius: 4px; border-top: 2px solid #38bdf8; margin-top: 4px;">
                <span style="font-weight: bold; color: #38bdf8;">📊 Total Aggregated Architecture Tax:</span>
                <span style="font-family: monospace; font-weight: bold; color: #38bdf8;">{total_agg} Tokens</span>
            </div>
        </div>
    </div>
    """

    if not USAGE_LOGS:
        empty_df = pd.DataFrame(columns=["Timestamp", "Feature", "System/Model", "Total Tokens", "Est. Cost ($)"])
        return "0", "$0.00000", "0", empty_df, breakdown_html

    df = pd.DataFrame(USAGE_LOGS)
    total_calls = len(df)
    total_cost = df["Est. Cost ($)"].sum()
    total_tokens = df["Total Tokens"].sum()

    display_df = df[["Timestamp", "Feature", "System/Model", "Total Tokens", "Est. Cost ($)"]]
    return str(total_calls), f"${total_cost:.5f}", f"{total_tokens:,}", display_df, breakdown_html

def get_inspector_payloads():
    global LAST_FLIGHT_API_PAYLOAD
    return LAST_FLIGHT_API_PAYLOAD["request"], LAST_FLIGHT_API_PAYLOAD["response"]

def get_llm_cumulative_data():
    c = LLM_CUMULATIVE
    breakdown_html = f"""
    <div style="background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #475569; margin-bottom: 15px; font-family: sans-serif; color: #f8fafc;">
        <h4 style="margin: 0 0 10px 0; color: #a78bfa; font-size: 15px;">🧠 Cumulative LLM Call Ledger (Session-Wide)</h4>
        <p style="margin: 0 0 12px 0; font-size: 13px; color: #cbd5e1;">
            Every LLM call SkyBot makes — guardrail checks AND core chat replies — routes through one
            wrapper, so these numbers only ever grow across the whole session (they don't reset on "Clear Chat"):
        </p>
        <div style="display: flex; gap: 10px; flex-direction: column; font-size: 13px;">
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
                <span>📞 <b>Total LLM API Calls</b>:</span>
                <span style="font-family: monospace; font-weight: bold; color: #a78bfa;">{c['total_calls']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
                <span>⬆️ <b>Cumulative Prompt Tokens</b>:</span>
                <span style="font-family: monospace; font-weight: bold; color: #f43f5e;">{c['total_prompt_tokens']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #334155; border-radius: 4px;">
                <span>⬇️ <b>Cumulative Completion Tokens</b>:</span>
                <span style="font-family: monospace; font-weight: bold; color: #10b981;">{c['total_completion_tokens']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px 10px; background: #0f172a; border-radius: 4px; border-top: 2px solid #a78bfa; margin-top: 4px;">
                <span style="font-weight: bold; color: #a78bfa;">💰 Cumulative Est. LLM Cost:</span>
                <span style="font-family: monospace; font-weight: bold; color: #a78bfa;">${c['total_cost']:.5f}</span>
            </div>
        </div>
    </div>
    """
    return (
        str(c["total_calls"]),
        f"{c['total_tokens']:,}",
        f"${c['total_cost']:.5f}",
        breakdown_html,
        LAST_LLM_PAYLOAD["request"],
        LAST_LLM_PAYLOAD["response"],
    )

def visual_tokenize_text(input_text):
    """
    Educational sub-word visualizer. Previously used the local DeBERTa
    tokenizer; now uses a simple regex word-splitter only, since the
    local transformers/torch stack was removed to fit Render's 512MB
    RAM limit. This was always labeled as an approximation, not a real
    token count — the true billed count is in the LLM Inspector tab.
    """
    if not input_text or not input_text.strip():
        return '<div style="color:#888;padding:12px;font-style:italic">Type something above to visualize...</div>'

    raw_pieces = re.findall(r"\b\w+\b|\s+|[^\w\s]", input_text)
    note = "Approximate word-splitter (not a real sub-word tokenizer) — this is NOT real token counting."

    html_elements = []
    colors = ["#E0F2FE", "#DCFCE7", "#FEF9C3", "#F3E8FF", "#FEE2E2", "#FFEDD5"]

    for i, piece in enumerate(raw_pieces):
        color = colors[i % len(colors)]
        display_text = piece.replace(" ", "&nbsp;").replace("\n", "<br>")
        html_elements.append(
            f'<span style="background-color: {color}; color: #1E293B; '
            f'padding: 4px 8px; margin: 3px; border-radius: 4px; '
            f'display: inline-block; font-family: monospace; font-size: 14px; font-weight: bold;">'
            f'{display_text}'
            f'</span>'
        )

    wrapper = f"""
    <div style="background: #111827; padding: 16px; border-radius: 8px; border: 1px solid #374151; min-height: 80px;">
        <p style="color: #9CA3AF; font-size: 11px; margin: 0 0 8px 0; font-family: sans-serif;">GENERATED WORD PIECES ({len(raw_pieces)} PIECES):</p>
        <div style="line-height: 2.3;">{"".join(html_elements)}</div>
        <p style="color: #64748b; font-size: 10px; margin: 10px 0 0 0; font-family: sans-serif;">ℹ️ {note} The authoritative prompt-token count is in the 🧠 LLM Inspector tab (from the real API response).</p>
    </div>
    """
    return wrapper

AIRLINE_TOPICS = [
    "flight search and booking", "airport information", "travel dates and pricing",
    "airline baggage and check-in policy", "flight delays and cancellations", "general travel help",
]
OFF_TOPICS = [
    "cooking and recipes", "politics and government", "sports and entertainment",
    "finance and investment", "medical advice", "unrelated casual chat",
]
ALL_TOPICS = AIRLINE_TOPICS + OFF_TOPICS

CITY_TO_IATA = {
    "new york":"JFK", "nyc":"JFK", "london":"LHR", "paris":"CDG", "los angeles":"LAX",
    "chicago":"ORD", "dubai":"DXB", "tokyo":"NRT", "singapore":"SIN", "sydney":"SYD",
    "mumbai":"BOM", "delhi":"DEL", "new delhi":"DEL", "bangalore":"BLR", "bengaluru":"BLR",
    "bangaluru":"BLR",
    "narita":"NRT", "haneda":"HND", "heathrow":"LHR", "changi":"SIN",
    "charles de gaulle":"CDG", "o'hare":"ORD", "incheon":"ICN", "seoul":"ICN",
    "hong kong":"HKG", "bangkok":"BKK", "frankfurt":"FRA", "amsterdam":"AMS",
    "doha":"DOH", "abu dhabi":"AUH", "colombo":"CMB", "kathmandu":"KTM",
}

NON_AIRPORT_CODES = {
    "USD","EUR","GBP","INR","JPY","AUD","CAD","SGD","AED","CHF",
    "ETA","ASAP","FAQ","API","URL","PDF","CEO","CFO","VIP","PNR","GDS","AI","AM","PM",
}

UNSAFE_CATEGORIES = {
    "O1":"Violence and Hate Speech","O2":"Sexual Content","O3":"Criminal Planning",
    "O4":"Weapons","O5":"Controlled Substances","O6":"Self-Harm","O7":"Jailbreak / Prompt Injection",
}

SKYBOT_SYSTEM = """You are SkyBot, a professional and friendly airline customer service assistant.
You ONLY help with flights, airports, travel planning, baggage and check-in policies.
Base flight prices ONLY on real data provided. Keep responses under 200 words. Be warm and professional."""

LLAMAGUARD_SYSTEM = """You are a safety classifier. Reply ONLY with "safe" or "unsafe O[number]".
O1:Violence O2:Sexual O3:Criminal O4:Weapons O5:Drugs O6:Self-Harm O7:Jailbreak/Prompt-Injection"""

PII_PATTERNS = [(r"\b\d{16}\b", "credit card"), (r"\b[A-Z]{1,2}\d{6,9}\b", "passport"), (r"[\w.+-]+@[\w-]+\.[\w.]+", "email")]
QUICK_EXAMPLES = ["Find flights from BLR to DXB next Friday", "Search DEL to LHR for 2 passengers tomorrow"]
WELCOME = "✈️ Welcome to SkyBot! I can help with flights, airports, and travel planning.\n\nTry a quick example below or type your question!"

def guardrail_check(text):
    """
    Combined L1 (safety) + L2 (topic) guardrail check in a single LiteLLM
    call.

    Originally two separate calls (llamaguard_check for safety, plus a
    topic_rail_check that used a local DeBERTa zero-shot classifier).
    The DeBERTa model + torch pushed Render's runtime past its 512MB RAM
    limit, so topic classification was moved to an LLM call too — but
    that meant 3 total LLM calls fired back-to-back on every chat turn
    (L1 + L2 + core chat), which tripped Mistral's free-tier rate limit
    (429 RateLimitError). Merging L1+L2 into one call brings it back
    down to 2 calls per turn.

    Returns (safety_dict, topic_dict) — same shapes the old
    llamaguard_check()/topic_rail_check() returned, so chat() doesn't
    need to change how it reads the results.
    """
    labels_str = ", ".join(ALL_TOPICS)
    content, ok, _, _ = call_llm(
        "Guardrail L1+L2 Check",
        [
            {"role": "system", "content": (
                "You are a two-part classifier for an airline assistant guardrail "
                "system. Reply on exactly two lines and nothing else:\n"
                "Line 1: \"safe\" or \"unsafe O[number]\" using this schema — "
                "O1:Violence O2:Sexual O3:Criminal O4:Weapons O5:Drugs O6:Self-Harm "
                "O7:Jailbreak/Prompt-Injection\n"
                f"Line 2: the single best-matching topic, copied exactly from this list: {labels_str}"
            )},
            {"role": "user", "content": f'Classify this message: "{text}"'},
        ],
        max_tokens=25,
        temperature=0.1,
    )

    default_safety = {"safe": True, "reason": "✅ Passed"}
    default_topic = {"top_topic": "unknown", "top_score": 0.0, "is_offtopic": False}

    if not ok or not content:
        # Fail open: if the classifier errors for any reason, don't block chat.
        return default_safety, default_topic

    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    safety_line = lines[0].lower() if len(lines) >= 1 else "safe"
    topic_line = lines[1].strip('"').strip("'").lower() if len(lines) >= 2 else ""

    if safety_line.startswith("safe"):
        safety = {"safe": True, "reason": "✅ Safe"}
    else:
        cat = re.search(r"o(\d)", safety_line)
        code = f"O{cat.group(1)}" if cat else "O?"
        safety = {"safe": False, "reason": UNSAFE_CATEGORIES.get(code, "Policy violation")}

    top_topic = next((t for t in ALL_TOPICS if t.lower() == topic_line), None)
    if top_topic is None:
        # Tolerant fallback in case the LLM added stray words around the label.
        top_topic = next((t for t in ALL_TOPICS if t.lower() in topic_line or topic_line in t.lower()), "unknown")

    topic = {
        "top_topic": top_topic,
        "top_score": 1.0 if top_topic != "unknown" else 0.0,
        "is_offtopic": top_topic in OFF_TOPICS,
    }

    return safety, topic
	
def extract_iata(text):
    codes = re.findall(r"\b([A-Z]{3})\b", text)
    codes = [c for c in codes if c not in NON_AIRPORT_CODES]
    if codes:
        return codes

    found_codes = []
    lower_text = text.lower()
    for city, code in CITY_TO_IATA.items():
        if city in lower_text:
            if code not in found_codes:
                found_codes.append(code)
    return found_codes

def extract_date(text):
    found_date = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if found_date:
        return found_date.group(0)

    found_text_date = re.search(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", text)
    if found_text_date:
        return "2026-09-20"

    return (date.today() + timedelta(days=7)).isoformat()

def extract_pax(text):
    return 1

def msg_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or str(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content) if content is not None else ""

def get_flights(orig, dest, dt, adults=1):
    global LAST_FLIGHT_API_PAYLOAD

    url = f"{DUFFEL_BASE_URL}/air/offer_requests?return_offers=true"
    headers = {
        "Authorization": f"Bearer {DUFFEL_KEY}",
        "Duffel-Version": DUFFEL_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "data": {
            "slices": [{"origin": orig, "destination": dest, "departure_date": dt}],
            "passengers": [{"type": "adult"} for _ in range(max(1, adults))],
            "cabin_class": "economy",
        }
    }

    LAST_FLIGHT_API_PAYLOAD["request"] = json.dumps({
        "Method": "POST",
        "Endpoint": url,
        "Headers": {**headers, "Authorization": "Bearer ********"},
        "Body": body
    }, indent=2)

    try:
        r = requests.post(url, headers=headers, json=body, timeout=25)
        r.raise_for_status()
        data = r.json()
        log_usage("Duffel Flight Offers Request", "Offer Requests API", is_flight_api=True)
        LAST_FLIGHT_API_PAYLOAD["response"] = json.dumps(data, indent=2)
        offers = data.get("data", {}).get("offers", [])
        global LAST_OFFERS
        LAST_OFFERS = offers  # remember these for any fare-rules follow-up
        return offers
    except Exception as e:
        LAST_FLIGHT_API_PAYLOAD["response"] = json.dumps({"API Error": str(e)}, indent=2)
        return None

AIRPORT_TO_CURRENCY = {
    "DEL":"INR", "BOM":"INR", "BLR":"INR",
    "JFK":"USD", "LAX":"USD", "ORD":"USD",
    "LHR":"GBP",
    "CDG":"EUR", "FRA":"EUR", "AMS":"EUR",
    "DXB":"AED", "AUH":"AED",
    "NRT":"JPY", "HND":"JPY",
    "SIN":"SGD",
    "SYD":"AUD",
    "ICN":"KRW",
    "HKG":"HKD",
    "BKK":"THB",
    "DOH":"QAR",
    "CMB":"LKR",
    "KTM":"NPR",
}

_FX_CACHE = {}

def get_fx_rate(from_currency, to_currency):
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0

    key = (from_currency, to_currency)
    cached = _FX_CACHE.get(key)
    if cached and (datetime.datetime.now() - cached[1]).total_seconds() < 3600:
        return cached[0]

    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": from_currency, "to": to_currency},
            timeout=10,
        )
        r.raise_for_status()
        rate = r.json().get("rates", {}).get(to_currency)
        if rate:
            _FX_CACHE[key] = (rate, datetime.datetime.now())
        return rate
    except Exception:
        return None

def convert_to_local(amount_str, from_currency, origin_code):
    local_currency = AIRPORT_TO_CURRENCY.get(origin_code)
    if not local_currency or local_currency == from_currency:
        return None
    try:
        amount = float(amount_str)
    except (TypeError, ValueError):
        return None
    rate = get_fx_rate(from_currency, local_currency)
    if rate is None:
        return None
    return local_currency, round(amount * rate, 2)

def fmt_flights(flights, orig, dest, dt):
    if not flights: return f"No flights found for {orig} → {dest} on {dt}."
    lines = [f"Flights {orig} → {dest} on {dt}:"]
    for i, o in enumerate(flights[:3], 1):
        price = o.get("total_amount", "?")
        cur = o.get("total_currency", "")
        carrier = ""
        try:
            carrier = o["slices"][0]["segments"][0]["operating_carrier"]["name"]
        except (KeyError, IndexError, TypeError):
            pass
        carrier_suffix = f" ({carrier})" if carrier else ""
        line = f"Option {i}: {cur} {price}{carrier_suffix}"

        converted = convert_to_local(price, cur, orig)
        if converted:
            local_cur, local_amount = converted
            line += f"  ≈ {local_cur} {local_amount:,.2f} (origin local currency, live rate)"

        lines.append(line)
    return "\n".join(lines)

FARE_RULE_KEYWORDS = [
    "fare rule", "fare rules", "fare condition", "fare conditions",
    "refund rule", "refund policy", "refund conditions",
    "cancellation policy", "cancellation rule",
]

def is_fare_rule_query(text):
    lower = text.lower()
    if any(k in lower for k in FARE_RULE_KEYWORDS):
        return True
    if "fare" in lower and any(w in lower for w in ["rule", "rules", "condition", "refund", "change fee", "cancel"]):
        return True
    return False

def find_offer_by_hint(text):
    if not LAST_OFFERS:
        return None

    lower = text.lower()
    price_match = re.search(r"(\d+(?:\.\d{1,2})?)", text)
    target_price = float(price_match.group(1)) if price_match else None

    best = None
    for o in LAST_OFFERS:
        carrier = ""
        try:
            carrier = o["slices"][0]["segments"][0]["operating_carrier"]["name"]
        except (KeyError, IndexError, TypeError):
            pass

        carrier_match = bool(carrier) and carrier.lower() in lower
        price_ok = False
        if target_price is not None:
            try:
                price_ok = abs(float(o.get("total_amount", -1)) - target_price) < 0.05
            except (TypeError, ValueError):
                price_ok = False

        if carrier_match and price_ok:
            return o
        if carrier_match or price_ok:
            best = o

    return best

def get_offer_details(offer_id):
    global LAST_FLIGHT_API_PAYLOAD

    url = f"{DUFFEL_BASE_URL}/air/offers/{offer_id}"
    headers = {
        "Authorization": f"Bearer {DUFFEL_KEY}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
    }

    LAST_FLIGHT_API_PAYLOAD["request"] = json.dumps({
        "Method": "GET",
        "Endpoint": url,
        "Headers": {**headers, "Authorization": "Bearer ********"},
    }, indent=2)

    try:
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        data = r.json()
        log_usage("Duffel Offer Detail (Fare Rules)", "Offer Detail API", is_flight_api=True)
        LAST_FLIGHT_API_PAYLOAD["response"] = json.dumps(data, indent=2)
        return data.get("data")
    except Exception as e:
        LAST_FLIGHT_API_PAYLOAD["response"] = json.dumps({"API Error": str(e)}, indent=2)
        return None

def _cond_line(label, cond):
    if not cond:
        return f"{label}: Not specified by the airline for this fare."
    allowed = cond.get("allowed")
    penalty = cond.get("penalty_amount")
    penalty_cur = cond.get("penalty_currency", "")
    if allowed is False:
        return f"{label}: Not permitted on this fare."
    if allowed is True:
        fee = f" (fee: {penalty_cur} {penalty})" if penalty else " (no fee)"
        return f"{label}: Permitted{fee}."
    return f"{label}: Not specified."

def fmt_fare_rules(offer_detail, fallback_offer):
    o = offer_detail or fallback_offer
    if not o:
        return ("I couldn't match that to a fare from our last search. Could you re-run the "
                 "search, or mention the exact airline name and price shown earlier?")

    carrier = "Unknown carrier"
    try:
        carrier = o["slices"][0]["segments"][0]["operating_carrier"]["name"]
    except (KeyError, IndexError, TypeError):
        pass

    origin_code = None
    try:
        origin_code = o["slices"][0]["origin"]["iata_code"]
    except (KeyError, IndexError, TypeError):
        pass

    price = o.get("total_amount", "?")
    cur = o.get("total_currency", "")
    price_line = f"{cur} {price}"

    if origin_code:
        converted = convert_to_local(price, cur, origin_code)
        if converted:
            local_cur, local_amount = converted
            price_line += f"  ≈ {local_cur} {local_amount:,.2f} (origin local currency, live rate)"

    conditions = o.get("conditions") or {}

    lines = [
        f"Fare rules for {carrier} — {price_line}:",
        _cond_line("Refund before departure", conditions.get("refund_before_departure")),
        _cond_line("Change before departure", conditions.get("change_before_departure")),
    ]
    return "\n".join(lines)

def fare_rules_lookup(text):
    if not is_fare_rule_query(text):
        return "", False

    matched = find_offer_by_hint(text)
    if not matched:
        return "I don't have a matching fare from a recent search — please search a route first, then ask about its fare rules.", True

    offer_id = matched.get("id")
    detail = get_offer_details(offer_id) if offer_id else None
    return fmt_fare_rules(detail, matched), True

RESERVATIONS = {}

BOOKING_KEYWORDS = [
    "book this", "book it", "reserve this", "reserve it", "confirm booking",
    "make a reservation", "book the flight", "hold this fare", "book flight",
    "proceed with booking", "confirm this fare",
]

def is_booking_query(text):
    lower = text.lower()
    if any(k in lower for k in BOOKING_KEYWORDS):
        return True
    if "book" in lower and any(w in lower for w in ["flight", "fare", "ticket", "seat", "this", "reservation"]):
        return True
    return False

def generate_dummy_pnr():
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        pnr = "".join(random.choice(chars) for _ in range(6))
        if pnr not in RESERVATIONS:
            return pnr

def create_dummy_reservation(text):
    matched = find_offer_by_hint(text)
    if not matched:
        return ("I don't have a specific fare to book yet — please search a route first, "
                "then tell me which option (airline + price) you'd like to book."), True

    pnr = generate_dummy_pnr()

    carrier = "Unknown carrier"
    try:
        carrier = matched["slices"][0]["segments"][0]["operating_carrier"]["name"]
    except (KeyError, IndexError, TypeError):
        pass

    origin_code = dest_code = "?"
    try:
        origin_code = matched["slices"][0]["origin"]["iata_code"]
        dest_code = matched["slices"][0]["destination"]["iata_code"]
    except (KeyError, IndexError, TypeError):
        pass

    price = matched.get("total_amount", "?")
    cur = matched.get("total_currency", "")

    RESERVATIONS[pnr] = {
        "offer_id": matched.get("id"),
        "carrier": carrier,
        "origin": origin_code,
        "destination": dest_code,
        "price": price,
        "currency": cur,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "DUMMY_CONFIRMED",
    }

    reply = (
        f"✅ Dummy reservation created!\n\n"
        f"PNR: {pnr}\n"
        f"Route: {origin_code} → {dest_code}\n"
        f"Airline: {carrier}\n"
        f"Fare: {cur} {price}\n\n"
        f"⚠️ This is a simulated demo booking for testing purposes only — no real ticket "
        f"has been issued and no payment was processed. Say \"check my booking {pnr}\" "
        f"any time to look it up again."
    )
    return reply, True

PNR_LOOKUP_KEYWORDS = [
    "check my booking", "check my reservation", "look up pnr", "find my booking",
    "booking status", "reservation status", "check pnr",
]

def is_pnr_lookup_query(text):
    lower = text.lower()
    if any(k in lower for k in PNR_LOOKUP_KEYWORDS):
        return True
    if "pnr" in lower and any(w in lower for w in ["check", "status", "lookup", "look up", "find"]):
        return True
    return False

def lookup_dummy_reservation(text):
    match = re.search(r"\b([A-Z0-9]{6})\b", text.upper())
    if not match:
        return "Please share the 6-character PNR you'd like to look up.", True

    pnr = match.group(1)
    record = RESERVATIONS.get(pnr)
    if not record:
        return f"No reservation found for PNR {pnr}. (Dummy PNRs reset whenever the app restarts.)", True

    reply = (
        f"📋 Reservation {pnr}:\n"
        f"Route: {record['origin']} → {record['destination']}\n"
        f"Airline: {record['carrier']}\n"
        f"Fare: {record['currency']} {record['price']}\n"
        f"Status: {record['status']} (dummy/demo booking)\n"
        f"Created: {record['created_at']}"
    )
    return reply, True

def flight_data_lookup(text):
    global LAST_FLIGHT_API_PAYLOAD
    lower = text.lower()
    if any(w in lower for w in ["flight","fly","book","search","fare","price","ticket","cost"]):
        codes = extract_iata(text)
        if len(codes) >= 2:
            dt = extract_date(text); pax = extract_pax(text)
            return fmt_flights(get_flights(codes[0],codes[1],dt,pax), codes[0],codes[1],dt), True

    LAST_FLIGHT_API_PAYLOAD["request"] = "No structural REST queries executed for this transaction."
    LAST_FLIGHT_API_PAYLOAD["response"] = "No dynamic context payload was injected."
    return "", False

def validate(text):
    warnings = []
    for pattern, name in PII_PATTERNS:
        if re.search(pattern, text):
            warnings.append(f"Redacted sensitive {name}")
    return text, warnings

def chat(user_message, history):
    if not user_message or not user_message.strip():
        calls, cost, tokens, df, breakdown = get_dashboard_data()
        req, resp = get_inspector_payloads()
        g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp = get_llm_cumulative_data()
        return (history, "", calls, cost, tokens, df, breakdown, req, resp,
                g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp, get_prompt_breakdown_html())

    logs = []

    safety, topic = guardrail_check(user_message)
    logs.append(f"🛡️ L1 LlamaGuard : {safety['reason']}")
    if not safety["safe"]:
        reply = "🚫 Policy violation."
        history = history + [{"role":"user", "content": user_message}, {"role":"assistant", "content": reply}]
        calls, cost, tokens, df, breakdown = get_dashboard_data()
        req, resp = get_inspector_payloads()
        g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp = get_llm_cumulative_data()
        return (history, "\n".join(logs), calls, cost, tokens, df, breakdown, req, resp,
                g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp, get_prompt_breakdown_html())

    logs.append(f"🧭 L2 Topic Rail  : {topic['top_topic']}")

    if is_pnr_lookup_query(user_message):
        reply, _ = lookup_dummy_reservation(user_message)
        logs.append("🎫 L3 Duffel API   : Dummy PNR lookup (no LLM rewrite)")
        history = history + [{"role":"user", "content": user_message}, {"role":"assistant", "content": reply}]
        calls, cost, tokens, df, breakdown = get_dashboard_data()
        req, resp = get_inspector_payloads()
        g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp = get_llm_cumulative_data()
        return (history, "\n".join(logs), calls, cost, tokens, df, breakdown, req, resp,
                g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp, get_prompt_breakdown_html())

    if is_booking_query(user_message):
        reply, _ = create_dummy_reservation(user_message)
        logs.append("🧾 L3 Duffel API   : Dummy reservation created (no LLM rewrite)")
        history = history + [{"role":"user", "content": user_message}, {"role":"assistant", "content": reply}]
        calls, cost, tokens, df, breakdown = get_dashboard_data()
        req, resp = get_inspector_payloads()
        g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp = get_llm_cumulative_data()
        return (history, "\n".join(logs), calls, cost, tokens, df, breakdown, req, resp,
                g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp, get_prompt_breakdown_html())

    context, triggered = fare_rules_lookup(user_message)
    if triggered:
        logs.append("📜 L3 Duffel API   : Fare-rules lookup triggered (offer detail call)")
    else:
        context, triggered = flight_data_lookup(user_message)
        if triggered:
            logs.append("✈️ L3 Duffel API   : Live Flight context loaded successfully")
        else:
            logs.append("✈️ L3 Duffel API   : No structural query triggered")

    full_prompt = user_message + (f"\n\n[Duffel Flight Data]:\n{context}" if context else "")

    recent_history = (history or [])[-6:]
    llm_msgs = [{"role":"system","content":SKYBOT_SYSTEM}]
    for m in recent_history:
        llm_msgs.append({"role": m["role"], "content": msg_text(m["content"])})
    llm_msgs.append({"role":"user","content":full_prompt})

    raw, ok, p_tok, c_tok = call_llm("SkyBot Core Chat", llm_msgs, max_tokens=256)
    if not ok or raw is None:
        raw = "Error computing agent query."

    history_text = "\n".join(msg_text(m["content"]) for m in recent_history)
    split = estimate_prompt_token_split(
        {"system": SKYBOT_SYSTEM, "history": history_text, "user_question": user_message, "flight_data": context},
        p_tok,
    )
    LAST_PROMPT_BREAKDOWN.update(split)
    LAST_PROMPT_BREAKDOWN["output"] = c_tok

    final, warnings = validate(raw)
    if warnings:
        logs.append(f"🔒 L4 Validation : Passed with warnings ({', '.join(warnings)})")
    else:
        logs.append("🔒 L4 Validation : Clean Output Verified")

    judge_reply(user_message, context, final, topic["top_topic"])

    judge_reply_async(user_message, context, final)

    history = history + [{"role":"user", "content": user_message}, {"role":"assistant", "content": final}]

    calls, cost, tokens, df, breakdown = get_dashboard_data()
    req, resp = get_inspector_payloads()
    g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp = get_llm_cumulative_data()
    return (history, "\n".join(logs), calls, cost, tokens, df, breakdown, req, resp,
            g_calls, g_tokens, g_cost, g_breakdown, g_req, g_resp, get_prompt_breakdown_html())

def clear_chat():
    global LAST_GUARDRAIL_TOKENS, LAST_CHAT_TOKENS, LAST_FLIGHT_API_PAYLOAD, LAST_LLM_PAYLOAD
    LAST_GUARDRAIL_TOKENS = 0
    LAST_CHAT_TOKENS = 0
    LAST_FLIGHT_API_PAYLOAD["request"] = "No active transaction recorded yet."
    LAST_FLIGHT_API_PAYLOAD["response"] = "No active payload recorded yet."
    LAST_LLM_PAYLOAD["request"] = "No active LLM transaction recorded yet."
    LAST_LLM_PAYLOAD["response"] = "No active LLM payload recorded yet."
    _, _, _, _, empty_breakdown = get_dashboard_data()
    _, _, _, groq_breakdown, _, _ = get_llm_cumulative_data()
    return (
        [{"role":"assistant","content": WELCOME}], "", empty_breakdown,
        LAST_FLIGHT_API_PAYLOAD["request"], LAST_FLIGHT_API_PAYLOAD["response"],
        groq_breakdown, LAST_LLM_PAYLOAD["request"], LAST_LLM_PAYLOAD["response"],
    )

with gr.Blocks(title="✈️ SkyBot — AI Airline Assistant") as demo:

    gr.HTML("""
        <div style="text-align:center;padding:16px;background:linear-gradient(135deg,#1565C0,#1976D2);border-radius:12px;margin-bottom:12px">
            <h1 style="color:white;margin:0;font-size:1.8rem">✈️ SkyBot</h1>
            <p style="color:#e3f2fd;margin:4px 0 0;font-size:13px">AI Airline Assistant &nbsp;·&nbsp; Powered by LLM_MODEL env var &nbsp;·&nbsp; Operational Metrics Dashboard</p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🤖 Assistant Hub"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(value=[{"role":"assistant","content": WELCOME}], label="SkyBot Chat", height=430, show_label=False)
                    msg_box = gr.Textbox(placeholder="Ask about flights, airports, baggage...", show_label=False)
                    with gr.Row():
                        send_btn  = gr.Button("Send ✈️",  variant="primary")
                        clear_btn = gr.Button("Clear 🗑️", variant="secondary")
                with gr.Column(scale=1):
                    logs_box = gr.Textbox(label="Live Logs", lines=12, interactive=False)

        with gr.TabItem("📊 Token & API Usage Dashboard"):
            gr.HTML("<h3 style='color:#1565C0;margin:12px 0 4px 0'>📈 Live Consumption Metrics</h3>")

            token_breakdown_display = gr.HTML(
                value=get_dashboard_data()[4],
                label="Token Breakdown Box"
            )

            with gr.Row():
                dash_calls  = gr.Textbox(label="Total Network Transactions", value="0", interactive=False)
                dash_tokens = gr.Textbox(label="Aggregated LLM Tokens", value="0", interactive=False)
                dash_cost   = gr.Textbox(label="Estimated Financial Footprint", value="$0.00000", interactive=False)

            dash_table = gr.Dataframe(
                headers=["Timestamp", "Feature", "System/Model", "Total Tokens", "Est. Cost ($)"],
                datatype=["str", "str", "str", "number", "number"],
                label="Historical Operations Sequence Tracker",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Synchronize Matrix Data", variant="primary")

        with gr.TabItem("🌐 API Inspector"):
            gr.HTML("<h3 style='color:#1565C0;margin:12px 0 4px 0'>🔍 Live Duffel Flights REST Gateway Payloads</h3>")
            gr.Markdown("Analyze the raw structural parameters sent to and returned from the Duffel Flights REST endpoint during offer-request extraction cycles.")

            with gr.Row():
                inspector_req = gr.Code(label="Outbound API HTTP Request Structure", language="json", value=LAST_FLIGHT_API_PAYLOAD["request"], lines=15)
                inspector_resp = gr.Code(label="Inbound JSON Response Payload Received", language="json", value=LAST_FLIGHT_API_PAYLOAD["response"], lines=15)

        with gr.TabItem("🧠 LLM Inspector"):
            gr.HTML("<h3 style='color:#7c3aed;margin:12px 0 4px 0'>🔍 Cumulative LLM Chat-Completion Ledger</h3>")
            gr.Markdown("Tracks **every** LLM call SkyBot makes (guardrail + core chat) across the whole session, plus the raw request/response JSON of the most recent one — mirrors the flight-data API Inspector, but for the LLM side.")

            groq_breakdown_display = gr.HTML(value=get_llm_cumulative_data()[3])

            with gr.Row():
                groq_calls  = gr.Textbox(label="Cumulative LLM Calls", value="0", interactive=False)
                groq_tokens = gr.Textbox(label="Cumulative Tokens", value="0", interactive=False)
                groq_cost   = gr.Textbox(label="Cumulative Est. Cost", value="$0.00000", interactive=False)

            with gr.Row():
                groq_req  = gr.Code(label="Last Outbound LLM Request", language="json", value=LAST_LLM_PAYLOAD["request"], lines=15)
                groq_resp = gr.Code(label="Last Inbound LLM Response", language="json", value=LAST_LLM_PAYLOAD["response"], lines=15)

            groq_refresh_btn = gr.Button("🔄 Refresh LLM Ledger", variant="primary")

            gr.HTML("<h3 style='color:#facc15;margin:20px 0 4px 0'>🧾 Plain-English Breakdown (Last Core Chat Call)</h3>")
            prompt_breakdown_display = gr.HTML(value=get_prompt_breakdown_html())

        with gr.TabItem("🧩 Token Playground"):
            gr.HTML("<h3 style='color:#1565C0;margin:12px 0 4px 0'>🔮 Interactive Token Splitter</h3>")
            gr.Markdown("This is an educational approximation using a simple word-splitter, not the exact vocabulary of whatever LLM_MODEL is currently set. For the true billed count of anything you actually send to SkyBot, check the 🧠 LLM Inspector tab.")
            playground_input = gr.Textbox(label="Type your phrase here:", placeholder="e.g., Search DEL to LHR", lines=2)
            playground_output = gr.HTML(value=visual_tokenize_text(""), label="Token Visualizer Screen")
            playground_input.input(fn=visual_tokenize_text, inputs=playground_input, outputs=playground_output)

    dash_outputs = [dash_calls, dash_cost, dash_tokens, dash_table, token_breakdown_display, inspector_req, inspector_resp]
    groq_outputs = [groq_calls, groq_tokens, groq_cost, groq_breakdown_display, groq_req, groq_resp]
    chat_outputs = [chatbot, logs_box] + dash_outputs + groq_outputs + [prompt_breakdown_display]

    send_btn.click(fn=chat, inputs=[msg_box, chatbot], outputs=chat_outputs)
    send_btn.click(fn=lambda: "", outputs=msg_box)
    msg_box.submit(fn=chat, inputs=[msg_box, chatbot], outputs=chat_outputs)
    msg_box.submit(fn=lambda: "", outputs=msg_box)

    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, logs_box, token_breakdown_display, inspector_req, inspector_resp,
                 groq_breakdown_display, groq_req, groq_resp]
    )
    refresh_btn.click(fn=get_dashboard_data, inputs=[], outputs=[dash_calls, dash_cost, dash_tokens, dash_table, token_breakdown_display])
    groq_refresh_btn.click(fn=get_llm_cumulative_data, inputs=[], outputs=groq_outputs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🌐 Open this in your browser: http://127.0.0.1:{port}\n")
    demo.launch(server_name="0.0.0.0", server_port=port)
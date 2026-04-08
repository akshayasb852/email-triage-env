"""
Inference Script for Email Triage Environment
Follows the required [START], [STEP], [END] log format exactly.
"""

import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration - exactly as required by validator ──────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.environ["API_KEY"]
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://akshayasb-email-triage-openenv.hf.space")
MAX_STEPS = 20
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage"

print(f"[DEBUG] API_BASE_URL: {API_BASE_URL}", flush=True)
print(f"[DEBUG] MODEL_NAME: {MODEL_NAME}", flush=True)
print(f"[DEBUG] API_KEY set: {bool(API_KEY and API_KEY != 'dummy-key')}", flush=True)
print(f"[DEBUG] ENV_BASE_URL: {ENV_BASE_URL}", flush=True)

# ─── OpenAI Client - initialized at module level ───────────────────────────────
try:
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"[DEBUG] OpenAI client created successfully", flush=True)
except Exception as e:
    print(f"[DEBUG] Client creation error: {e}", flush=True)
    client = None

# ─── Logging ───────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── Agent Decision ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant.
Classify emails into: urgent, normal, or spam.
Assign priority 1-5 (1=highest).
Respond ONLY with JSON: {"label": "urgent|normal|spam", "priority": 1-5, "summary": "brief summary"}"""

def get_agent_action(observation: dict) -> dict:
    user_prompt = f"Subject: {observation.get('subject','')}\nFrom: {observation.get('sender','')}\nBody: {observation.get('body','')}\n\nClassify this email. Respond ONLY with JSON: {{\"label\": \"urgent|normal|spam\", \"priority\": 1-5, \"summary\": \"brief summary\"}}"
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=100,
    )
    text = completion.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    action = json.loads(text)
    if action.get("label") not in ["urgent", "normal", "spam"]:
        action["label"] = "normal"
    if not isinstance(action.get("priority"), int) or not (1 <= action["priority"] <= 5):
        action["priority"] = 3
    if not action.get("summary"):
        action["summary"] = "No summary"
    return action
# ─── Environment API Calls ─────────────────────────────────────────────────────
def env_reset(task_name: str) -> dict:
    try:
        response = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
        return response.json()
    except Exception as e:
        print(f"[DEBUG] Reset error: {e}", flush=True)
        return {"observation": {"subject": "", "sender": "", "body": "", "timestamp": ""}}

def env_step(task_name: str, action: dict) -> dict:
    try:
        response = requests.post(f"{ENV_BASE_URL}/step", json={"task_name": task_name, "label": action["label"], "priority": action["priority"], "summary": action["summary"]}, timeout=30)
        return response.json()
    except Exception as e:
        print(f"[DEBUG] Step error: {e}", flush=True)
        return {"reward": 0.0, "done": True, "observation": {}, "info": {}}

def env_score(task_name: str) -> float:
    try:
        response = requests.get(f"{ENV_BASE_URL}/score?task_name={task_name}", timeout=30)
        return response.json().get("score", 0.0)
    except Exception as e:
        print(f"[DEBUG] Score error: {e}", flush=True)
        return 0.0

# ─── Run One Task ──────────────────────────────────────────────────────────────
def run_task(task_name: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = env_reset(task_name)
        obs = reset_result.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            action = get_agent_action(obs)
            action_str = f"label={action['label']},priority={action['priority']}"
            step_result = env_step(task_name, action)
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            error = step_result.get("info", {}).get("error", None)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            if done:
                break
            obs = step_result.get("observation", obs)

        score = env_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = []
    print(f"[DEBUG] Starting Email Triage Inference", flush=True)

    for task_name in TASKS:
        print(f"\n[DEBUG] Running task: {task_name}", flush=True)
        result = run_task(task_name)
        results.append(result)

    print("\n[DEBUG] ===== FINAL RESULTS =====", flush=True)
    total_score = 0.0
    for r in results:
        print(f"[DEBUG] Task: {r['task']} | Score: {r['score']:.3f} | Success: {r['success']} | Steps: {r['steps']}", flush=True)
        total_score += r["score"]
    avg_score = total_score / len(results)
    print(f"[DEBUG] Average Score: {avg_score:.3f}", flush=True)

if __name__ == "__main__":
    main()
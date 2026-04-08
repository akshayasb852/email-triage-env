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

# ─── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("SPACE_URL") or "https://akshayasb-email-triage-openenv.hf.space"
MAX_STEPS = 20
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_client():
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] Client creation error: {e}", flush=True)
        return None

SYSTEM_PROMPT = """You are an expert email triage assistant.
Classify emails into: urgent, normal, or spam.
Assign priority 1-5 (1=highest).
Respond ONLY with JSON: {"label": "urgent|normal|spam", "priority": 1-5, "summary": "brief summary"}"""

def get_agent_action(client, observation: dict) -> dict:
    if client is None:
        return {"label": "normal", "priority": 3, "summary": "No client available"}
    try:
        user_prompt = f"Subject: {observation.get('subject','')}\nFrom: {observation.get('sender','')}\nBody: {observation.get('body','')}"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
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
    except Exception as e:
        print(f"[DEBUG] Agent error: {e}", flush=True)
        return {"label": "normal", "priority": 3, "summary": "Could not parse email"}

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

def run_task(client, task_name: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = env_reset(task_name)
        obs = reset_result.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            action = get_agent_action(client, obs)
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

def main():
    try:
        client = get_client()
    except Exception as e:
        print(f"[DEBUG] Failed to create client: {e}", flush=True)
        client = None

    results = []
    print(f"[DEBUG] Starting Email Triage Inference", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API Base: {API_BASE_URL}", flush=True)

    for task_name in TASKS:
        print(f"\n[DEBUG] Running task: {task_name}", flush=True)
        result = run_task(client, task_name)
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
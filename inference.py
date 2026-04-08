"""
Inference Script for Email Triage Environment
Follows the required [START], [STEP], [END] log format exactly.
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MAX_STEPS = 20
TEMPERATURE = 0.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage"

# ─── Logging ───────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── OpenAI Client ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ─── Agent Decision ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant.
Your job is to classify emails into exactly one of these categories:
- urgent: emails requiring immediate attention (server down, security breach, critical issues)
- normal: regular business emails (meetings, updates, requests)
- spam: unwanted/junk emails (prizes, scams, fake offers)

You must also assign a priority from 1 (highest) to 5 (lowest):
- Priority 1: Critical, needs immediate action
- Priority 2: High priority, needs action today
- Priority 3: Medium priority, normal business
- Priority 4: Low priority, can wait
- Priority 5: Spam/ignore

Respond ONLY with a valid JSON object in this exact format:
{"label": "urgent|normal|spam", "priority": 1-5, "summary": "one line summary"}

No other text. Just the JSON."""


def get_agent_action(client: OpenAI, observation: dict) -> dict:
    user_prompt = f"""Triage this email:

Subject: {observation['subject']}
From: {observation['sender']}
Body: {observation['body']}
Timestamp: {observation['timestamp']}

Respond with JSON only: {{"label": "urgent|normal|spam", "priority": 1-5, "summary": "brief summary"}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=100,
        )
        text = completion.choices[0].message.content.strip()

        # Clean up response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        action = json.loads(text)

        # Validate
        if action.get("label") not in ["urgent", "normal", "spam"]:
            action["label"] = "normal"
        if not isinstance(action.get("priority"), int) or not (1 <= action["priority"] <= 5):
            action["priority"] = 3
        if not action.get("summary"):
            action["summary"] = "No summary provided"

        return action

    except Exception as e:
        print(f"[DEBUG] Agent error: {e}", flush=True)
        return {"label": "normal", "priority": 3, "summary": "Could not parse email"}


# ─── Environment API Calls ─────────────────────────────────────────────────────
def env_reset(task_name: str) -> dict:
    response = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    return response.json()


def env_step(task_name: str, action: dict) -> dict:
    response = requests.post(
        f"{ENV_BASE_URL}/step",
        json={
            "task_name": task_name,
            "label": action["label"],
            "priority": action["priority"],
            "summary": action["summary"],
        },
        timeout=30,
    )
    return response.json()


def env_score(task_name: str) -> float:
    response = requests.get(f"{ENV_BASE_URL}/score?task_name={task_name}", timeout=30)
    return response.json().get("score", 0.0)


# ─── Run One Task ──────────────────────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_result = env_reset(task_name)
        obs = reset_result["observation"]

        for step in range(1, MAX_STEPS + 1):
            # Get agent action
            action = get_agent_action(client, obs)
            action_str = f"label={action['label']},priority={action['priority']}"

            # Step environment
            step_result = env_step(task_name, action)
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            error = step_result.get("info", {}).get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            # Update observation
            obs = step_result.get("observation", obs)

        # Get final score
        score = env_score(task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error = str(e)
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    client = get_client()
    results = []

    print(f"[DEBUG] Starting Email Triage Inference", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] API Base: {API_BASE_URL}", flush=True)

    for task_name in TASKS:
        print(f"\n[DEBUG] Running task: {task_name}", flush=True)
        result = run_task(client, task_name)
        results.append(result)

    # Summary
    print("\n[DEBUG] ===== FINAL RESULTS =====", flush=True)
    total_score = 0.0
    for r in results:
        print(f"[DEBUG] Task: {r['task']} | Score: {r['score']:.3f} | Success: {r['success']} | Steps: {r['steps']}", flush=True)
        total_score += r["score"]

    avg_score = total_score / len(results)
    print(f"[DEBUG] Average Score: {avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()